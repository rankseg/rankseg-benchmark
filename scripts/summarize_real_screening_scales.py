"""Audit and aggregate real-only, multi-scale CPU screening measurements."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
from rankseg_benchmark.common.paths import installed_rankseg_root
import statistics

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("real_scale_calibration", ROOT / "scripts/calibrate_screening_cpu.py")
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def performance(rows):
    if not rows:
        return None
    off = statistics.mean(r["timing"]["full_ms"] for r in rows)
    on = statistics.mean(r["timing"]["screened_ms"] for r in rows)
    return dict(samples=len(rows), full_ms=off, screened_ms=on, speedup=off/on,
                saved_us=(off-on)*1000,
                min_sample_speedup=min(r["timing"]["speedup"] for r in rows),
                max_sample_speedup=max(r["timing"]["speedup"] for r in rows),
                stable_wins=sum(r["timing"]["classification"] == "win" for r in rows),
                stable_losses=sum(r["timing"]["classification"] == "loss" for r in rows),
                slower_samples=sum(r["timing"]["speedup"] < 1 for r in rows))


def aggregate(rows):
    active = sum(r["counts"]["active_entries"] for r in rows)
    total = sum(r["counts"]["total_entries"] for r in rows)
    positive = sum(r["counts"]["forced_positive"] for r in rows)
    negative = sum(r["counts"]["forced_negative"] for r in rows)
    candidates = sum(r["counts"]["undecided"] for r in rows)
    pruned = sum(r["counts"]["class_pruned_entries"] for r in rows)
    if positive + negative + candidates != active or active + pruned != total:
        raise ValueError("Invalid screening partition")
    active_rows = [r for r in rows if r["counts"]["active_rows"] > 0]
    return dict(all_inputs=performance(rows), active_inputs=performance(active_rows),
                all_pruned_inputs=len(rows)-len(active_rows),
                active_entries=active, total_entries=total,
                forced_positive=positive, forced_negative=negative, candidates=candidates,
                screening_fraction_active=(positive+negative)/active if active else None,
                class_pruned_fraction_all=pruned/total,
                fallback_rows=sum(r["counts"]["fallback_rows"] for r in rows),
                changed_inputs=sum(r["different_pixels"] > 0 for r in rows),
                different_pixels=sum(r["different_pixels"] for r in rows),
                prediction_elements=sum(r["prediction_elements"] for r in rows))


def audit_report(report):
    if not report["complete"] or not report["real_only"] or report["sampling"] != "uniform" or report["device"] != "cpu":
        raise ValueError("Expected complete real-only uniform CPU run")
    profiles = {p["id"]: p for p in report["profiles"]}
    if len(profiles) != len(report["profiles"]) or any(p["kind"] != "real_derived" for p in profiles.values()):
        raise ValueError("Duplicate or synthetic profiles")
    expected = {(p["id"], d, threads) for p in profiles.values() for d in p["tested_dims"] for threads in report["threads"]}
    actual = set()
    for r in report["records"]:
        key = (r["profile"], r["dim"], r["threads"])
        if key in actual or key not in expected or r["kind"] != "real_derived":
            raise ValueError("Unexpected or duplicate configuration")
        actual.add(key)
        profile = profiles[r["profile"]]
        if r["dataset"] != profile["dataset"] or r["channels"] != profile["channels"] or r["batch"] != profile["batch"]:
            raise ValueError("Profile metadata mismatch")
        if len(r["rounds"]) != report["rounds"]:
            raise ValueError("Missing timing rounds")
        for round_values in r["rounds"]:
            if set(round_values) != {"full", "screened"}:
                raise ValueError("Missing paired method")
            for values in round_values.values():
                samples = values["samples_ms"]
                if len(samples) != report["repeats"] or not all(math.isfinite(t) and t > 0 for t in samples):
                    raise ValueError("Invalid timings")
                if values["median_ms"] != statistics.median(samples):
                    raise ValueError("Incorrect timing median")
        if r["timing"] != calibration.timing_summary(r["rounds"]):
            raise ValueError("Timing summary mismatch")
        if set(r["objectives"]) != {"full", "screened"}:
            raise ValueError("Missing objective method")
        for value in r["objectives"].values():
            eps = value["max_regret_eps"]
            if (not math.isfinite(eps) or eps < 0 or value["status"] not in ("passed", "failed")
                    or (value["status"] == "passed") != (eps <= 4)):
                raise ValueError("Invalid objective status")
        aggregate([r])
    if actual != expected:
        raise ValueError("Incomplete configuration coverage")


def summarize(reports):
    if not reports:
        raise ValueError("No reports")
    groups = defaultdict(list)
    input_hashes = defaultdict(set)
    keys = set()
    first = reports[0]
    def identities(report):
        return {p["id"]: (p["source_input_sha256"], p["draw_order_sha256"])
                for p in report["profiles"]}
    for report in reports:
        audit_report(report)
        if (report["source_hashes"] != first["source_hashes"] or report["torch"] != first["torch"]
                or report["cpu"] != first["cpu"] or identities(report) != identities(first)):
            raise ValueError("Source, environment or source samples differ between runs")
        for r in report["records"]:
            key = r["profile"], r["dim"], r["threads"]
            if key in keys:
                raise ValueError("Duplicate configuration across runs")
            keys.add(key)
            input_hashes[r["profile"], r["dim"]].add(r["input_sha256"])
            groups[r["dataset"], r["threads"], r["dim"]].append(r)
    if any(len(hashes) != 1 for hashes in input_hashes.values()):
        raise ValueError("Subsample differs between thread counts")
    all_rows = [r for rows in groups.values() for r in rows]
    result = dict(complete=True, cpu=first["cpu"], torch=first["torch"], profiles=len(first["profiles"]),
                  configurations=len(all_rows), source_hashes=first["source_hashes"],
                  objective_checks=2*len(all_rows),
                  objective_failures=sum(v["status"] != "passed" for r in all_rows for v in r["objectives"].values()),
                  max_regret_eps=max(v["max_regret_eps"] for r in all_rows for v in r["objectives"].values()),
                  results=[])
    for (dataset, threads, dim), rows in sorted(groups.items()):
        result["results"].append(dict(dataset=dataset, threads=threads, dim=dim, **aggregate(rows)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [json.loads((path / "results.json").read_text()) for path in args.inputs]
    result = summarize(reports)
    if calibration.bench.source_hashes(installed_rankseg_root()) != result["source_hashes"]:
        raise ValueError("Core code changed since measurements")
    result["inputs"] = [{"path": str(path.resolve()), "sha256": calibration.bench.acceptance.sha256(path / "results.json")}
                        for path in args.inputs]
    result["auditor_sha256"] = calibration.bench.acceptance.sha256(__file__)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("complete", "profiles", "configurations", "objective_checks",
                                            "objective_failures", "max_regret_eps")}, indent=2))


if __name__ == "__main__":
    main()
