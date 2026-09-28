"""Read-only audit of complete CPU comparison results and sampling coverage."""
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
spec = importlib.util.spec_from_file_location("cpu_audit_helpers", ROOT / "scripts/benchmark_screening_cpu.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def audit(directory):
    report = json.loads((directory / "summary.json").read_text())
    if report["complete"] is not True or report["device"] != "cpu":
        raise ValueError("Incomplete or non-CPU results")
    _, counts = bench.acceptance.load_helpers(installed_rankseg_root(), ROOT)
    if bench.source_hashes(installed_rankseg_root()) != report["source_hashes"]:
        raise ValueError("RankSEG source differs from the measured version")
    groups = defaultdict(list)
    fingerprints = defaultdict(set)
    for line in (directory / "records.jsonl").read_text().splitlines():
        row = json.loads(line)
        groups[row["dataset"], row["threads"]].append(row)
        fingerprints[row["dataset"], row["fold"], row["source_row"]].add(
            (row["input_sha256"], row["label_sha256"]))
        for method, value in row["methods"].items():
            samples = value["samples_ms"]
            if len(samples) != report["repeats"] or not all(math.isfinite(t) and t > 0 for t in samples):
                raise ValueError("Invalid timing samples")
            if statistics.median(samples) != value["median_ms"]:
                raise ValueError("Incorrect median")
            score = bench.acceptance.unit_scores(value["counts"], row["dataset"] == "kits")
            if score != value["scores"]:
                raise ValueError("Incorrect independent metrics")
            if method != "argmax":
                diag = value["objective"]
                if not math.isfinite(diag["max_regret_eps"]) or diag["max_regret_eps"] < 0:
                    raise ValueError("Nonfinite objective")
                if (diag["status"] == "passed") != (diag["max_regret_eps"] <= 4):
                    raise ValueError("Objective budget misreported")
        for counts_row in row["screening_counts"]["rows"]:
            if counts_row["forced_positive"] + counts_row["forced_negative"] + counts_row["undecided"] != counts_row["active_entries"]:
                raise ValueError("Certificate partition mismatch")
            if counts_row["active_entries"] + counts_row["class_pruned_entries"] != counts_row["total_entries"]:
                raise ValueError("Pruning partition mismatch")
        active = any(r["active_rows"] for r in row["screening_counts"]["rows"])
        if row["observed_screening_calls"] != int(active):
            raise ValueError("Unexpected dispatch")
    if any(len(values) != 1 for values in fingerprints.values()):
        raise ValueError("Input or labels changed across thread counts")
    if set(groups) != {(r["dataset"], r["threads"]) for r in report["results"]}:
        raise ValueError("Missing or extra dataset/thread group")
    for entry in report["results"]:
        rows = groups[entry["dataset"], entry["threads"]]
        expected = [(source["fold"], i) for source, indices in zip(entry["sources"], entry["selected_rows"])
                    for i in indices]
        if [(r["fold"], r["source_row"]) for r in rows] != expected:
            raise ValueError("Sample coverage/order mismatch")
        if len(expected) != len(set(expected)):
            raise ValueError("Duplicate samples")
        if bench.summarize(rows, counts) != entry["summary"]:
            raise ValueError("Aggregate summary mismatch")
    return report, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report, groups = audit(args.directory)
    total = sum(len(rows) for rows in groups.values())
    passed = sum(row["methods"][m]["objective"]["status"] == "passed"
                 for rows in groups.values() for row in rows for m in ("full", "screened"))
    print(json.dumps(dict(audit="passed", dataset_thread_groups=len(report["results"]),
                          sample_thread_pairs=total, objective_passed=passed,
                          objective_failed=2 * total - passed), indent=2))


if __name__ == "__main__":
    main()
