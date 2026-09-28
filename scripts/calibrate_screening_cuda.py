"""Real-image CUDA scale comparison; dispatch overrides are OUTSIDE timing.

No synthetic performance inputs, inference, interpolation or silent OOM retry.
Compare public off, optimized full sort, forced screening and production on.
Use the same nested uniform real subsets as the CPU calibration, plus full inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import importlib.util
import json
import math
from pathlib import Path
from rankseg_benchmark.common.paths import installed_rankseg_root
import platform
import random
import statistics
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cuda_scale_sources", ROOT / "scripts/calibrate_screening_cpu.py")
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)
METHODS = ("full", "full_optimized", "screened", "production")
DIMS = [64, 256, 1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]


def control(method, oracle):
    if method not in METHODS:
        raise ValueError("Unknown method")
    return nullcontext() if method == "production" else oracle.decoder_control(method)


def planned_dims(profile, dims, include_full):
    total = profile["probs"].flatten(2).shape[-1]
    return sorted({d for d in dims if d <= total} | ({total} if include_full else set()))


def pair_summary(rounds, reference):
    return cpu.timing_summary([{"full": r[reference], "screened": r["screened"]} for r in rounds])


@torch.no_grad()
def measure(p, decoders, oracle, warmup, repeats, index):
    values, predictions, dispatch = {}, {}, {}
    names = METHODS[index % len(METHODS):] + METHODS[:index % len(METHODS)]
    for method in names:
        with control(method, oracle):
            def call():
                return decoders[method].predict(p)
            dispatch[method] = cpu.bench.acceptance.observed_screening_call(call)
            for _ in range(warmup):
                pred = call()
                del pred
            torch.cuda.synchronize()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            pred = call()
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - resident) / 2**20
            predictions[method] = pred.cpu()
            del pred
            values[method] = dict(peak_mib=peak, samples_ms=[])
    for repeat in range(repeats):
        shift = (repeat + index) % len(METHODS)
        for method in METHODS[shift:] + METHODS[:shift]:
            with control(method, oracle):
                # Controls, host-to-device copies and diagnostics are untimed.
                torch.cuda.synchronize()
                start = time.perf_counter()
                pred = decoders[method].predict(p)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                del pred
            values[method]["samples_ms"].append(elapsed)
    for v in values.values():
        v["median_ms"] = statistics.median(v["samples_ms"])
    return values, predictions, dispatch


def audit(report):
    if (not report["complete"] or report["device"] != "cuda" or not report["real_only"]
            or report["sampling"] != "uniform"):
        raise ValueError("Expected complete real-only CUDA run")
    profiles = {p["id"]: p for p in report["profiles"]}
    if len(profiles) != len(report["profiles"]) or any(p["kind"] != "real_derived" for p in profiles.values()):
        raise ValueError("Duplicate or non-real profile")
    expected = {(p["id"], d) for p in profiles.values() for d in p["tested_dims"]}
    seen = set()
    for r in report["records"]:
        key = r["profile"], r["dim"]
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate configuration")
        seen.add(key)
        if r["dataset"] != profiles[r["profile"]]["dataset"]:
            raise ValueError("Profile mismatch")
        if len(r["rounds"]) != report["rounds"]:
            raise ValueError("Missing rounds")
        for block in r["rounds"]:
            if set(block) != set(METHODS):
                raise ValueError("Missing method")
            for v in block.values():
                if (len(v["samples_ms"]) != report["repeats"]
                        or not all(math.isfinite(t) and t > 0 for t in v["samples_ms"])
                        or v["median_ms"] != statistics.median(v["samples_ms"])
                        or not math.isfinite(v["peak_mib"]) or v["peak_mib"] < 0):
                    raise ValueError("Invalid timing/memory")
        if set(r["objectives"]) != set(METHODS):
            raise ValueError("Missing objective")
        for v in r["objectives"].values():
            eps = v["max_regret_eps"]
            if (not math.isfinite(eps) or eps < 0 or v["status"] not in ("passed", "failed")
                    or (v["status"] == "passed") != (eps <= 4)):
                raise ValueError("Inconsistent objective status")
        counts = r["counts"]
        if (counts["forced_positive"] + counts["forced_negative"] + counts["undecided"] != counts["active_entries"]
                or counts["active_entries"] + counts["class_pruned_entries"] != counts["total_entries"]
                or counts["fallback_rows"] or counts["actual_sort_entries"] != counts["undecided"]):
            raise ValueError("Invalid candidate partition or unexpected retry")
        for ref in ("full", "full_optimized", "production"):
            if r["comparisons"][ref] != pair_summary(r["rounds"], ref):
                raise ValueError("Comparison mismatch")
        active = counts["active_rows"] > 0
        expected_dispatch = dict(full=0, full_optimized=0, screened=int(active),
                                 production=int(active and not r["production_bypass"]))
        if r["dispatch"] != expected_dispatch:
            raise ValueError("Wrong measured dispatch")
    if seen != expected:
        raise ValueError("Incomplete configurations")


def aggregate(rows):
    if not rows:
        return None
    times = {m: statistics.mean(statistics.median(rnd[m]["median_ms"] for rnd in r["rounds"])
                                for r in rows) for m in METHODS}
    peaks = {m: statistics.mean(max(rnd[m]["peak_mib"] for rnd in r["rounds"])
                                for r in rows) for m in METHODS}
    active = sum(r["counts"]["active_entries"] for r in rows)
    removed = sum(r["counts"]["forced_positive"] + r["counts"]["forced_negative"] for r in rows)
    return dict(samples=len(rows), mean_ms=times, mean_peak_mib=peaks,
                speedup_vs={m: times[m]/times["screened"] for m in METHODS if m != "screened"},
                memory_reduction_vs={m: 1-peaks["screened"]/peaks[m] if peaks[m] else None
                                     for m in METHODS if m != "screened"},
                stable_wins_vs_optimized=sum(r["comparisons"]["full_optimized"]["classification"] == "win" for r in rows),
                slower_vs_optimized=sum(r["comparisons"]["full_optimized"]["speedup"] < 1 for r in rows),
                screening_fraction_active=removed/active if active else None)


def summarize(report):
    audit(report)
    groups = defaultdict(list)
    for r in report["records"]:
        groups[r["dataset"], str(r["dim"])].append(r)
        if r["is_full_input"]:
            groups[r["dataset"], "full_input"].append(r)
    return dict(configurations=len(report["records"]), objective_checks=len(report["records"])*len(METHODS),
                objective_failures=sum(v["status"] != "passed" for r in report["records"] for v in r["objectives"].values()),
                max_regret_eps=max(v["max_regret_eps"] for r in report["records"] for v in r["objectives"].values()),
                results=[dict(dataset=ds, scale=scale, all_inputs=aggregate(rows),
                              active_inputs=aggregate([r for r in rows if r["counts"]["active_rows"]]))
                         for (ds, scale), rows in sorted(groups.items())])


@torch.no_grad()
def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU substitution")
    oracle, counts = cpu.bench.acceptance.load_helpers(installed_rankseg_root(), ROOT)
    from rankseg import RankSEG, _screening
    import triton
    if _screening._cuda_backend() is None:
        raise RuntimeError("Triton required for this matched-backend calibration")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True)
    frozen = cpu.bench.source_hashes(installed_rankseg_root())
    inputs, sources = cpu.load_profiles(args)
    jobs = []
    for i, p in enumerate(inputs):
        p["tested_dims"] = planned_dims(p, args.dims, args.include_full)
        p["unavailable_dims"] = [d for d in args.dims if d > p["probs"].flatten(2).shape[-1]]
        if not p["tested_dims"]:
            raise ValueError("No valid size for source")
        jobs.extend((i, d) for d in p["tested_dims"])
    production_mode = "auto" if getattr(args, "production_screening", "true") == "auto" else True
    report = dict(complete=False, device="cuda", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  production_screening=production_mode,
                  triton=triton.__version__, python=platform.python_version(), threads=args.threads,
                  rounds=args.rounds, repeats=args.repeats, warmup=args.warmup, dims=args.dims,
                  real_only=True, sampling="uniform", sampling_seed=args.sampling_seed,
                  samples_per_dataset=args.samples_per_dataset, include_full=args.include_full,
                  source_hashes=frozen, sources=sources, profiles=[cpu.profile_metadata(p) for p in inputs],
                  harness_sha256=cpu.bench.acceptance.sha256(__file__),
                  helper_sha256={str(p): cpu.bench.acceptance.sha256(p) for p in (
                      ROOT / "scripts/calibrate_screening_cpu.py", ROOT / "scripts/benchmark_screening_cpu.py",
                      ROOT / "scripts/benchmark_screening_acceptance.py",
                      ROOT / "scripts/nnunet/collect_screening_proportions.py",
                      ROOT / "rankseg_benchmark/nnunet/screening_benchmark.py")},
                  records=[])
    decoders = {(mode, m): RankSEG(metric="dice", solver="RMA", smooth=0, pruning_prob=.5,
                                  output_mode=mode,
                                  safe_screening=production_mode if m == "production" else m != "full")
                for mode in ("multilabel", "multiclass") for m in METHODS}
    records = {}
    with (args.output_dir / "timings.jsonl").open("x") as journal:
        for round_index in range(args.rounds):
            order = jobs.copy()
            random.Random(12345 + round_index).shuffle(order)
            for job_index, (i, dim) in enumerate(order):
                profile = inputs[i]
                key = profile["id"], dim
                host = cpu.make_input(profile, dim)
                before = cpu.bench.tensor_hash(host)
                p = host.cuda()
                if not torch.equal(host, p.cpu()):
                    raise AssertionError("Probability transfer changed values")
                timings, predictions, dispatch = measure(
                    p, {m: decoders[profile["mode"], m] for m in METHODS}, oracle,
                    args.warmup, args.repeats, job_index + round_index)
                hashes = {m: cpu.bench.tensor_hash(v) for m, v in predictions.items()}
                bypass = cpu.bench.acceptance.small_input_bypass(p, production_mode)
                equivalent = "full_optimized" if bypass else "screened"
                if not torch.equal(predictions["production"], predictions[equivalent]):
                    raise AssertionError("Production output differs from its actual route")
                if key not in records:
                    row = dict(profile=profile["id"], dataset=profile["dataset"], dim=dim,
                               is_full_input=dim == profile["probs"].flatten(2).shape[-1],
                               channels=p.shape[1], input_sha256=before, output_sha256=hashes,
                               production_bypass=bypass, dispatch=dispatch, rounds=[], objectives={},
                               different_elements={m: int((predictions[m] != predictions["screened"]).sum())
                                                   for m in METHODS if m != "screened"},
                               prediction_elements=predictions["screened"].numel())
                    optimum = oracle.bounded_binary_oracle(host)
                    for m in METHODS:
                        with control(m, oracle):
                            mask = decoders["multilabel", m].predict(p).cpu()
                        try:
                            diag = oracle.objective_regret(host, mask, optimum)
                            row["objectives"][m] = {**diag, "status": "passed"}
                        except oracle.ObjectiveBudgetExceeded as exc:
                            row["objectives"][m] = {**exc.diagnostics, "status": "failed"}
                        del mask
                    row["counts"] = counts.summarize([counts.screening_counts(p, list(range(p.shape[1])))])
                    records[key] = row
                elif (records[key]["input_sha256"] != before or records[key]["output_sha256"] != hashes
                      or records[key]["dispatch"] != dispatch):
                    raise AssertionError("Inputs, outputs or dispatch changed between rounds")
                if not torch.equal(host, p.cpu()) or cpu.bench.tensor_hash(host) != before:
                    raise AssertionError("Input modified")
                records[key]["rounds"].append(timings)
                journal.write(json.dumps(dict(profile=key[0], dim=dim, round=round_index, methods=timings),
                                         allow_nan=False) + "\n")
                del p, host, predictions
            journal.flush()
    if cpu.bench.source_hashes(installed_rankseg_root()) != frozen:
        raise AssertionError("Core changed during measurements")
    for key in sorted(records):
        row = records[key]
        row["comparisons"] = {m: pair_summary(row["rounds"], m) for m in METHODS if m != "screened"}
        report["records"].append(row)
    report["complete"] = True
    summary = summarize(report)
    for name, value in (("results.json", report), ("summary.json", summary)):
        with (args.output_dir / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-screening", choices=("true", "auto"), default="true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=DIMS)
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument("--sampling-seed", type=int, default=20260925)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--natural-cache", type=Path, default=ROOT / ".cache/screening-1884f076-first100")
    parser.add_argument("--kits-root", type=Path, default=Path(
        "/home/ben/.cache/huggingface/hub/datasets--ZixunWang--rankseg-benchmark/snapshots") / cpu.bench.acceptance.REVISION)
    args = parser.parse_args()
    if (min(args.dims + [args.samples_per_dataset, args.threads, args.rounds, args.repeats]) < 1
            or args.warmup < 0 or args.samples_per_dataset % 5 or args.dims != sorted(set(args.dims))):
        parser.error("Invalid sizes/counts; sample count must be a multiple of five")
    args.real_only, args.sampling = True, "uniform"
    run(args)


if __name__ == "__main__":
    main()
