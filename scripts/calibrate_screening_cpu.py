"""Measure CPU screening crossovers without implementing an auto threshold.

Real-cache inputs use evenly spaced or seeded uniform-without-replacement spatial
samples, retaining original probabilities (no interpolation/renormalization).
These are derived workloads, not whole-image quality measurements. Synthetic
profiles are optional; --real-only excludes them entirely. All timing jobs are
serial, with shuffled order across three rounds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from rankseg_benchmark.common.paths import installed_rankseg_root
import platform
import random
import statistics

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibration_cpu_helpers", ROOT / "scripts/benchmark_screening_cpu.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
spec = importlib.util.spec_from_file_location("calibration_profiles", ROOT / "scripts/benchmark_screening_cpu_synthetic.py")
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)
DIMS = [256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536]
SHAPES = [(1, 1), (1, 3), (1, 16), (4, 3)]


def spatial_subset(probs, dim):
    flat = probs.flatten(2)
    total = flat.shape[-1]
    if not 1 <= dim <= total:
        raise ValueError("Requested spatial size is outside the source")
    indices = (torch.arange(dim, dtype=torch.int64) * (total - 1) // (dim - 1)
               if dim > 1 else torch.zeros(1, dtype=torch.int64))
    return flat.index_select(-1, indices)


def make_input(profile, dim):
    if profile["kind"] == "real_derived":
        if "draw_order" in profile:
            flat = profile["probs"].flatten(2)
            if not 1 <= dim <= flat.shape[-1]:
                raise ValueError("Requested spatial size is outside the source")
            # Nested uniform subsets, restored to spatial order. Every channel
            # uses the same positions; extraction/sorting is outside timing.
            indices = profile["draw_order"][:dim].sort().values
            return flat.index_select(-1, indices)
        return spatial_subset(profile["probs"], dim)
    b, c = profile["batch"], profile["channels"]
    return profiles.make_probabilities(profile["scenario"], b * c, dim, profile["seed"]).reshape(b, c, dim)


def timing_summary(rounds):
    ratios = [r["full"]["median_ms"] / r["screened"]["median_ms"] for r in rounds]
    full = statistics.median(r["full"]["median_ms"] for r in rounds)
    screened = statistics.median(r["screened"]["median_ms"] for r in rounds)
    # A reporting noise margin only, not an inference/dispatch threshold or CI.
    classification = "win" if min(ratios) > 1.05 else "loss" if max(ratios) < .95 else "mixed_or_near"
    return dict(full_ms=full, screened_ms=screened, speedup=full / screened,
                saved_us=(full - screened) * 1000, round_speedups=ratios,
                classification=classification)


def stable_suffix(rows, dims):
    """Smallest *tested* D whose entire tested tail is classified as winning."""
    by_dim = defaultdict(list)
    for row in rows:
        by_dim[row["dim"]].append(row["timing"]["classification"])
    if set(by_dim) != set(dims):
        raise ValueError("Incomplete dimension coverage")
    for index, dim in enumerate(dims):
        if all(all(value == "win" for value in by_dim[d]) for d in dims[index:]):
            return dim
    return None


def selected_source_rows(dataset, entries, real_only, samples_per_dataset):
    if not real_only:
        return [(entries[0], [0, 49])] if dataset != "kits" else [(entries[0], [0]), (entries[4], [0])]
    if samples_per_dataset < 1:
        raise ValueError("Sample count must be positive")
    if dataset == "kits":
        if len(entries) != 5 or samples_per_dataset % 5:
            raise ValueError("KiTS sample count must be divisible by five for equal fold coverage")
        count = samples_per_dataset // 5
        if any(source["rows"] < count for source in entries):
            raise ValueError("Not enough KiTS rows for the requested sample count")
        return [(source, stratum_centers(source["rows"], count)) for source in entries]
    if len(entries) != 1 or entries[0]["rows"] < samples_per_dataset:
        raise ValueError("Not enough natural-image rows for the requested sample count")
    return [(entries[0], stratum_centers(entries[0]["rows"], samples_per_dataset))]


def stratum_centers(total, count):
    """Uniform file-position coverage without emphasizing first/last slices."""
    if not 1 <= count <= total:
        raise ValueError("Invalid stratum count")
    return [(2 * i + 1) * total // (2 * count) for i in range(count)]


def sampling_order(profile_id, total, seed):
    # Avoid Python's process-randomized hash. Case identity and seed determine
    # the exact permutation, independently of thread/job/profile ordering.
    digest = hashlib.sha256(f"{seed}:{profile_id}".encode()).digest()
    derived_seed = int.from_bytes(digest[:8], "little") % (2**63)
    order = torch.randperm(total, generator=torch.Generator().manual_seed(derived_seed))
    return order, derived_seed


def profile_metadata(profile):
    return {key: value for key, value in profile.items() if key not in ("probs", "draw_order")}


def load_profiles(args):
    from rankseg_benchmark.datasets import REGISTRY
    from rankseg_benchmark.runner import _rankseg_channels, _rankseg_output_mode
    result, sources = [], {}
    for b, c in ([] if args.real_only else SHAPES):
        for scenario in profiles.SCENARIOS:
            for seed in (3401, 7819):
                result.append(dict(id=f"synthetic-b{b}-c{c}-{scenario}-seed{seed}", kind="synthetic",
                                   batch=b, channels=c, scenario=scenario, seed=seed,
                                   mode="multilabel" if c == 1 else "multiclass"))
    # Predeclared indices, independent of predictions, labels or performance.
    for dataset in ("pascal_voc", "cityscapes", "ade20k", "kits"):
        dataset_spec = REGISTRY[dataset]
        mode = _rankseg_output_mode(dataset_spec, "RMA")
        channels = _rankseg_channels(dataset_spec, mode)
        metadata, entries = bench.acceptance.cache_sources(dataset, args.natural_cache, args.kits_root)
        selected_sources = selected_source_rows(dataset, entries, args.real_only, args.samples_per_dataset)
        sources[dataset] = {"metadata": metadata, "files": entries}
        for source, indices in selected_sources:
            for p, _, identity in bench.selected_samples(dataset_spec, source, indices):
                selected = p[:, channels].contiguous()
                profile = dict(id=f"{dataset}-fold{identity['fold']}-row{identity['source_row']}",
                               kind="real_derived", dataset=dataset, identity=identity,
                               batch=1, channels=len(channels), mode=mode,
                               original_shape=list(p.shape), source_input_sha256=bench.tensor_hash(p),
                               sampling=args.sampling, probs=selected)
                if args.sampling == "uniform":
                    order, seed = sampling_order(profile["id"], selected.flatten(2).shape[-1], args.sampling_seed)
                    profile.update(draw_order=order, sampling_seed=seed,
                                   draw_order_sha256=bench.tensor_hash(order))
                result.append(profile)
    return result, sources


@torch.no_grad()
def run(args):
    oracle, counts = bench.acceptance.load_helpers(installed_rankseg_root(), ROOT)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    # Avoid the environment's default large thread pool during cache setup.
    # Each timing block sets its explicitly requested thread count below.
    torch.set_num_threads(args.threads[0])
    frozen = bench.source_hashes(installed_rankseg_root())
    inputs, sources = load_profiles(args)
    jobs = []
    for index, profile in enumerate(inputs):
        available = (profile["probs"].flatten(2).shape[-1] if profile["kind"] == "real_derived"
                     else max(args.dims))
        profile["tested_dims"] = [d for d in args.dims if d <= available]
        profile["unavailable_dims"] = [d for d in args.dims if d > available]
        if not profile["tested_dims"]:
            raise ValueError(f"Every requested size exceeds source {profile['id']}")
        jobs.extend((index, dim) for dim in profile["tested_dims"])
    report = dict(complete=False, device="cpu", torch=torch.__version__, python=platform.python_version(),
                  cpu=next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")),
                  dims=args.dims, threads=args.threads, rounds=args.rounds, repeats=args.repeats,
                  warmup=args.warmup, source_hashes=frozen, sources=sources,
                  profiles=[profile_metadata(p) for p in inputs],
                  real_only=args.real_only, sampling=args.sampling, sampling_seed=args.sampling_seed,
                  samples_per_dataset=args.samples_per_dataset if args.real_only else 2,
                  harness_sha256=bench.acceptance.sha256(__file__),
                  helper_sha256={str(path): bench.acceptance.sha256(path) for path in (
                      ROOT / "scripts/benchmark_screening_cpu.py",
                      ROOT / "scripts/benchmark_screening_acceptance.py",
                      ROOT / "rankseg_benchmark/nnunet/screening_benchmark.py",
                      ROOT / "scripts/nnunet/collect_screening_proportions.py")},
                  classification="win: every round >1.05x; loss: every round <0.95x; otherwise mixed/near",
                  timing="median of round medians; public predict only; no routing copy or diagnostics",
                  records=[])
    records, optima, fingerprints = {}, {}, {}
    decoders = {(mode, method): bench.acceptance.make_decoder(method, mode, oracle)
                for mode in ("multiclass", "multilabel") for method in ("full", "screened")}
    with (args.output_dir / "timings.jsonl").open("x") as journal:
        for round_index in range(args.rounds):
            # Rotate thread-block order and deterministically shuffle workloads.
            threads_order = args.threads[round_index % len(args.threads):] + args.threads[:round_index % len(args.threads)]
            order = jobs.copy()
            random.Random(12345 + round_index).shuffle(order)
            for threads in threads_order:
                torch.set_num_threads(threads)
                for job_index, (profile_index, dim) in enumerate(order):
                    profile = inputs[profile_index]
                    key = (profile["id"], dim, threads)
                    p = make_input(profile, dim)
                    before = bench.tensor_hash(p)
                    input_key = (profile["id"], dim)
                    if input_key in fingerprints and fingerprints[input_key] != before:
                        raise AssertionError("Input changed across rounds or thread counts")
                    fingerprints[input_key] = before
                    calls = {m: lambda d=decoders[profile["mode"], m]: d.predict(p) for m in ("full", "screened")}
                    timings, predictions = bench.measure(calls, args.warmup, args.repeats, job_index + round_index)
                    result_hashes = {m: bench.tensor_hash(pred) for m, pred in predictions.items()}
                    if key not in records:
                        row = dict(profile=profile["id"], kind=profile["kind"], batch=profile["batch"],
                                   channels=profile["channels"], mode=profile["mode"],
                                   scenario=profile.get("scenario"), dataset=profile.get("dataset"),
                                   dim=dim, threads=threads, input_sha256=before, output_sha256=result_hashes,
                                   different_pixels=int((predictions["full"] != predictions["screened"]).sum()),
                                   prediction_elements=predictions["full"].numel(), rounds=[], objectives={})
                        if input_key not in optima:
                            optima[input_key] = oracle.bounded_binary_oracle(p)
                        for method in calls:
                            mask = decoders["multilabel", method].predict(p)
                            try:
                                diag = oracle.objective_regret(p, mask, optima[input_key])
                                row["objectives"][method] = {**diag, "status": "passed"}
                            except oracle.ObjectiveBudgetExceeded as error:
                                row["objectives"][method] = {**error.diagnostics, "status": "failed"}
                            del mask
                        row["counts"] = counts.summarize([counts.screening_counts(p[b:b+1], list(range(p.shape[1])))
                                                         for b in range(p.shape[0])])
                        records[key] = row
                    elif records[key]["output_sha256"] != result_hashes:
                        raise AssertionError("Output changed between repeated rounds on fixed CPU threads")
                    if bench.tensor_hash(p) != before:
                        raise AssertionError("Probability input modified")
                    records[key]["rounds"].append(timings)
                    journal.write(json.dumps(dict(profile=key[0], dim=dim, threads=threads,
                                                  round=round_index, methods=timings), allow_nan=False) + "\n")
                    del predictions, p, calls
                journal.flush()
    if bench.source_hashes(installed_rankseg_root()) != frozen:
        raise AssertionError("RankSEG source changed")
    for key in sorted(records):
        row = records[key]
        if len(row["rounds"]) != args.rounds:
            raise AssertionError("Missing rounds")
        row["timing"] = timing_summary(row["rounds"])
        report["records"].append(row)
    expected = len(jobs) * len(args.threads)
    if len(records) != expected:
        raise AssertionError("Missing configurations")
    report["complete"] = True
    with (args.output_dir / "results.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(configurations=len(records), timing_blocks=len(records)*args.rounds,
                          objectives_passed=sum(d["status"] == "passed" for r in records.values()
                                                for d in r["objectives"].values())), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=DIMS)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--real-only", action="store_true", help="Exclude every synthetic profile")
    parser.add_argument("--samples-per-dataset", type=int, default=10,
                        help="Real-only mode: evenly spaced cache rows; KiTS count divisible by five")
    parser.add_argument("--sampling", choices=("spaced", "uniform"), default="spaced")
    parser.add_argument("--sampling-seed", type=int, default=20260925)
    parser.add_argument("--natural-cache", type=Path, default=ROOT / ".cache/screening-1884f076-first100")
    parser.add_argument("--kits-root", type=Path, default=Path(
        "/home/ben/.cache/huggingface/hub/datasets--ZixunWang--rankseg-benchmark/snapshots") / bench.acceptance.REVISION)
    args = parser.parse_args()
    if min(args.dims + args.threads + [args.rounds, args.repeats]) < 1 or args.warmup < 0:
        parser.error("Invalid measurement counts")
    if args.dims != sorted(set(args.dims)) or len(args.threads) != len(set(args.threads)):
        parser.error("Dimensions must be increasing and unique; thread counts must be unique")
    if args.real_only and (args.samples_per_dataset < 1 or args.samples_per_dataset % 5):
        parser.error("Real-only samples-per-dataset must be a positive multiple of five")
    run(args)


if __name__ == "__main__":
    main()
