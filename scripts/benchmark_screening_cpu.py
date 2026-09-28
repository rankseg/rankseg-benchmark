"""Offline, paired CPU screening comparison; no production dispatch overrides.

Uses pinned mini-benchmark caches and the independent acceptance oracles. Natural
images are fixed prefixes; KiTS slices are uniformly spaced within each fold.
KiTS metrics here are sampled-slice means, NOT full case/fold or 3D volume scores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from rankseg_benchmark.common.paths import installed_rankseg_root
import platform
import statistics
import sys
import time

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "cpu_acceptance_helpers", ROOT / "scripts/benchmark_screening_acceptance.py")
acceptance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acceptance)
METHODS = ("argmax", "full", "screened")


def selected_rows(total, count, spaced=False):
    if total < 1 or count < 1:
        raise ValueError("Row counts must be positive")
    count = min(total, count)
    if not spaced or count == 1:
        return list(range(count))
    return [i * (total - 1) // (count - 1) for i in range(count)]


def selected_samples(spec, source, indices):
    """Decode only selected rows, while streaming the source in file order."""
    from rankseg_benchmark.datasets import _decode_probs, _decode_label
    if not indices or indices != sorted(set(indices)) or not 0 <= indices[0] <= indices[-1] < source["rows"]:
        raise ValueError("Invalid sample indices")
    wanted = set(indices)
    for index, batch in enumerate(pq.ParquetFile(source["path"]).iter_batches(batch_size=1)):
        if index in wanted:
            row = batch.to_pylist()[0]
            p = _decode_probs(row["probs"], num_classes=spec.num_classes)
            y = _decode_label(row["label"], num_classes=spec.num_classes, output_mode=spec.output_mode)
            case = row.get(spec.case_id_key) if spec.case_id_key else None
            if p.dtype != torch.float32 or p.shape[1:] != y.shape or (spec.eval_unit == "case" and case is None):
                raise ValueError("Invalid cached input")
            yield p[None], y[None], dict(source_row=index, fold=source["fold"], case_id=case)
            wanted.remove(index)
            if not wanted:
                return
    raise ValueError("Cache ended before all selected rows were read")


@torch.no_grad()
def measure(operations, warmup, repeats, index):
    if warmup < 0 or repeats < 1:
        raise ValueError("Invalid repetition counts")
    names = tuple(operations)
    values = {name: {"samples_ms": []} for name in names}
    for _ in range(warmup):
        for call in operations.values():
            prediction = call()
            del prediction
    for repeat in range(repeats):
        shift = (index + repeat) % len(names)
        for name in names[shift:] + names[:shift]:
            start = time.perf_counter()
            prediction = operations[name]()
            elapsed = (time.perf_counter() - start) * 1000
            values[name]["samples_ms"].append(elapsed)
            del prediction
    # Outputs and diagnostics are collected outside the measured intervals.
    predictions = {name: call() for name, call in operations.items()}
    for value in values.values():
        value["median_ms"] = statistics.median(value["samples_ms"])
    return values, predictions


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def source_hashes(rankseg_path):
    return {str(p.relative_to(rankseg_path)): acceptance.sha256(p)
            for p in sorted((rankseg_path / "rankseg").rglob("*.py"))}


def summarize(records, counts_helper):
    methods = {}
    for method in METHODS:
        entries = [row["methods"][method] for row in records]
        scores = [entry["scores"] for entry in entries if entry["scores"] is not None]
        methods[method] = {
            "mean_median_ms": statistics.mean(entry["median_ms"] for entry in entries),
            **{key: statistics.mean(score[key] for score in scores) if scores else None
               for key in ("dice", "iou")},
        }
        if method != "argmax":
            methods[method].update(
                objective_passed=sum(e["objective"]["status"] == "passed" for e in entries),
                max_regret_eps=max(e["objective"]["max_regret_eps"] for e in entries))
    return {
        "samples": len(records), "methods": methods,
        "speedup": methods["full"]["mean_median_ms"] / methods["screened"]["mean_median_ms"],
        "screened_faster_samples": sum(r["methods"]["screened"]["median_ms"] <
                                       r["methods"]["full"]["median_ms"] for r in records),
        "changed_samples": sum(r["different_pixels"] > 0 for r in records),
        "different_pixels": sum(r["different_pixels"] for r in records),
        "total_pixels": sum(r["prediction_pixels"] for r in records),
        "worst_dice_delta_pp": min(100 * (r["methods"]["screened"]["scores"]["dice"] -
                                         r["methods"]["full"]["scores"]["dice"])
                                   for r in records if r["methods"]["full"]["scores"] is not None),
        "screening": counts_helper.summarize([r["screening_counts"] for r in records]),
    }


@torch.no_grad()
def run(args):
    oracle, counts_helper = acceptance.load_helpers(args.rankseg_path, args.medical_root)
    from rankseg_benchmark.datasets import REGISTRY
    from rankseg_benchmark.runner import _rankseg_channels, _rankseg_output_mode, _rankseg_predict
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frozen = source_hashes(args.rankseg_path)
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    report = dict(complete=False, cpu=cpu, torch=torch.__version__, python=platform.python_version(),
                  device="cpu", threads=args.threads, warmup=args.warmup, repeats=args.repeats,
                  screening_mode=getattr(args, "screening_mode", "true"),
                  source_hashes=frozen, harness_sha256=acceptance.sha256(__file__),
                  helper_sha256={str(p): acceptance.sha256(p) for p in (
                      ROOT / "scripts/benchmark_screening_acceptance.py",
                      args.medical_root / "rankseg_benchmark/nnunet/screening_benchmark.py",
                      args.medical_root / "scripts/nnunet/collect_screening_proportions.py")},
                  timing="mean of per-sample median wall time; routing included; serial runs",
                  objective="Dice RMA smooth=0 pruning_prob=0.5; 4 float32 eps budget",
                  results=[])
    with (args.output_dir / "records.jsonl").open("x") as journal:
        for dataset in args.datasets:
            spec = REGISTRY[dataset]
            medical = spec.eval_unit == "case"
            mode = _rankseg_output_mode(spec, "RMA")
            channels = _rankseg_channels(spec, mode)
            metadata, sources = acceptance.cache_sources(dataset, args.natural_cache, args.kits_root)
            chosen = [selected_rows(s["rows"], args.kits_per_fold if medical else args.natural_count,
                                    spaced=medical) for s in sources]
            for threads in args.threads:
                torch.set_num_threads(threads)
                records = []
                screening_mode = "auto" if getattr(args, "screening_mode", "true") == "auto" else True
                decoders = {m: acceptance.make_decoder(m, mode, oracle, screening_mode) for m in METHODS[1:]}
                binary = {m: acceptance.make_decoder(m, "multilabel", oracle, screening_mode) for m in METHODS[1:]}
                for source, indices in zip(sources, chosen):
                    for probs, labels, identity in selected_samples(spec, source, indices):
                        before = tensor_hash(probs)
                        operations = {m: (lambda d=d: _rankseg_predict(d, probs, spec=spec,
                                                                      rankseg_output_mode=mode))
                                      for m, d in decoders.items()}
                        operations = {"argmax": lambda: probs.argmax(1), **operations}
                        values, predictions = measure(operations, args.warmup, args.repeats, len(records))
                        row = dict(dataset=dataset, threads=threads, **identity, shape=list(probs.shape),
                                   input_sha256=before, label_sha256=tensor_hash(labels), methods=values,
                                   different_pixels=int((predictions["full"] != predictions["screened"]).sum()),
                                   prediction_pixels=predictions["full"].numel())
                        for method, pred in predictions.items():
                            value = acceptance.confusion(pred, labels, spec.num_classes, spec.ignore_index, medical)
                            values[method].update(counts=value, scores=acceptance.unit_scores(value, medical))
                        del predictions
                        selected = probs[:, channels]
                        row["screening_counts"] = counts_helper.screening_counts(selected, channels)
                        calls = acceptance.observed_screening_call(operations["screened"])
                        if calls != int(selected.max().item() > .5):
                            raise AssertionError("Unexpected CPU screening dispatch")
                        row["observed_screening_calls"] = calls
                        optimum = oracle.bounded_binary_oracle(selected)
                        for method, decoder in binary.items():
                            mask = decoder.predict(selected)
                            try:
                                diag = oracle.objective_regret(selected, mask, optimum)
                                values[method]["objective"] = {**diag, "status": "passed"}
                            except oracle.ObjectiveBudgetExceeded as error:
                                values[method]["objective"] = {**error.diagnostics, "status": "failed"}
                            del mask
                        if tensor_hash(probs) != before:
                            raise AssertionError("Input modified")
                        if not records and threads == args.threads[0]:
                            torch.save({"probs": probs, "dataset": dataset, "identity": identity},
                                       args.output_dir / f"{dataset}-memory-input.pt")
                        records.append(row)
                        journal.write(json.dumps(row, allow_nan=False) + "\n")
                        journal.flush()
                        del selected, optimum, probs, labels, operations
                if len(records) != sum(map(len, chosen)):
                    raise AssertionError("Sample coverage mismatch")
                report["results"].append(dict(dataset=dataset, threads=threads,
                    source_metadata=metadata, sources=sources, selected_rows=chosen,
                    subset=True, rankseg_output_mode=mode, channels=channels,
                    metrics="sampled slice mean (not volume/case/fold mean)" if medical else "image mean",
                    summary=summarize(records, counts_helper)))
                (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(f"{dataset}, threads={threads}: {len(records)} samples complete", flush=True)
    if source_hashes(args.rankseg_path) != frozen:
        raise AssertionError("RankSEG source changed during benchmark")
    report["complete"] = True
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankseg-path", type=Path, default=installed_rankseg_root())
    parser.add_argument("--medical-root", type=Path, default=ROOT)
    parser.add_argument("--natural-cache", type=Path, default=ROOT / ".cache/screening-1884f076-first100")
    parser.add_argument("--kits-root", type=Path, default=Path(
        "/home/ben/.cache/huggingface/hub/datasets--ZixunWang--rankseg-benchmark/snapshots") / acceptance.REVISION)
    parser.add_argument("--datasets", nargs="+", choices=("pascal_voc", "cityscapes", "ade20k", "kits"),
                        default=["pascal_voc", "cityscapes", "ade20k", "kits"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--natural-count", type=int, default=10)
    parser.add_argument("--kits-per-fold", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--screening-mode", choices=("true", "auto"), default="true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads + [args.natural_count, args.kits_per_fold, args.repeats]) < 1 or args.warmup < 0:
        parser.error("Counts/threads must be positive; warmup must be nonnegative")
    if len(set(args.threads)) != len(args.threads) or len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate threads/datasets")
    run(args)


if __name__ == "__main__":
    main()
