"""Offline paired acceptance on pinned mini-benchmark probability caches.

Natural-image prefixes are explicitly subsets. KiTS uses all cached slices and
the repository's slice -> case -> fold metric, not whole-volume Dice. No inference.
Diagnostics, independent metrics and exhaustive float64 oracles are untimed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1884f0766268cdc62730e696f24dcc913d551b35"
METHODS = ("argmax", "full", "full_optimized", "screened", "screened_forced")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_helpers(rankseg_path, medical_root):
    sys.path[:0] = [str(rankseg_path), str(ROOT), str(medical_root)]
    import rankseg
    if Path(rankseg.__file__).resolve().parent != rankseg_path.resolve() / "rankseg":
        raise RuntimeError("Wrong RankSEG checkout loaded")
    from rankseg_benchmark.nnunet import screening_benchmark as oracle
    spec = importlib.util.spec_from_file_location(
        "acceptance_counts", medical_root / "scripts/nnunet/collect_screening_proportions.py")
    counts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(counts)
    return oracle, counts


def make_decoder(method, mode, oracle, screening_mode=True):
    from rankseg import RankSEG
    if method in ("full", "screened"):
        # Normal on/off paths do not pay for any private dispatch overrides.
        return RankSEG(metric="dice", solver="RMA", smooth=0, pruning_prob=.5,
                       output_mode=mode, safe_screening=screening_mode if method == "screened" else False)
    if method == "full_optimized":
        return oracle.ScreeningDecoder("full_optimized", mode)
    if method == "screened_forced":
        return oracle.ScreeningDecoder("screened", mode, production_dispatch=False)
    raise ValueError("Unknown RankSEG method")


def confusion(prediction, target, classes, ignore=255, medical=False):
    """Independent NumPy counts; never invoke the benchmark's accumulator here."""
    pred, label = np.asarray(prediction).reshape(-1), np.asarray(target).reshape(-1)
    if pred.shape != label.shape:
        raise ValueError("Prediction/target shape mismatch")
    keep = label != ignore if ignore is not None else np.ones(label.shape, dtype=bool)
    pred, label = pred[keep], label[keep]
    if medical:
        # Match _MedicalCaseMetric's existing label clamping, not the natural
        # image accumulator's rule. Decoding itself never sees the labels.
        label = np.clip(label, 0, classes - 1)
    else:
        # Hosted natural-image caches can use C itself for void/padding (e.g.
        # VOC label 21). ConfusionAccumulator excludes every out-of-range GT.
        valid = (label >= 0) & (label < classes)
        pred, label = pred[valid], label[valid]
    if ((pred < 0) | (pred >= classes)).any():
        raise ValueError("Invalid class indices")
    tp = np.bincount(label[pred == label].astype(np.int64), minlength=classes)
    pc = np.bincount(pred.astype(np.int64), minlength=classes)
    lc = np.bincount(label.astype(np.int64), minlength=classes)
    return dict(tp=tp.tolist(), fp=(pc - tp).tolist(), fn=(lc - tp).tolist())


def unit_scores(counts, medical=False):
    tp, fp, fn = (np.asarray(counts[k], dtype=np.float64) for k in ("tp", "fp", "fn"))
    if not (tp + fn).sum():
        return None  # All ignored pixels, not a perfect empty prediction.
    if medical:
        # Exactly the existing KiTS binary slice convention, including empty
        # foreground slices. No new epsilon or empty-class rule is introduced.
        active = (2 * tp + fp + fn) > 0
        if int(active.sum()) < 2:
            return {"dice": 1., "iou": 1.}
        return {"dice": float(2 * tp[1] / (2 * tp[1] + fp[1] + fn[1])),
                "iou": float(tp[1] / (tp[1] + fp[1] + fn[1]))}
    active = tp + fn > 0
    return {"dice": float(((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7))[active].mean()),
            "iou": float(((tp + 1e-7) / (tp + fp + fn + 1e-7))[active].mean())}


def independent_metrics(records, method, medical=False):
    if medical:
        groups = defaultdict(list)
        for row in records:
            key = row["fold"], row["case_id"]
            score = unit_scores(row["methods"][method]["counts"], True)
            groups[key]  # Preserve cases whose slices contain only ignored labels.
            if score is not None:
                groups[key].append(score)
        folds = defaultdict(list)
        for (fold, _), scores in groups.items():
            folds[fold].append({k: statistics.mean(s[k] for s in scores) if scores else 0.
                                for k in ("dice", "iou")})
        scores = [{k: statistics.mean(s[k] for s in cases) for k in ("dice", "iou")}
                  for cases in folds.values()]
    else:
        scores = [unit_scores(row["methods"][method]["counts"]) for row in records]
        scores = [s for s in scores if s is not None]
    return {k: statistics.mean(s[k] for s in scores) if scores else 0. for k in ("dice", "iou")}


def cache_sources(dataset, natural_cache, kits_root):
    if dataset == "kits":
        paths = [kits_root / f"kits/fold{fold}/test-00000.parquet" for fold in range(5)]
        metadata = {"selection": "all stored slices in all five folds", "revision": REVISION,
                    "complete_source": True}
        if kits_root.name != REVISION:
            raise ValueError("KiTS must use the pinned HF snapshot revision")
    else:
        paths = [natural_cache / f"{dataset}-first100.parquet"]
        source = paths[0].with_suffix(".source.json")
        metadata = json.loads(source.read_text())
        if (metadata["dataset"] != dataset or metadata["revision"] != REVISION
                or metadata["rows"] != 100):
            raise ValueError("Natural-image cache provenance mismatch")
        metadata = {**metadata, "source_metadata_sha256": sha256(source), "complete_source": False}
    entries = []
    for fold, path in enumerate(paths):
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows < 1 or (dataset != "kits" and rows != 100):
            raise ValueError("Unexpected cache row count")
        entries.append(dict(path=str(path.resolve()), sha256=sha256(path), rows=rows,
                            fold=fold if dataset == "kits" else None))
    return metadata, entries


def samples(spec, sources):
    from rankseg_benchmark.datasets import _decode_probs, _decode_label
    for source in sources:
        index = 0
        for batch in pq.ParquetFile(source["path"]).iter_batches(batch_size=1):
            for row in batch.to_pylist():
                p = _decode_probs(row["probs"], num_classes=spec.num_classes)
                y = _decode_label(row["label"], num_classes=spec.num_classes, output_mode=spec.output_mode)
                if p.dtype != torch.float32 or p.shape[1:] != y.shape:
                    raise ValueError("Wrong probability dtype or label shape")
                if spec.eval_unit == "case" and row.get(spec.case_id_key) is None:
                    raise ValueError("Missing KiTS case ID")
                yield p[None], y[None], dict(source_row=index, fold=source["fold"],
                                            case_id=row.get(spec.case_id_key) if spec.case_id_key else None)
                index += 1
        if index != source["rows"]:
            raise ValueError("Cache row count changed")


def small_input_bypass(probs, screening_mode=True):
    from rankseg import _rankseg_algo as algo
    if hasattr(algo, "_rma_dice_use_screening"):
        return not algo._rma_dice_use_screening(probs, screening_mode)
    b, c = probs.shape[:2]
    dim = math.prod(probs.shape[2:])
    return bool(probs.is_cuda and (
        (c == 1 and dim <= algo._RMA_CUDA_SCREENING_MIN_SINGLE_CHANNEL_DIM)
        or (b * c <= algo._RMA_CUDA_SCREENING_FEW_ROWS
            and dim <= algo._RMA_CUDA_SCREENING_MIN_FEW_ROWS_DIM)))


def observed_screening_call(operation):
    """One untimed call verifies real dispatch; hook is removed before timing."""
    from rankseg import _rankseg_algo as algo
    original = algo._rma_dice_screened_masks
    calls = []
    def checked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    try:
        algo._rma_dice_screened_masks = checked
        prediction = operation()
        del prediction
    finally:
        algo._rma_dice_screened_masks = original
    return len(calls)


@torch.no_grad()
def measure(operations, warmup, repeats, index, controls=None):
    """No correctness hook, metric, oracle, or count collection in timed calls."""
    values, predictions = {}, {}
    controls = controls or {}
    for name, call in operations.items():
        with controls.get(name, nullcontext)():
            for _ in range(warmup):
                prediction = call()
                del prediction
            torch.cuda.synchronize()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            prediction = call()
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - resident) / 2**20
            predictions[name] = prediction.cpu()
            del prediction
        values[name] = {"peak_mib": peak, "samples_ms": []}
    names = tuple(operations)
    for repeat in range(repeats):
        shift = (repeat + index) % len(names)
        for name in names[shift:] + names[:shift]:
            with controls.get(name, nullcontext)():
                torch.cuda.synchronize()
                start = time.perf_counter()
                prediction = operations[name]()
                torch.cuda.synchronize()
                values[name]["samples_ms"].append((time.perf_counter() - start) * 1000)
                del prediction
    for row in values.values():
        row["median_ms"] = statistics.median(row["samples_ms"])
        assert len(row["samples_ms"]) == repeats
        assert all(math.isfinite(t) and t > 0 for t in row["samples_ms"])
        assert math.isfinite(row["peak_mib"]) and row["peak_mib"] >= 0
    return values, predictions


def summarize(report, counts_helper):
    records = report["records"]
    if len(records) != sum(s["rows"] for s in report["sources"]):
        raise ValueError("Incomplete dataset")
    keys = [(r["fold"], r["source_row"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicated samples")
    results = {}
    for method in METHODS:
        metrics = independent_metrics(records, method, report["dataset"] == "kits")
        repo = report["repository_metrics"][method]
        for k, name in (("dice", "mDice"), ("iou", "mIoU")):
            if not math.isclose(metrics[k], repo[name], abs_tol=1e-12, rel_tol=0):
                raise AssertionError(f"Independent metric disagreement: {method}/{k}")
        rows = [r["methods"][method] for r in records]
        objective = [r["objective"] for r in rows] if method != "argmax" else []
        results[method] = {**metrics, "mean_ms": statistics.mean(r["median_ms"] for r in rows),
                           "mean_peak_mib": statistics.mean(r["peak_mib"] for r in rows),
                           "max_peak_mib": max(r["peak_mib"] for r in rows),
                           "objective_passed": sum(r["status"] == "passed" for r in objective),
                           "objective_failed": sum(r["status"] == "failed" for r in objective),
                           "max_regret_eps": max((r["max_regret_eps"] for r in objective), default=None)}
    certificate = counts_helper.summarize([r["screening_counts"] for r in records])
    active = certificate["active_entries"]
    actual_sorted = sum(sum(row["active_entries"] for row in r["screening_counts"]["rows"])
                        if r["small_input_bypass"] else
                        sum(row["actual_sort_entries"] for row in r["screening_counts"]["rows"])
                        for r in records)
    changes = {}
    for comparator in ("full", "full_optimized"):
        deltas = []
        for r in records:
            a = unit_scores(r["methods"][comparator]["counts"], report["dataset"] == "kits")
            b = unit_scores(r["methods"]["screened"]["counts"], report["dataset"] == "kits")
            if a is not None and b is not None:
                deltas.append((100 * (b["dice"] - a["dice"]), r["fold"], r["source_row"], r["case_id"]))
        changes[comparator] = {"changed_samples": sum(r["different_pixels"][comparator] > 0 for r in records),
                               "different_pixels": sum(r["different_pixels"][comparator] for r in records),
                               "worst_unit_dice_delta_pp": min(deltas, default=None)}
    return {"samples": len(records), "case_count": len({(r["fold"], r["case_id"]) for r in records})
            if report["dataset"] == "kits" else None,
            "methods": results, "metrics_audit": "passed", "forced_screening_certificate": certificate,
            "production_small_input_bypass_samples": sum(r["small_input_bypass"] for r in records),
            "production_screening_calls": sum(r["observed_screening_calls"] for r in records),
            "production_effective_sort_avoidance_active": 1 - actual_sorted / active if active else None,
            "differences_vs_screened": changes,
            "screened_checks_passed": results["screened"]["objective_passed"] == len(records),
            "forced_screening_checks_passed": results["screened_forced"]["objective_passed"] == len(records)}


@torch.no_grad()
def run_dataset(args, dataset, oracle, counts_helper, frozen):
    from rankseg_benchmark.datasets import get_spec
    from rankseg_benchmark.metrics import ConfusionAccumulator, MedicalCaseAccumulator, FoldMeanAccumulator
    from rankseg_benchmark.runner import _rankseg_channels, _rankseg_output_mode, _rankseg_predict
    spec = get_spec(dataset)
    mode = _rankseg_output_mode(spec, "RMA")
    channels = _rankseg_channels(spec, mode)
    metadata, sources = cache_sources(dataset, args.natural_cache, args.kits_root)
    if spec.eval_unit == "case":
        accumulators = {m: {f: MedicalCaseAccumulator(spec.num_classes, spec.ignore_index, True)
                            for f in range(5)} for m in METHODS}
    else:
        accumulators = {m: ConfusionAccumulator(spec.num_classes, spec.output_mode, spec.ignore_index,
                                               spec.evaluation_class_ids) for m in METHODS}
    screening_mode = "auto" if getattr(args, "screening_mode", "true") == "auto" else True
    decoders = {m: make_decoder(m, mode, oracle, screening_mode) for m in METHODS if m != "argmax"}
    binary = {m: make_decoder(m, "multilabel", oracle, screening_mode) for m in METHODS if m != "argmax"}
    # Timed calls use plain public decoders. Scoped control setup/teardown is
    # outside the clock, just as in the real-scale calibration harness.
    controls = {"full_optimized": lambda: oracle.decoder_control("full_optimized"),
                "screened_forced": lambda: oracle.decoder_control("screened")}
    timed_decoders = {m: d.decoder if m in controls else d for m, d in decoders.items()}
    from rankseg import RankSEG
    default_decoder = RankSEG(metric="dice", solver="RMA", smooth=0, pruning_prob=.5, output_mode=mode)
    report = dict(complete=False, dataset=dataset, selection=metadata, sources=sources,
                  rankseg_output_mode=mode, channels=channels, source_hashes=frozen,
                  warmup=args.warmup, repeats=args.repeats, batch_size=1,
                  torch=torch.__version__, device=torch.cuda.get_device_name(), records=[],
                  screening_mode=screening_mode, timing_controls="outside timed intervals")
    import triton
    report["triton"] = triton.__version__
    output = args.output_dir / f"{dataset}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{dataset}: starting {sum(s['rows'] for s in sources)} samples", flush=True)
    with (args.output_dir / f"{dataset}.records.jsonl").open("x") as journal:
        for index, (cpu_probs, labels, identity) in enumerate(samples(spec, sources)):
            probs = cpu_probs.cuda()
            operations = {m: (lambda d=d, p=probs: _rankseg_predict(d, p, spec=spec, rankseg_output_mode=mode))
                          for m, d in timed_decoders.items()}
            operations = {"argmax": lambda p=probs: p.argmax(1), **operations}
            values, predictions = measure(operations, args.warmup, args.repeats, index, controls)
            default_prediction = _rankseg_predict(default_decoder, probs, spec=spec, rankseg_output_mode=mode)
            if not torch.equal(default_prediction.cpu(), predictions["full"]):
                raise AssertionError("Default output differs from explicit False")
            del default_prediction
            record = {**identity, "shape": list(probs.shape), "methods": values,
                      "default_matches_false": True,
                      "input_sha256": hashlib.sha256(cpu_probs.numpy().tobytes()).hexdigest(),
                      "label_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest()}
            for method, pred in predictions.items():
                values[method]["counts"] = confusion(pred, labels, spec.num_classes, spec.ignore_index,
                                                      spec.eval_unit == "case")
                if spec.eval_unit == "case":
                    accumulators[method][identity["fold"]].update_slice(identity["case_id"], pred[0], labels[0])
                else:
                    accumulators[method].update(pred, labels)
            record["different_pixels"] = {m: int((predictions[m] != predictions["screened"]).sum())
                                           for m in METHODS if m != "screened"}
            del predictions
            selected = probs[:, channels]
            record["small_input_bypass"] = small_input_bypass(selected, screening_mode)
            record["observed_screening_calls"] = observed_screening_call(operations["screened"])
            active = bool(selected.max().item() > .5)
            if record["observed_screening_calls"] != int(active and not record["small_input_bypass"]):
                raise AssertionError("Observed dispatch differs from source-derived dispatch")
            record["screening_counts"] = counts_helper.screening_counts(selected, channels)
            optimum = oracle.bounded_binary_oracle(selected)
            for method, decoder in binary.items():
                mask = decoder.predict(selected)
                try:
                    diag = oracle.objective_regret(selected, mask, optimum)
                    values[method]["objective"] = {**diag, "status": "passed"}
                except oracle.ObjectiveBudgetExceeded as error:
                    values[method]["objective"] = {**error.diagnostics, "status": "failed"}
                del mask
            # No tolerance here: this checks that none of the diagnostics or
            # decoder methods modified the shared input, not output equality.
            if not torch.equal(probs.cpu(), cpu_probs):
                raise AssertionError("Probability input modified")
            report["records"].append(record)
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            if (index + 1) % 100 == 0:
                journal.flush()
                (args.output_dir / "progress.json").write_text(json.dumps(
                    {"dataset": dataset, "completed_samples": index + 1,
                     "expected_samples": sum(s["rows"] for s in sources)}))
            del probs, selected, optimum, operations
    repository_metrics = {}
    for method, accumulator in accumulators.items():
        if spec.eval_unit == "case":
            folded = FoldMeanAccumulator(spec.num_classes)
            for fold, acc in accumulator.items():
                folded.add_fold(fold, acc)
            accumulator = folded
        result = accumulator.summary()
        repository_metrics[method] = {k: result[k] for k in ("mDice", "mIoU")}
        if "folds" in result:
            repository_metrics[method]["folds"] = result["folds"]
    report["repository_metrics"] = repository_metrics
    report["summary"] = summarize(report, counts_helper)
    for path, digest in frozen.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Frozen source changed: {path}")
    for source in sources:
        if sha256(source["path"]) != source["sha256"]:
            raise RuntimeError("Probability cache changed during benchmark")
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{dataset}: complete; screened numerical checks "
          f"{report['summary']['screened_checks_passed']}", flush=True)
    torch.cuda.empty_cache()
    return report["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankseg-path", type=Path, required=True)
    parser.add_argument("--medical-root", type=Path, default=ROOT)
    parser.add_argument("--natural-cache", type=Path, default=ROOT / ".cache/screening-1884f076-first100")
    parser.add_argument("--kits-root", type=Path, default=Path("/home/ben/.cache/huggingface/hub/")
                        / "datasets--ZixunWang--rankseg-benchmark/snapshots" / REVISION)
    parser.add_argument("--datasets", nargs="+", default=["pascal_voc", "cityscapes", "ade20k", "kits"],
                        choices=["pascal_voc", "cityscapes", "ade20k", "kits"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--screening-mode", choices=("true", "auto"), default="true",
                        help="public screened method; explicit in each report, preserving historical True runs")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or len(args.datasets) != len(set(args.datasets)):
        parser.error("Positive warmup/repeats and unique datasets required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no implicit CPU fallback")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    oracle, counts_helper = load_helpers(args.rankseg_path, args.medical_root)
    torch.set_num_threads(4)
    sources = [*args.rankseg_path.joinpath("rankseg").rglob("*.py"),
               *ROOT.joinpath("rankseg_benchmark").rglob("*.py"), Path(__file__).resolve(),
               args.medical_root / "rankseg_benchmark/nnunet/screening_benchmark.py",
               args.medical_root / "scripts/nnunet/collect_screening_proportions.py"]
    frozen = {str(path.resolve()): sha256(path) for path in sources}
    report = {"complete": False, "source_hashes": frozen, "datasets": {},
              "screening_mode": args.screening_mode,
              "protocol": "B=1; same probabilities; normal public on/off dispatch; separate optimized-full and forced-screening controls; routing/output conversion included; synchronized decoder-only timings; mean of per-sample medians; incremental peak allocated memory including output, excluding resident inputs; Dice objective, IoU additionally measured"}
    status = args.output_dir / "summary.json"
    status.write_text(json.dumps(report, indent=2))
    for dataset in args.datasets:
        report["datasets"][dataset] = run_dataset(args, dataset, oracle, counts_helper, frozen)
        status.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["complete"] = True
    report["all_screened_checks_passed"] = all(r["screened_checks_passed"] for r in report["datasets"].values())
    report["all_forced_screening_checks_passed"] = all(r["forced_screening_checks_passed"] for r in report["datasets"].values())
    status.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("All selected datasets completed", flush=True)


if __name__ == "__main__":
    main()
