"""End-to-end benchmark runner: iterate samples, run RankSEG + argmax baseline, collect metrics & timing."""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from tqdm import tqdm

from rankseg_benchmark.datasets import (
    DatasetSpec,
    get_dataset_size_from_cache,
    get_dataset_size_from_metadata,
    get_local_artifact_size,
    iter_batches,
    iter_batches_with_metadata,
)
from rankseg_benchmark.metrics import ConfusionAccumulator, FoldMeanAccumulator, MedicalCaseAccumulator
from rankseg_benchmark.timing import Timer

LOGGER = logging.getLogger(__name__)


@dataclass
class RunResult:
    method: str  # "argmax" / "rankseg-<solver>"
    confusion: ConfusionAccumulator | FoldMeanAccumulator
    timer: Timer
    unit: str = "img"
    evaluation_class_ids: tuple[int, ...] | None = None

    def summary(self) -> dict:
        s = self.confusion.summary()
        result = {
            "method": self.method,
            # per-image metrics (as in the RankSEG-RMA paper)
            "mDice": s["mDice"],
            "mIoU": s["mIoU"],
            # per-class breakdown (averaged over images where class is active)
            "per_class_dice": s["per_class_dice"],
            "per_class_iou": s["per_class_iou"],
            "active_per_class": s["active_per_class"],
            # timing
            "mean_ms": self.timer.mean_ms,
            "median_ms": self.timer.median_ms,
            "n_calls": self.timer.n_calls,
            "unit": self.unit,
        }
        if "folds" in s:
            result["folds"] = s["folds"]
        return result


def _baseline_argmax(probs: torch.Tensor, output_mode: str) -> torch.Tensor:
    if output_mode == "multiclass":
        return torch.argmax(probs, dim=1)
    return probs > 0.5  # multilabel baseline = thresholding


def _rankseg_output_mode(spec: DatasetSpec, solver: str) -> str:
    if spec.rankseg_channels is not None and len(spec.rankseg_channels) != spec.num_classes:
        if spec.output_mode == "multiclass" and len(spec.rankseg_channels) != 1:
            raise ValueError("multiclass channel selection currently requires exactly one RankSEG channel")
        return "multilabel"
    if spec.output_mode == "multiclass" and spec.num_classes == 2 and solver.strip().upper() in {
        "BA",
        "TRNA",
        "BA+TRNA",
    }:
        return "multilabel"
    return spec.output_mode


def _rankseg_channels(spec: DatasetSpec, rankseg_output_mode: str) -> tuple[int, ...]:
    if spec.rankseg_channels is not None:
        return spec.rankseg_channels
    if spec.output_mode == "multiclass" and rankseg_output_mode == "multilabel":
        if spec.num_classes != 2:
            raise ValueError("automatic foreground-channel routing requires a two-class multiclass dataset")
        return (1,)
    return tuple(range(spec.num_classes))


def _rankseg_predict(
    rankseg,
    probs: torch.Tensor,
    *,
    spec: DatasetSpec,
    rankseg_output_mode: str,
) -> torch.Tensor:
    channels = _rankseg_channels(spec, rankseg_output_mode)
    rankseg_probs = probs[:, channels]
    prediction = rankseg.predict(rankseg_probs)
    if spec.output_mode == "multiclass" and rankseg_output_mode == "multilabel":
        if len(channels) != 1 or prediction.shape[1] != 1:
            raise ValueError("multiclass conversion requires one selected RankSEG channel")
        foreground_class = channels[0]
        if foreground_class == spec.background_class_id:
            raise ValueError("the selected RankSEG channel must not be the background class")
        return torch.where(
            prediction[:, 0].bool(),
            foreground_class,
            spec.background_class_id,
        ).long()
    if spec.output_mode == "multilabel" and len(channels) != spec.num_classes:
        full_prediction = torch.zeros(
            prediction.shape[0],
            spec.num_classes,
            *prediction.shape[2:],
            dtype=prediction.dtype,
            device=prediction.device,
        )
        full_prediction[:, channels] = prediction
        return full_prediction
    return prediction


def _predict_timed_batch(
    probs: torch.Tensor,
    *,
    spec: DatasetSpec,
    rankseg_output_mode: str,
    rankseg,
    base_timer: Timer,
    rs_timer: Timer,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_n = probs.size(0)
    with base_timer.measure(units=batch_n):
        base_pred = _baseline_argmax(probs, spec.output_mode)
    with rs_timer.measure(units=batch_n):
        rs_pred = _rankseg_predict(
            rankseg,
            probs,
            spec=spec,
            rankseg_output_mode=rankseg_output_mode,
        )
    return base_pred, rs_pred


def configure_rankseg_path(rankseg_path: str | os.PathLike[str] | None = None) -> Path | None:
    """Prepend a local RankSEG checkout to sys.path before importing rankseg.

    The explicit argument wins over RANKSEG_PATH. If neither is set, Python's
    normal import resolution is left untouched, so an installed rankseg package
    is used.
    """
    raw_path = rankseg_path or os.environ.get("RANKSEG_PATH")
    if not raw_path:
        LOGGER.info("Using RankSEG from the active Python environment")
        return None

    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RankSEG path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"RankSEG path is not a directory: {path}")

    path_str = str(path)
    if sys.path[:1] != [path_str]:
        sys.path = [p for p in sys.path if p != path_str]
        sys.path.insert(0, path_str)
    LOGGER.info("Using local RankSEG checkout: %s", path)
    return path


def run_benchmark(
    spec: DatasetSpec,
    *,
    metric: str = "dice",
    solver: str = "RMA",
    device: str = "cpu",
    limit: int | None = None,
    warmup: int = 3,
    batch_size: int = 8,
    cache_dataset: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
    artifact_dir: str | os.PathLike[str] | None = None,
    progress: bool = True,
    rankseg_path: str | os.PathLike[str] | None = None,
) -> tuple[RunResult, RunResult]:
    """Run argmax baseline and RankSEG on the same data stream. Returns (baseline, rankseg)."""
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if spec.eval_unit == "volume" and batch_size != 1:
        LOGGER.warning("Forcing batch_size=1 for volume-level latency and bounded memory use")
        batch_size = 1
    LOGGER.info("Preparing benchmark runner")
    configure_rankseg_path(rankseg_path)
    LOGGER.info("Importing RankSEG")
    from rankseg import RankSEG  # imported lazily so help/--list-datasets work without rankseg installed

    dev = torch.device(device)
    use_cuda = dev.type == "cuda"
    if use_cuda:
        LOGGER.info(
            "Using CUDA device: %s (index=%s)",
            torch.cuda.get_device_name(dev) if dev.index is not None else torch.cuda.get_device_name(),
            dev.index if dev.index is not None else torch.cuda.current_device(),
        )
    else:
        LOGGER.info("Using device: %s", dev)

    rankseg_output_mode = _rankseg_output_mode(spec, solver)
    rankseg_channels = _rankseg_channels(spec, rankseg_output_mode)
    if spec.rankseg_channels is not None and len(spec.rankseg_channels) != spec.num_classes:
        LOGGER.info(
            "Dataset %s routes probability channels %s to RankSEG with output_mode=%s for solver=%s",
            spec.name,
            rankseg_channels,
            rankseg_output_mode,
            solver,
        )
    elif rankseg_output_mode != spec.output_mode:
        LOGGER.info(
            "Using RankSEG output_mode=%s for solver=%s on binary multiclass dataset %s: "
            "BA/TRNA-style solvers operate on binary masks, so RankSEG receives only the foreground "
            "probability channel and predictions are converted back to binary class labels for evaluation",
            rankseg_output_mode,
            solver,
            spec.name,
        )
    LOGGER.info(
        "Constructing RankSEG: solver=%s metric=%s dataset_output_mode=%s rankseg_output_mode=%s",
        solver,
        metric,
        spec.output_mode,
        rankseg_output_mode,
    )
    rankseg = RankSEG(metric=metric, solver=solver, output_mode=rankseg_output_mode)

    if spec.eval_unit == "case":
        return _run_case_benchmark(
            spec,
            rankseg=rankseg,
            rankseg_output_mode=rankseg_output_mode,
            solver=solver,
            dev=dev,
            use_cuda=use_cuda,
            limit=limit,
            warmup=warmup,
            batch_size=batch_size,
            cache_dataset=cache_dataset,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
            progress=progress,
        )

    base_conf = ConfusionAccumulator(
        spec.num_classes,
        spec.output_mode,
        spec.ignore_index,
        spec.evaluation_class_ids,
    )
    rs_conf = ConfusionAccumulator(
        spec.num_classes,
        spec.output_mode,
        spec.ignore_index,
        spec.evaluation_class_ids,
    )
    base_timer = Timer(use_cuda=use_cuda)
    rs_timer = Timer(use_cuda=use_cuda)
    sample_unit = "volume" if spec.eval_unit == "volume" else "img"

    progress_total = limit
    if progress_total is None:
        if spec.source_type == "local_artifacts":
            progress_total = get_local_artifact_size(spec, artifact_dir=artifact_dir)
        else:
            progress_total = get_dataset_size_from_metadata(spec)
    if spec.source_type == "huggingface" and cache_dataset and progress_total is None:
        progress_total = get_dataset_size_from_cache(spec, cache_dir=str(cache_dir) if cache_dir is not None else None)
    if progress and progress_total is None:
        LOGGER.info("Progress bar will show processed images and throughput; ETA requires --limit or dataset metadata")

    samples = iter_batches(
        spec,
        limit=limit,
        device=dev,
        batch_size=batch_size,
        cache_dataset=cache_dataset,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        artifact_dir=artifact_dir,
    )
    progress_bar = (
        tqdm(desc=f"{spec.name} [{solver}/{metric}]", total=progress_total, unit=sample_unit, dynamic_ncols=True)
        if progress
        else None
    )

    LOGGER.info(
        "Starting sample loop: warmup=%d timed_limit=%s batch_size=%d dataset_mode=%s",
        warmup,
        limit if limit is not None else "all",
        batch_size,
        "cache-first" if cache_dataset else "streaming",
    )
    warmed = 0
    processed = 0
    try:
        for probs, label in samples:
            batch_n = probs.size(0)
            progress_n = batch_n
            # Warmup: run a few times without recording timing (CUDA kernel JIT).
            if warmed < warmup:
                warmup_n = min(warmup - warmed, batch_n)
                LOGGER.debug(
                    "Warmup batch slice: samples=%d warmed_after=%d/%d probs_shape=%s label_shape=%s",
                    warmup_n,
                    warmed + warmup_n,
                    warmup,
                    tuple(probs[:warmup_n].shape),
                    tuple(label[:warmup_n].shape),
                )
                _ = _baseline_argmax(probs[:warmup_n], spec.output_mode)
                _ = _rankseg_predict(
                    rankseg,
                    probs[:warmup_n],
                    spec=spec,
                    rankseg_output_mode=rankseg_output_mode,
                )
                warmed += warmup_n

            LOGGER.debug(
                "Timed batch %d: batch_size=%d probs_shape=%s label_shape=%s",
                rs_timer.n_batches + 1,
                batch_n,
                tuple(probs.shape),
                tuple(label.shape),
            )
            base_pred, rs_pred = _predict_timed_batch(
                probs,
                spec=spec,
                rankseg_output_mode=rankseg_output_mode,
                rankseg=rankseg,
                base_timer=base_timer,
                rs_timer=rs_timer,
            )
            base_conf.update(base_pred, label)
            rs_conf.update(rs_pred, label)

            previous_processed = processed
            processed += batch_n
            if processed == batch_n or processed // 25 > previous_processed // 25:
                LOGGER.info(
                    "Processed %d timed %ss: argmax_mean=%.2f ms/%s RankSEG_mean=%.2f ms/%s",
                    processed,
                    sample_unit,
                    base_timer.mean_ms,
                    sample_unit,
                    rs_timer.mean_ms,
                    sample_unit,
                )
            if progress_bar is not None:
                progress_bar.update(progress_n)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    LOGGER.info(
        "Benchmark loop complete: warmup_samples=%d timed_samples=%d",
        warmed,
        processed,
    )
    LOGGER.info(
        "Timing summary: argmax mean=%.2f ms/%s median=%.2f ms/%s samples=%d batches=%d | "
        "RankSEG mean=%.2f ms/%s median=%.2f ms/%s samples=%d batches=%d",
        base_timer.mean_ms,
        sample_unit,
        base_timer.median_ms,
        sample_unit,
        base_timer.n_calls,
        base_timer.n_batches,
        rs_timer.mean_ms,
        sample_unit,
        rs_timer.median_ms,
        sample_unit,
        rs_timer.n_calls,
        rs_timer.n_batches,
    )

    return (
        RunResult("argmax", base_conf, base_timer, unit=sample_unit, evaluation_class_ids=spec.evaluation_class_ids),
        RunResult(
            f"rankseg-{solver}",
            rs_conf,
            rs_timer,
            unit=sample_unit,
            evaluation_class_ids=spec.evaluation_class_ids,
        ),
    )


def _run_case_benchmark(
    spec: DatasetSpec,
    *,
    rankseg,
    rankseg_output_mode: str,
    solver: str,
    dev: torch.device,
    use_cuda: bool,
    limit: int | None,
    warmup: int,
    batch_size: int,
    cache_dataset: bool,
    cache_dir: str | None,
    artifact_dir: str | None,
    progress: bool,
) -> tuple[RunResult, RunResult]:
    if spec.num_folds is None or spec.num_folds < 1:
        raise ValueError(f"Case-level dataset {spec.name!r} requires num_folds >= 1")
    if not spec.hf_data_dir:
        raise ValueError(f"Case-level dataset {spec.name!r} requires hf_data_dir")
    if not spec.case_id_key:
        raise ValueError(f"Case-level dataset {spec.name!r} requires case_id_key")

    base_conf = FoldMeanAccumulator(spec.num_classes)
    rs_conf = FoldMeanAccumulator(spec.num_classes)
    base_timer = Timer(use_cuda=use_cuda)
    rs_timer = Timer(use_cuda=use_cuda)
    warmed = 0
    processed = 0

    LOGGER.info(
        "Starting case-level benchmark: folds=%d warmup=%d timed_limit_per_fold=%s batch_size=%d",
        spec.num_folds,
        warmup,
        limit if limit is not None else "all",
        batch_size,
    )

    for fold in range(spec.num_folds):
        fold_spec = replace(spec, hf_data_dir=f"{spec.hf_data_dir}/fold{fold}")
        fold_base = MedicalCaseAccumulator(
            spec.num_classes,
            spec.ignore_index,
            binary=spec.num_classes == 2,
        )
        fold_rs = MedicalCaseAccumulator(
            spec.num_classes,
            spec.ignore_index,
            binary=spec.num_classes == 2,
        )

        progress_total = limit
        if cache_dataset and progress_total is None:
            progress_total = get_dataset_size_from_cache(
                fold_spec,
                cache_dir=cache_dir,
            )
        progress_bar = (
            tqdm(
                desc=f"{spec.name} fold {fold}/{spec.num_folds - 1} [{solver}]",
                total=progress_total,
                unit="slice",
                dynamic_ncols=True,
            )
            if progress
            else None
        )

        samples = iter_batches_with_metadata(
            fold_spec,
            limit=limit,
            device=dev,
            batch_size=batch_size,
            cache_dataset=cache_dataset,
            cache_dir=cache_dir,
            artifact_dir=artifact_dir,
        )

        try:
            for probs, label, metadata in samples:
                batch_n = probs.size(0)
                progress_n = batch_n
                if warmed < warmup:
                    warmup_n = min(warmup - warmed, batch_n)
                    _ = _baseline_argmax(probs[:warmup_n], spec.output_mode)
                    _ = _rankseg_predict(
                        rankseg,
                        probs[:warmup_n],
                        spec=spec,
                        rankseg_output_mode=rankseg_output_mode,
                    )
                    warmed += warmup_n

                base_pred, rs_pred = _predict_timed_batch(
                    probs,
                    spec=spec,
                    rankseg_output_mode=rankseg_output_mode,
                    rankseg=rankseg,
                    base_timer=base_timer,
                    rs_timer=rs_timer,
                )

                for i, meta in enumerate(metadata):
                    if spec.case_id_key not in meta:
                        raise KeyError(f"Missing case id field {spec.case_id_key!r} in dataset row metadata")
                    case_id = meta[spec.case_id_key]
                    fold_base.update_slice(case_id, base_pred[i], label[i])
                    fold_rs.update_slice(case_id, rs_pred[i], label[i])

                previous_processed = processed
                processed += batch_n
                if processed == batch_n or processed // 25 > previous_processed // 25:
                    LOGGER.info(
                        "Processed %d timed slices: argmax_mean=%.2f ms/slice RankSEG_mean=%.2f ms/slice",
                        processed,
                        base_timer.mean_ms,
                        rs_timer.mean_ms,
                    )
                if progress_bar is not None:
                    progress_bar.update(progress_n)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        base_conf.add_fold(fold, fold_base)
        rs_conf.add_fold(fold, fold_rs)
        base_summary = fold_base.summary()
        rs_summary = fold_rs.summary()
        LOGGER.info(
            "Fold %d complete: cases=%d argmax_mIoU=%.2f RankSEG_mIoU=%.2f",
            fold,
            fold_base.n_cases,
            100.0 * float(base_summary["mIoU"]),
            100.0 * float(rs_summary["mIoU"]),
        )

    LOGGER.info(
        "Case-level benchmark complete: warmup_slices=%d timed_slices=%d folds=%d",
        warmed,
        processed,
        spec.num_folds,
    )

    return (
        RunResult("argmax", base_conf, base_timer, unit="slice", evaluation_class_ids=spec.evaluation_class_ids),
        RunResult(
            f"rankseg-{solver}",
            rs_conf,
            rs_timer,
            unit="slice",
            evaluation_class_ids=spec.evaluation_class_ids,
        ),
    )


def format_report(
    baseline: RunResult,
    rankseg_res: RunResult,
    *,
    class_names: list[str] | None = None,
    per_class: bool = False,
) -> str:
    from tabulate import tabulate

    b = baseline.summary()
    r = rankseg_res.summary()

    def _percent_row(label: str, key: str) -> list[str]:
        bv, rv = b[key], r[key]
        return [
            label,
            f"{100.0 * bv:.2f}",
            f"{100.0 * rv:.2f}",
            f"{100.0 * (rv - bv):+.2f}",
        ]

    performance_rows = [
        _percent_row("mDice", "mDice"),
        _percent_row("mIoU", "mIoU"),
    ]
    performance = tabulate(
        performance_rows,
        headers=["", baseline.method, rankseg_res.method, "Improvement"],
        tablefmt="github",
    )

    runtime_unit = b["unit"]
    runtime_rows = [
        [f"mean ms / {runtime_unit}", f"{b['mean_ms']:.2f}", f"{r['mean_ms']:.2f}", f"{r['mean_ms'] - b['mean_ms']:+.2f}"],
        [f"median ms / {runtime_unit}", f"{b['median_ms']:.2f}", f"{r['median_ms']:.2f}",
         f"{r['median_ms'] - b['median_ms']:+.2f}"],
    ]
    runtime = tabulate(
        runtime_rows,
        headers=["", baseline.method, rankseg_res.method, "Overhead"],
        tablefmt="github",
    )
    report = "## Performance\n\n" + performance + "\n\n## Runtime\n\n" + runtime

    if not per_class:
        return report

    n_classes = b["per_class_dice"].numel()
    class_ids = baseline.evaluation_class_ids or tuple(range(n_classes))
    active = b["active_per_class"]
    rows = []
    for c in class_ids:
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        rows.append([
            c,
            name,
            int(active[c].item()),
            f"{100.0 * b['per_class_dice'][c].item():.2f}",
            f"{100.0 * r['per_class_dice'][c].item():.2f}",
            f"{100.0 * (r['per_class_dice'][c] - b['per_class_dice'][c]).item():+.2f}",
            f"{100.0 * b['per_class_iou'][c].item():.2f}",
            f"{100.0 * r['per_class_iou'][c].item():.2f}",
            f"{100.0 * (r['per_class_iou'][c] - b['per_class_iou'][c]).item():+.2f}",
        ])
    active_unit = "volumes" if runtime_unit == "volume" else "slices" if runtime_unit == "slice" else "imgs"
    per_class_tbl = tabulate(
        rows,
        headers=["cls", "name", f"#{active_unit} (active)",
                 "Dice (base)", "Dice (RankSEG)", "Dice Improvement",
                 "IoU (base)", "IoU (RankSEG)", "IoU Improvement"],
        tablefmt="github",
    )
    return report + "\n\n## Per-class gain breakdown (averaged over active evaluation units)\n\n" + per_class_tbl
