"""End-to-end benchmark runner: iterate samples, run RankSEG + argmax baseline, collect metrics & timing."""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from rankseg_benchmark.datasets import DatasetSpec, get_dataset_size_from_cache, get_dataset_size_from_metadata, iter_batches
from rankseg_benchmark.metrics import ConfusionAccumulator
from rankseg_benchmark.timing import Timer

LOGGER = logging.getLogger(__name__)


@dataclass
class RunResult:
    method: str  # "argmax" / "rankseg-<solver>"
    confusion: ConfusionAccumulator
    timer: Timer

    def summary(self) -> dict:
        s = self.confusion.summary()
        return {
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
        }


def _baseline_argmax(probs: torch.Tensor, output_mode: str) -> torch.Tensor:
    if output_mode == "multiclass":
        return torch.argmax(probs, dim=1)
    return probs > 0.5  # multilabel baseline = thresholding


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
    progress: bool = True,
    rankseg_path: str | os.PathLike[str] | None = None,
) -> tuple[RunResult, RunResult]:
    """Run argmax baseline and RankSEG on the same data stream. Returns (baseline, rankseg)."""
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

    LOGGER.info("Constructing RankSEG: solver=%s metric=%s output_mode=%s", solver, metric, spec.output_mode)
    rankseg = RankSEG(metric=metric, solver=solver, output_mode=spec.output_mode)

    base_conf = ConfusionAccumulator(spec.num_classes, spec.output_mode, spec.ignore_index)
    rs_conf = ConfusionAccumulator(spec.num_classes, spec.output_mode, spec.ignore_index)
    base_timer = Timer(use_cuda=use_cuda)
    rs_timer = Timer(use_cuda=use_cuda)

    progress_total = limit
    if progress_total is None:
        progress_total = get_dataset_size_from_metadata(spec)
    if cache_dataset and progress_total is None:
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
    )
    progress_bar = (
        tqdm(desc=f"{spec.name} [{solver}/{metric}]", total=progress_total, unit="img", dynamic_ncols=True)
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
                _ = rankseg.predict(probs[:warmup_n])
                warmed += warmup_n
                if warmup_n == batch_n:
                    if progress_bar is not None:
                        progress_bar.update(progress_n)
                    continue
                probs = probs[warmup_n:]
                label = label[warmup_n:]
                batch_n = probs.size(0)

            LOGGER.debug(
                "Timed batch %d: batch_size=%d probs_shape=%s label_shape=%s",
                rs_timer.n_batches + 1,
                batch_n,
                tuple(probs.shape),
                tuple(label.shape),
            )
            with base_timer.measure(units=batch_n):
                base_pred = _baseline_argmax(probs, spec.output_mode)
            base_conf.update(base_pred, label)

            with rs_timer.measure(units=batch_n):
                rs_pred = rankseg.predict(probs)
            rs_conf.update(rs_pred, label)

            previous_processed = processed
            processed += batch_n
            if processed == batch_n or processed // 25 > previous_processed // 25:
                LOGGER.info(
                    "Processed %d timed samples: argmax_mean=%.2f ms/img RankSEG_mean=%.2f ms/img",
                    processed,
                    base_timer.mean_ms,
                    rs_timer.mean_ms,
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
        "Timing summary: argmax mean=%.2f ms/img median=%.2f ms/img samples=%d batches=%d | "
        "RankSEG mean=%.2f ms/img median=%.2f ms/img samples=%d batches=%d",
        base_timer.mean_ms,
        base_timer.median_ms,
        base_timer.n_calls,
        base_timer.n_batches,
        rs_timer.mean_ms,
        rs_timer.median_ms,
        rs_timer.n_calls,
        rs_timer.n_batches,
    )

    return (
        RunResult("argmax", base_conf, base_timer),
        RunResult(f"rankseg-{solver}", rs_conf, rs_timer),
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

    runtime_rows = [
        ["mean ms / img", f"{b['mean_ms']:.2f}", f"{r['mean_ms']:.2f}", f"{r['mean_ms'] - b['mean_ms']:+.2f}"],
        ["median ms / img", f"{b['median_ms']:.2f}", f"{r['median_ms']:.2f}",
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
    active = b["active_per_class"]  # same across baseline / rankseg (depends on label / pred)
    rows = []
    for c in range(n_classes):
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
    per_class_tbl = tabulate(
        rows,
        headers=["cls", "name", "#imgs (active)",
                 "Dice (base)", "Dice (RankSEG)", "Dice Improvement",
                 "IoU (base)", "IoU (RankSEG)", "IoU Improvement"],
        tablefmt="github",
    )
    return report + "\n\n## Per-class gain breakdown (averaged over active images)\n\n" + per_class_tbl
