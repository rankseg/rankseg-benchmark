"""Command-line entry: ``rankseg-bench --dataset X --solver Y``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

from rankseg_benchmark.datasets import REGISTRY, get_spec, list_datasets
from rankseg_benchmark.runner import format_report, run_benchmark

LOGGER = logging.getLogger(__name__)


def _display_cache_dir(cache_dir: Path | None) -> str:
    if cache_dir is not None:
        return str(cache_dir.expanduser())
    if os.environ.get("HF_DATASETS_CACHE"):
        return str(Path(os.environ["HF_DATASETS_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        return str(Path(os.environ["HF_HOME"]).expanduser() / "datasets")
    return str(Path.home() / ".cache" / "huggingface" / "datasets")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rankseg-bench",
        description="Benchmark RankSEG against the argmax baseline on pre-computed probability masks.",
    )
    p.add_argument(
        "--dataset",
        choices=list_datasets(),
        help="Which benchmark dataset to evaluate. Required unless --list-datasets is set.",
    )
    p.add_argument(
        "--solver",
        default="RMA",
        help="RankSEG solver (RMA, BA, TRNA, BA+TRNA). Default: RMA.",
    )
    p.add_argument(
        "--metric",
        default="dice",
        choices=["dice", "iou"],
        help="Metric to optimize. Default: dice.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference. Default: cuda if available else cpu.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples (for quick smoke runs).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup samples (timing not recorded). Default: 3.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Maximum number of same-shape samples per inference batch. "
            "Variable-size samples are flushed into smaller batches. Default: 8."
        ),
    )
    p.add_argument(
        "--cache-dataset",
        action="store_true",
        help="Download/cache the full Hugging Face dataset locally before running instead of streaming rows.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache directory used with --cache-dataset.",
    )
    p.add_argument(
        "--per-class",
        action="store_true",
        help="Also print per-class Dice/IoU gain breakdown.",
    )
    p.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        default=None,
        help="If set, write full results (including per-class arrays) to this JSON file.",
    )
    p.add_argument(
        "--rankseg-path",
        type=Path,
        default=None,
        help=(
            "Optional local RankSEG checkout to import instead of an installed package. "
            "Equivalent to setting RANKSEG_PATH; the CLI flag takes precedence."
        ),
    )
    p.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit.",
    )
    log_group = p.add_mutually_exclusive_group()
    log_group.add_argument(
        "--quiet",
        action="store_true",
        help="Only print warnings/errors plus the final report.",
    )
    log_group.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debug details while loading data and running inference.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    if args.list_datasets:
        for name in list_datasets():
            spec = REGISTRY[name]
            print(f"  {name:<12} {spec.num_classes:>4} classes  {spec.output_mode:<10}  {spec.description}")
        return 0

    if not args.dataset:
        parser.error("--dataset is required (or pass --list-datasets to see options).")

    spec = get_spec(args.dataset)
    LOGGER.info("Starting RankSEG benchmark")
    LOGGER.info(
        "Configuration: dataset=%s split=%s data_dir=%s classes=%d mode=%s ignore_index=%s",
        spec.name,
        spec.hf_split,
        spec.hf_data_dir or "<repo root>",
        spec.num_classes,
        spec.output_mode,
        spec.ignore_index,
    )
    LOGGER.info(
        "Run options: solver=%s metric=%s device=%s limit=%s warmup=%d batch_size=%d cache_dataset=%s cache_dir=%s rankseg_path=%s",
        args.solver,
        args.metric,
        args.device,
        args.limit if args.limit is not None else "all",
        args.warmup,
        args.batch_size,
        args.cache_dataset,
        _display_cache_dir(args.cache_dir),
        args.rankseg_path or "<installed package or RANKSEG_PATH>",
    )
    baseline, rs = run_benchmark(
        spec,
        metric=args.metric,
        solver=args.solver,
        device=args.device,
        limit=args.limit,
        warmup=args.warmup,
        batch_size=args.batch_size,
        cache_dataset=args.cache_dataset,
        cache_dir=args.cache_dir,
        rankseg_path=args.rankseg_path,
    )

    print(format_report(baseline, rs, per_class=args.per_class))

    if args.json_out:
        LOGGER.info("Serializing full benchmark results to JSON: %s", args.json_out)
        payload = {
            "dataset": spec.name,
            "metric": args.metric,
            "solver": args.solver,
            "device": args.device,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "cache_dataset": args.cache_dataset,
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "rankseg_path": str(args.rankseg_path) if args.rankseg_path else None,
            "baseline": _serialize(baseline.summary()),
            "rankseg": _serialize(rs.summary()),
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON results -> {args.json_out}")
        LOGGER.info("JSON results written")

    return 0


def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    package_level = logging.WARNING if quiet else logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("rankseg_benchmark").setLevel(package_level)


def _serialize(s: dict) -> dict:
    return {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in s.items()
    }


if __name__ == "__main__":
    sys.exit(main())
