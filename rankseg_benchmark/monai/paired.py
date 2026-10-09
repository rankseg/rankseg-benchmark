"""Paired, warmed CUDA timing of Argmax and RMA Dice decoders.

The probability cache is validated through the normal local-artifact loader.
Each volume is warmed independently so Triton compilation for a new spatial
shape cannot be mistaken for steady-state decoder latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import torch

from rankseg_benchmark.quick.datasets import get_spec, iter_batches_with_metadata
from rankseg_benchmark.quick.runner import _rankseg_channels, _rankseg_output_mode


def _measure_call(predict: Callable[[torch.Tensor], torch.Tensor], probabilities: torch.Tensor) -> float:
    torch.cuda.synchronize(probabilities.device)
    started = time.perf_counter()
    prediction = predict(probabilities)
    torch.cuda.synchronize(probabilities.device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    del prediction
    return elapsed_ms


def _peak_increment_mib(predict: Callable[[torch.Tensor], torch.Tensor], probabilities: torch.Tensor) -> float:
    torch.cuda.synchronize(probabilities.device)
    resident = torch.cuda.memory_allocated(probabilities.device)
    torch.cuda.reset_peak_memory_stats(probabilities.device)
    prediction = predict(probabilities)
    torch.cuda.synchronize(probabilities.device)
    peak = torch.cuda.max_memory_allocated(probabilities.device) - resident
    del prediction
    return peak / 2**20


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Paired benchmark requires at least one volume")
    methods = ("argmax", "full", "screened")
    mean_median_ms = {method: statistics.mean(record[method]["median_ms"] for record in records) for method in methods}
    mean_peak_mib = {
        method: statistics.mean(record[method]["peak_increment_mib"] for record in records) for method in methods
    }
    return {
        "volumes": len(records),
        "mean_of_volume_medians_ms": mean_median_ms,
        "mean_incremental_peak_mib": mean_peak_mib,
        "changed_pixels": sum(record["changed_pixels"] for record in records),
    }


@torch.inference_mode()
def benchmark(artifact_dir: Path, *, warmup: int, repeats: int, limit: int | None = None) -> dict:
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must both be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("Paired MONAI timing requires CUDA; no CPU substitution is performed")
    artifact_dir = artifact_dir.expanduser().resolve()
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Paired benchmark requires a complete MONAI probability cache")
    spec = get_spec(manifest["benchmark_id"])
    if spec.source_type != "local_artifacts" or spec.eval_unit != "volume":
        raise ValueError("Paired benchmark requires a local MONAI volume target")

    from rankseg import RankSEG
    import rankseg

    output_mode = _rankseg_output_mode(spec, "RMA")
    channels = _rankseg_channels(spec, output_mode)
    decoders = {
        "full": RankSEG(metric="dice", solver="RMA", output_mode=output_mode, safe_screening=False),
        "screened": RankSEG(metric="dice", solver="RMA", output_mode=output_mode, safe_screening=True),
    }
    methods = ("argmax", "full", "screened")
    records = []
    samples = iter_batches_with_metadata(spec, limit=limit, device="cuda", batch_size=1, artifact_dir=artifact_dir)
    for index, (probabilities, _, metadata) in enumerate(samples):
        case_id = metadata[0]["case_id"]
        if case_id != manifest["cases"][index]["case_id"]:
            raise ValueError("Artifact case order does not match the manifest")
        selected = probabilities[:, channels].contiguous()
        predictions = {method: decoder.predict(selected) for method, decoder in decoders.items()}
        changed_pixels = int((predictions["full"] != predictions["screened"]).sum().item())
        del predictions
        calls = {
            "argmax": (lambda tensor: torch.argmax(tensor, dim=1), probabilities),
            "full": (decoders["full"].predict, selected),
            "screened": (decoders["screened"].predict, selected),
        }
        for iteration in range(warmup):
            for offset in range(len(methods)):
                method = methods[(index + iteration + offset) % len(methods)]
                _measure_call(*calls[method])
        record = {"case_id": case_id, "changed_pixels": changed_pixels}
        for method in methods:
            record[method] = {"peak_increment_mib": _peak_increment_mib(*calls[method]), "samples_ms": []}
        for repetition in range(repeats):
            for offset in range(len(methods)):
                method = methods[(index + repetition + offset) % len(methods)]
                record[method]["samples_ms"].append(_measure_call(*calls[method]))
        for method in methods:
            record[method]["median_ms"] = statistics.median(record[method]["samples_ms"])
        records.append(record)
        print(f"[{index + 1}/{min(limit or len(manifest['cases']), len(manifest['cases']))}] {case_id}", flush=True)

    if len(records) != min(limit or len(manifest["cases"]), len(manifest["cases"])):
        raise ValueError("Paired benchmark did not process every selected case")
    return {
        "benchmark_id": spec.name,
        "artifact_dir": str(artifact_dir),
        "complete_cohort": limit is None or limit >= len(manifest["cases"]),
        "warmup_per_volume": warmup,
        "repeats_per_volume": repeats,
        "rankseg_version": version("rankseg"),
        "rankseg_module": str(Path(rankseg.__file__).resolve()),
        "torch_version": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "summary": summarize(records),
        "cases": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = benchmark(args.artifact_dir, warmup=args.warmup, repeats=args.repeats, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
