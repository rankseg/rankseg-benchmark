"""Post-run screening counts, outside timing and without changing RankSEG.

Use the same CUDA certificate kernels, row layout, guards and workspace limits as
the forced-screening benchmark. Counts concern binary search, NOT final multiclass
eligibility. One entry means one class probability at one voxel, not a unique voxel.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


COUNT_KEYS = ("total_entries", "active_entries", "class_pruned_entries", "forced_positive",
              "forced_negative", "undecided", "actual_sort_entries", "fallback_rows",
              "active_rows", "total_rows")


def fractions(counts):
    """Ratios of summed counts, never averages of unequal-volume percentages."""
    active, total = counts["active_entries"], counts["total_entries"]
    return {
        "screening_fraction_active": ((counts["forced_positive"] + counts["forced_negative"]) / active
                                      if active else None),
        "effective_sort_avoidance_active": (1 - counts["actual_sort_entries"] / active if active else None),
        "class_pruned_fraction_all": counts["class_pruned_entries"] / total if total else None,
        "actual_sort_fraction_all": counts["actual_sort_entries"] / total if total else None,
    }


def summarize(records):
    totals = {key: 0 for key in COUNT_KEYS}
    by_label = {}
    workspace = 0
    for record in records:
        workspace += record["sort_workspace_elements"]
        for row in record["rows"]:
            label = str(row["label"])
            dest = by_label.setdefault(label, {key: 0 for key in COUNT_KEYS})
            for key in COUNT_KEYS:
                totals[key] += row[key]
                dest[key] += row[key]
    return {"cases": len(records), **totals, **fractions(totals),
            "sort_workspace_elements": workspace,
            "by_label": {key: {**value, **fractions(value)} for key, value in by_label.items()}}


@torch.no_grad()
def screening_counts(probs, labels):
    """Count H/L/M with the exact certificate backend; sorting is not repeated.

    CUDA requires the fused backend used by this experiment. The CPU branch is
    the existing pure-Torch certificate formula, useful for small regression tests.
    Forced-screening dispatch is assumed; fallback and padding are counted separately.
    """
    from rankseg import _screening as screening
    from rankseg._validation import validate_probability_tensor

    validate_probability_tensor(probs)
    if probs.shape[0] != 1 or len(labels) != probs.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("Expected one volume and one unique label per channel")
    probs = probs if probs.dtype == torch.float64 else probs.float()
    flat = probs.flatten(2).reshape(probs.shape[1], -1)
    channels, dim = flat.shape
    prepared = screening._rma_dice_screening_statistics(probs.flatten(2)) if probs.is_cuda else None
    if probs.is_cuda and prepared is None:
        raise RuntimeError("Fused CUDA backend required for matched statistics")
    if prepared is None:
        means, maxima = probs.flatten(2).sum(-1).reshape(-1), flat.amax(-1)
    else:
        means, maxima, state = prepared
        means, maxima = means.reshape(-1), maxima.reshape(-1)
    active = maxima > 0.5
    indices = active.nonzero(as_tuple=True)[0]
    original_rows = indices.cpu().tolist()
    counts_by_row = {}
    workspace = 0
    if indices.numel():
        all_active = prepared is not None or indices.numel() == channels
        rows = flat if all_active else flat.index_select(0, indices)
        selected_means = means if all_active else means.index_select(0, indices)
        selected_maxima = maxima if all_active else maxima.index_select(0, indices)
        guard = screening._ROUNDING_GUARD * torch.finfo(probs.dtype).eps
        if probs.is_cuda:
            backend = screening._cuda_backend()
            if backend is None:
                raise RuntimeError("Fused CUDA backend required for matched statistics")
            positive, undecided, n_forced, mass, partial_counts = backend.screening_candidates(
                rows, selected_means, selected_maxima, guard,
                active, statistics=state, return_counts=True, materialize_candidates=False,
            )
            candidate_counts = partial_counts.sum(-1)
        else:
            positive = (rows > 0.5 + guard).contiguous()
            n_forced = positive.sum(-1)
            mass = torch.where(positive, rows, 0).sum(-1)
            lower = torch.maximum(torch.maximum(selected_maxima / (selected_means + 2),
                                                mass / (n_forced + selected_means + 1)),
                                  selected_means / (dim + selected_means + 1))
            lower = (lower - guard).clamp(min=0, max=0.5)
            undecided = (rows >= lower[:, None]) & ~positive
            candidate_counts = undecided.sum(-1)
        forced = n_forced.cpu().tolist()
        candidates = candidate_counts.cpu().tolist()
        if prepared is not None:
            forced = [forced[row] for row in original_rows]
            candidates = [candidates[row] for row in original_rows]
        del positive, undecided, mass
        kept = []
        for row, h, m in zip(original_rows, forced, candidates):
            if not (0 <= h <= dim and 0 <= m <= dim - h):
                raise AssertionError("Invalid H/M certificate partition")
            # Older source snapshots retain candidate-budget retries; current
            # RankSEG solves every M. Keep historical diagnostics reproducible.
            fallback = (m > getattr(screening, "_MAX_CANDIDATE_FRACTION", math.inf) * dim
                        or m > getattr(screening, "_MAX_CANDIDATE_ELEMENTS", math.inf))
            counts_by_row[row] = (h, dim - h - m, m, fallback)
            kept.append(0 if fallback else m)
            if fallback:
                workspace += dim
        for group in screening._candidate_groups(kept):
            workspace += len(group) * max(kept[row] for row in group)
    records = []
    for channel, label in enumerate(labels):
        is_active = channel in counts_by_row
        h, l, m, fallback = counts_by_row.get(channel, (0, 0, 0, False))
        assert not is_active or h + l + m == dim
        actual_sort = (dim if fallback else m) if is_active else 0
        row = {"channel": channel, "label": label, "total_entries": dim,
               "active_entries": dim if is_active else 0,
               "class_pruned_entries": 0 if is_active else dim,
               "forced_positive": h, "forced_negative": l, "undecided": m,
               "actual_sort_entries": actual_sort, "fallback_rows": int(fallback),
               "active_rows": int(is_active), "total_rows": 1}
        records.append({**row, **fractions(row)})
    return {"rows": records, "sort_workspace_elements": workspace}


def run(args):
    sys.path.insert(0, str(args.rankseg_path.resolve()))
    from rankseg_benchmark.nnunet.config import load_dataset_config
    from rankseg_benchmark.nnunet.io import load_probabilities, restore_cropped_probabilities
    from rankseg_benchmark.nnunet.screening_benchmark import prepare_cases, sha256
    import rankseg

    if Path(rankseg.__file__).resolve().parent != args.rankseg_path.resolve() / "rankseg":
        raise RuntimeError("A different RankSEG checkout was already imported")

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite statistics: {args.output}")
    benchmark = json.loads(args.benchmark_report.read_text())
    if (not benchmark.get("complete") or benchmark.get("subset") is not False
            or type(benchmark.get("force_screening")) is not bool or benchmark.get("dtype") != "float32"):
        raise ValueError("A completed full-cohort screening benchmark is required")
    if not torch.cuda.is_available() or torch.__version__ != benchmark["torch"]:
        raise RuntimeError("Use the same CUDA/PyTorch environment as the benchmark")
    import triton
    if triton.__version__ != benchmark["triton"] or torch.cuda.get_device_name() != benchmark["device"]:
        raise RuntimeError("CUDA device/Triton environment differs from the benchmark")
    for name, digest in benchmark["rankseg_sources"].items():
        if sha256(args.rankseg_path / "rankseg" / name) != digest:
            raise ValueError(f"RankSEG source changed: {name}")
    manifest = Path(benchmark["manifest"])
    if sha256(manifest) != benchmark["manifest_sha256"]:
        raise ValueError("Manifest changed")
    config = load_dataset_config(manifest)
    cases, _, available = prepare_cases(config, 0)
    expected = {r["case_id"]: r for r in benchmark["records"]}
    ids = [c.case_id for c in cases]
    if (ids != benchmark["selected_case_ids"] or len(expected) != available
            or len(benchmark["records"]) != available or set(expected) != set(ids)):
        raise ValueError("Cohort does not match benchmark")
    torch.set_num_threads(4)
    report = {"dataset": benchmark["dataset"], "complete": False,
              "benchmark_report": str(args.benchmark_report.resolve()),
              "benchmark_sha256": sha256(args.benchmark_report), "collector_sha256": sha256(__file__),
              "rankseg_sources": benchmark["rankseg_sources"], "expected_cases": available,
              "torch": torch.__version__, "triton": triton.__version__, "device": benchmark["device"],
              "dtype": "float32", "timed": False,
              "unit": "class-probability entries, not unique spatial voxels",
              "aggregation": "ratios of summed counts; screening denominator excludes class-pruned rows",
              "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    for case in cases:
        old = expected[case.case_id]
        if sha256(case.probabilities) != old["probability_sha256"]:
            raise ValueError(f"Probability cache changed: {case.case_id}")
        probabilities, _ = load_probabilities(case.probabilities, config.probability_key)
        probabilities, restoration = restore_cropped_probabilities(
            probabilities, probability_path=case.probabilities, target_shape=tuple(old["shape"][2:]),
            background_channel=config.channel_labels.index(0))
        if restoration != old["restoration"]:
            raise ValueError(f"Probability restoration differs: {case.case_id}")
        values = torch.from_numpy(np.ascontiguousarray(probabilities, dtype=np.float32)).unsqueeze(0).cuda()
        if list(values.shape) != old["shape"]:
            raise ValueError(f"Shape differs: {case.case_id}")
        if not benchmark["force_screening"]:
            from rankseg import _rankseg_algo as algo
            if (benchmark.get("screening_mode") == "auto"
                    and not algo._rma_dice_use_screening(values, "auto")):
                raise ValueError("Production auto bypass: forced-screening counts would misstate sorting")
            dim, channels = math.prod(values.shape[2:]), values.shape[1]
            if (not hasattr(algo, "_rma_dice_use_screening") and (
                    (channels == 1 and dim <= algo._RMA_CUDA_SCREENING_MIN_SINGLE_CHANNEL_DIM)
                    or (channels <= algo._RMA_CUDA_SCREENING_FEW_ROWS
                        and dim <= algo._RMA_CUDA_SCREENING_MIN_FEW_ROWS_DIM))):
                raise ValueError("Production small-input bypass: forced-screening counts would misstate sorting")
        counts = screening_counts(values, config.channel_labels)
        report["records"].append({"case_id": case.case_id, "fold": old["fold"],
                                  "probability_sha256": old["probability_sha256"], **counts})
        del values, probabilities
        torch.cuda.empty_cache()
        report["summary"] = summarize(report["records"])
        report["complete"] = len(report["records"]) == available
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--rankseg-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
