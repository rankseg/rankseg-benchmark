"""Paired real-cache benchmark with exact candidate replay checks, outside timing.

Uses benchmark_screening_update's arguments, cohorts and metric conventions.
The baseline snapshot must predate compact candidate replay. Both implementations
receive identical statistics, then independently construct masks and pack every
retained candidate in stable row-major order. No cache or checkpoint writes.
"""

import importlib.util
from pathlib import Path
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "paired_candidate_benchmark", Path(__file__).with_name("benchmark_screening_update.py"),
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from rankseg import _screening as screening


@torch.no_grad()
def checked_measure(probs, baseline, mode, repeats=11, **kwargs):
    backend = screening._cuda_backend()
    previous = sys.modules["screening_baseline_screening_cuda"]
    original = backend.screening_candidates
    audit = {"calls": 0, "different_candidates": 0, "rows": []}

    def checked(*args, **options):
        result = original(*args, **options)
        assert isinstance(result[1], backend._CandidateBounds), "dense candidates unexpectedly allocated"
        old_options = dict(options)
        old_options.pop("materialize_candidates")
        state = old_options.get("statistics")
        if state is not None:
            # Share only read-only statistics; independently construct H and M.
            old_options["statistics"] = (torch.empty_like(state[0]), *state[1:])
        reference = previous.screening_candidates(*args, **old_options)
        for position in (0, 2, 3, 4):
            assert torch.equal(result[position], reference[position]), "H/statistics/block counts changed"
        rows = args[0]
        offsets = result[4].cumsum(-1)
        raw_lengths = offsets[:, -1].cpu().tolist()
        lengths = [n if n <= getattr(screening, "_MAX_CANDIDATE_FRACTION", float("inf")) * rows.shape[-1]
                   and n <= getattr(screening, "_MAX_CANDIDATE_ELEMENTS", float("inf")) else 0 for n in raw_lengths]
        before = previous.pack_candidates(reference[1], offsets, lengths)
        after = backend.pack_candidates(result[1], offsets, lengths)
        assert torch.equal(before, after), "packed candidate membership/order changed"
        active = args[4] if len(args) > 4 else options.get("active")
        active_rows = rows.shape[0] if active is None else int(active.sum())
        active_entries = active_rows * rows.shape[-1]
        audit["rows"].append({
            "shape": list(rows.shape), "raw_candidates": sum(raw_lengths),
            "retained_candidates": sum(lengths), "active_rows": active_rows,
            "fallback_rows": sum(n != kept for n, kept in zip(raw_lengths, lengths)),
            "screened_active_percent": 100 * (1 - sum(raw_lengths) / active_entries) if active_entries else None,
            "nonempty_blocks_percent": 100 * int(torch.count_nonzero(result[4])) / result[4].numel(),
            "dense_candidate_mask_mib_avoided": rows.numel() / 2**20,
        })
        audit["calls"] += 1
        return result

    try:
        backend.screening_candidates = checked
        prediction = runner.current.rankseg_rma(probs, output_mode=mode, safe_screening=True)
        del prediction
    finally:
        backend.screening_candidates = original
    # Audit tensors and hooks are gone before any timed/peak-memory call.
    torch.cuda.synchronize()
    report, predictions = original_measure(probs, baseline, mode, repeats=repeats, **kwargs)
    report["candidate_replay_audit"] = audit
    return report, predictions


original_measure = runner.measure
runner.measure = checked_measure

if __name__ == "__main__":
    runner.main()
