"""Paired decoder benchmark and exact cached-identity/statistics audit.

Uses existing held-out probability caches, never inference or checkpoints.
Diagnostic hooks and per-stage timings are excluded from whole-decoder timing.
"""

import importlib.util
from pathlib import Path
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "unique_benchmark", Path(__file__).with_name("benchmark_unique_assignment.py"),
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
runner = helper.runner
original_measure = helper.original_measure

from rankseg import _screening as screening


@torch.no_grad()
def audit_assignment(old, new, masks, probs, means, active, repeats):
    status, n, h = old.unique_statistics(masks.contiguous(), probs)
    cached, new_n, new_h = new.unique_statistics(masks.contiguous(), probs, cache_unique=True)
    decoded = (torch.where(cached >= 2, torch.ones_like(cached), cached * 2)
               if probs.shape[1] <= 254 else cached)
    assert torch.equal(status, decoded), "pixel classification changed"
    assert torch.equal(n, new_n), "unique counts changed"
    assert torch.equal(h, new_h), "unique floating sums changed"
    del decoded
    old_args = masks, probs, status, means, n, h, active
    new_args = masks, probs, cached, means, new_n, new_h, active
    policies = ((False, 255), (True, -(2**63)), (True, 2**63 - 1))
    for void, index in policies:
        a = old.dice_nonoverlap(*old_args, void=void, void_index=index)
        b = new.dice_nonoverlap(*new_args, void=void, void_index=index, cache_unique=True)
        assert torch.equal(a, b), "fixed-statistics assignment changed"
        del a, b
    wrappers = {
        "before": lambda: old.dice_nonoverlap_from_masks(masks, probs, means, active),
        "after": lambda: new.dice_nonoverlap_from_masks(masks, probs, means, active),
    }
    a, b = wrappers["before"](), wrappers["after"]()
    assert torch.equal(a, b), "fixed-mask wrapper changed"
    del a, b
    stages = {
        "statistics": {
            "before": lambda: old.unique_statistics(masks, probs),
            "after": lambda: new.unique_statistics(masks, probs, cache_unique=True),
        },
        "assignment": {
            "before": lambda: old.dice_nonoverlap(*old_args),
            "after": lambda: new.dice_nonoverlap(*new_args, cache_unique=True),
        },
        "statistics_and_assignment": wrappers,
    }
    times = {name: helper.kernel_times(calls, repeats=repeats) for name, calls in stages.items()}
    peaks = {}
    for name, call in wrappers.items():
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = call()
        torch.cuda.synchronize()
        peaks[name] = (torch.cuda.max_memory_allocated() - resident) / 2**20
        del result
    return dict(statistics_exact=True, different_pixels=0, policies=len(policies),
                unique_percent=100 * int(n.sum()) / status.numel(),
                stage_median_ms=times, wrapper_peak_mib=peaks)


@torch.no_grad()
def checked_measure(probs, baseline, mode, repeats=11, **kwargs):
    audit = []
    original = runner.current._rma_dice_nonoverlap
    new = screening._cuda_backend()
    old = sys.modules["screening_baseline_screening_cuda"]

    def checked(masks, p, means, active, policy, void_index):
        audit.append(audit_assignment(old, new, masks, p, means, active, repeats))
        return original(masks, p, means, active, policy, void_index)

    try:
        runner.current._rma_dice_nonoverlap = checked
        prediction = runner.current.rankseg_rma(probs, output_mode=mode, safe_screening=True)
        del prediction
    finally:
        runner.current._rma_dice_nonoverlap = original
    record, predictions = original_measure(probs, baseline, mode, repeats=repeats, **kwargs)
    record["cached_unique_audit"] = audit
    if mode == "multilabel":
        assert not audit and record["different_pixels"] == 0
    return record, predictions


runner.measure = checked_measure
if __name__ == "__main__":
    runner.main()
