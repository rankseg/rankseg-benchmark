"""Paired real-cache benchmark plus exact, frozen-input assignment audit.

Uses benchmark_screening_update's arguments, case selection and metrics. The
extra audit runs outside timing and compares both kernels on exactly the same
binary masks and full-image/unique statistics, avoiding upstream CUDA cumsum
repeatability as a confounder. No inference or cache/checkpoint changes.
"""

import importlib.util
from pathlib import Path
import statistics
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "paired_real_benchmark", Path(__file__).with_name("benchmark_screening_update.py"),
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from rankseg import _screening as screening


def kernel_times(calls, repeats=11):
    samples = {name: [] for name in calls}
    for _ in range(3):
        for call in calls.values():
            result = call()
            del result
    for repeat in range(repeats):
        for name in (list(calls) if repeat % 2 == 0 else list(reversed(calls))):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            result = calls[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
            del result
    return {name: statistics.median(values) for name, values in samples.items()}


@torch.no_grad()
def checked_measure(probs, baseline, mode, repeats=11, **kwargs):
    audit = {"calls": 0, "different_pixels": 0, "policies": []}
    original = runner.current._rma_dice_nonoverlap
    new_backend = screening._cuda_backend()
    old_backend = sys.modules["screening_baseline_screening_cuda"]

    def checked(masks, p, means, active, policy, void_index):
        # Check the complete statistics+assignment entrypoint as well.
        old = baseline._rma_dice_nonoverlap(masks, p, means, active, policy, void_index)
        new = original(masks, p, means, active, policy, void_index)
        assert old is not None and new is not None
        assert torch.equal(old, new), "fixed-mask wrapper labels changed"
        del old
        status, n, h = new_backend.unique_statistics(masks, p)
        args = (masks, p, status, means, n, h, active)
        for void, index in ((False, 255), (True, -(2**63)), (True, 2**63 - 1)):
            old = old_backend.dice_nonoverlap(*args, void=void, void_index=index)
            candidate = new_backend.dice_nonoverlap(*args, void=void, void_index=index)
            assert torch.equal(old, candidate), "fixed-statistics kernel labels changed"
            audit["policies"].append({"void": void, "void_index": index})
            del old, candidate
        audit["calls"] += 1
        audit["unique_percent"] = 100 * int(n.sum()) / status.numel()
        audit["assignment_median_ms"] = kernel_times({
            "before": lambda: old_backend.dice_nonoverlap(*args),
            "after": lambda: new_backend.dice_nonoverlap(*args),
        }, repeats=repeats)
        return new

    try:
        runner.current._rma_dice_nonoverlap = checked
        prediction = runner.current.rankseg_rma(probs, output_mode=mode, safe_screening=True)
        del prediction
    finally:
        runner.current._rma_dice_nonoverlap = original
    # No monkeypatches, audit tensors or audit CUDA events remain in timing.
    torch.cuda.synchronize()
    record, predictions = original_measure(probs, baseline, mode, repeats=repeats, **kwargs)
    record["fixed_input_assignment_audit"] = audit
    if mode == "multilabel":
        assert record["different_pixels"] == 0, "binary path unexpectedly changed"
    return record, predictions


original_measure = runner.measure
runner.measure = checked_measure

if __name__ == "__main__":
    runner.main()
