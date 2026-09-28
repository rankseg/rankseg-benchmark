"""Small real-cache paired benchmark with exact fused-statistics audits.

Reuses benchmark_screening_update's CLI. nnunet intentionally selects one
representative per dataset plus liver_43, not complete folds or cohorts.
"""
import importlib.util
from pathlib import Path
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "paired_validation_benchmark", Path(__file__).with_name("benchmark_screening_update.py"),
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from rankseg import _screening as screening

original_measure = runner.measure


@torch.no_grad()
def audited_measure(probs, baseline, mode, repeats=11, **kwargs):
    old = sys.modules["screening_baseline_screening_cuda"]
    new = screening._cuda_backend()
    rows = probs.reshape(probs.shape[0] * probs.shape[1], -1)
    guard = screening._ROUNDING_GUARD * torch.finfo(rows.dtype).eps
    before = old.screening_statistics(rows, guard)
    maximum, after = new.validated_screening_statistics(rows, guard)
    assert maximum == float(rows.max())
    for a, b in zip((*before[:2], *before[2][1:]), (*after[:2], *after[2][1:])):
        assert torch.equal(a, b), "fused validation changed screening statistics/bounds"
    del before, after
    report, predictions = original_measure(probs, baseline, mode, repeats=repeats, **kwargs)
    report["statistics_exact"] = True
    return report, predictions


original_medical = runner.nnunet_cases


def small_medical(args, baseline):
    import rankseg_benchmark.nnunet.screening_benchmark as data
    original_prepare = data.prepare_cases
    def first(config, limit):
        cases, audit, available = original_prepare(config, limit)
        # The zero-limit call separately locates the large liver regression.
        return (cases[:1] if limit else cases), audit, available
    data.prepare_cases = first
    try:
        yield from original_medical(args, baseline)
    finally:
        data.prepare_cases = original_prepare


runner.measure = audited_measure
runner.nnunet_cases = small_medical
if __name__ == "__main__":
    runner.main()
