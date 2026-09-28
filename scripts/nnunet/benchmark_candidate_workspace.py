"""Small paired real-cache benchmark with exact candidate movement audits.

Reuses benchmark_screening_validation's CLI, cohorts and exact-statistics audit.
Gather inputs and complete mask writeback are checked outside timed calls.
"""
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "validation_cohorts", Path(__file__).with_name("benchmark_screening_validation.py"),
)
cohorts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohorts)
runner = cohorts.runner
from rankseg import _screening as screening

original_measure = runner.measure


@torch.no_grad()
def audited_measure(probs, baseline, mode, repeats=11, **kwargs):
    backend = screening._cuda_backend()
    original_gather = backend.gather_candidate_group
    original_scatter = backend.scatter_candidate_group
    audit = dict(gather_calls=0, scatter_calls=0, shapes=[], different_values=0, different_writes=0)

    def gather(rows, packed, group, starts, lengths, means, n_forced, forced_mass):
        result = original_gather(rows, packed, group, starts, lengths, means, n_forced, forced_mass)
        values, mu, n, h, metadata = result
        expected = rows.new_full((len(group), max(lengths)), -1)
        for i, row in enumerate(group):
            expected[i, :lengths[i]] = rows[row, packed[starts[i]:starts[i] + lengths[i]] - row * rows.shape[1]]
        if len(group) == 1:
            expected = expected[0]
        assert torch.equal(values, expected), "sort input/order changed"
        assert torch.equal(mu, means[group]) and torch.equal(n, n_forced[group]) and torch.equal(h, forced_mass[group])
        audit["gather_calls"] += 1
        audit["shapes"].append([len(group), max(lengths)])
        return result

    def scatter(masks, packed, metadata, order, opt):
        expected = masks.clone()
        group, starts, lengths = metadata.cpu().tolist()
        width = order.shape[-1]
        for i, row in enumerate(group):
            permutation = order if len(group) == 1 else order[i]
            tau = opt if len(group) == 1 else opt[i]
            selected = torch.empty(width, device=masks.device, dtype=torch.bool)
            selected.scatter_(0, permutation, torch.arange(width, device=masks.device) < tau)
            expected.reshape(-1)[packed[starts[i]:starts[i] + lengths[i]]] = selected[:lengths[i]]
        original_scatter(masks, packed, metadata, order, opt)
        assert torch.equal(masks, expected), "mask writeback changed or wrote outside candidates"
        audit["scatter_calls"] += 1

    try:
        backend.gather_candidate_group = gather
        backend.scatter_candidate_group = scatter
        prediction = runner.current.rankseg_rma(probs, output_mode=mode, safe_screening=True)
        del prediction
    finally:
        backend.gather_candidate_group = original_gather
        backend.scatter_candidate_group = original_scatter
    assert audit["gather_calls"] == audit["scatter_calls"]
    # All diagnostic clones and hooks have been released before timing/peaks.
    report, predictions = original_measure(probs, baseline, mode, repeats=repeats, **kwargs)
    report["workspace_audit"] = audit
    return report, predictions


runner.measure = audited_measure
if __name__ == "__main__":
    runner.main()
