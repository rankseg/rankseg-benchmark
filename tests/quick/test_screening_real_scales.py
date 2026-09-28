"""Real-scale reports separate all-pruned shortcuts and reject invalid evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("real_scales", ROOT / "scripts/summarize_real_screening_scales.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.fixture
def report():
    result = dict(complete=True, real_only=True, sampling="uniform", device="cpu", torch="test", cpu="test",
                  source_hashes={"test": "unchanged"}, threads=[4], rounds=3, repeats=3, profiles=[], records=[])
    for index, (off, on, active) in enumerate([(10., 5., True), (2., .5, False)]):
        identity = f"sample-{index}"
        result["profiles"].append(dict(id=identity, dataset="test", kind="real_derived", tested_dims=[4],
                                       batch=1, channels=2, source_input_sha256=identity, draw_order_sha256=identity))
        rounds = [{m: {"samples_ms": [ms]*3, "median_ms": ms} for m, ms in (("full", off), ("screened", on))}
                  for _ in range(3)]
        result["records"].append(dict(profile=identity, dataset="test", kind="real_derived", dim=4, threads=4,
            batch=1, channels=2, input_sha256=identity, rounds=rounds,
            timing=summary.calibration.timing_summary(rounds),
            objectives={m: {"status": "passed", "max_regret_eps": .1} for m in ("full", "screened")},
            different_pixels=0, prediction_elements=4,
            counts=dict(total_entries=8, active_entries=4 if active else 0, active_rows=int(active),
                        forced_positive=1 if active else 0, forced_negative=2 if active else 0,
                        undecided=1 if active else 0, class_pruned_entries=4 if active else 8,
                        fallback_rows=0)))
    return result


def test_aggregation_separates_shortcuts_and_uses_ratio_of_mean_times(report):
    before = deepcopy(report)
    result = summary.summarize([report])
    assert report == before
    assert result["objective_checks"] == 4 and result["objective_failures"] == 0
    group = result["results"][0]
    assert group["all_inputs"]["speedup"] == pytest.approx(12/5.5)
    assert group["active_inputs"]["speedup"] == 2.
    assert group["all_pruned_inputs"] == 1
    assert group["screening_fraction_active"] == .75
    assert group["class_pruned_fraction_all"] == .75


def test_no_active_inputs_is_not_a_screening_success(report):
    value = summary.aggregate([report["records"][1]])
    assert value["active_inputs"] is None and value["screening_fraction_active"] is None
    assert value["all_pruned_inputs"] == 1


@pytest.mark.parametrize("kind", ["incomplete", "synthetic", "duplicate", "missing", "rounds", "median",
                                  "timing", "objective", "partition", "profile"])
def test_invalid_report_rejected(report, kind):
    row = report["records"][0]
    if kind == "incomplete":
        report["complete"] = False
    elif kind == "synthetic":
        report["profiles"][0]["kind"] = "synthetic"
    elif kind == "duplicate":
        report["records"].append(deepcopy(row))
    elif kind == "missing":
        report["records"].pop()
    elif kind == "rounds":
        row["rounds"].pop()
    elif kind == "median":
        row["rounds"][0]["full"]["median_ms"] = 100.
    elif kind == "timing":
        row["rounds"][0]["full"]["samples_ms"][0] = float("nan")
    elif kind == "objective":
        row["objectives"]["screened"]["max_regret_eps"] = 5.
    elif kind == "partition":
        row["counts"]["forced_positive"] += 1
    else:
        row["channels"] = 3
    with pytest.raises(ValueError):
        summary.audit_report(report)


def test_failed_objective_remains_visible(report):
    report["records"][0]["objectives"]["screened"] = {"status": "failed", "max_regret_eps": 5.}
    result = summary.summarize([report])
    assert result["objective_failures"] == 1 and result["max_regret_eps"] == 5.


def test_cross_thread_comparison_requires_identical_subsamples(report):
    other = deepcopy(report)
    other["threads"] = [8]
    for row in other["records"]:
        row["threads"] = 8
    result = summary.summarize([report, other])
    assert result["configurations"] == 4
    other["records"][0]["input_sha256"] = "different-subsample"
    with pytest.raises(ValueError, match="Subsample"):
        summary.summarize([report, other])


def test_duplicate_runs_and_changed_permutations_rejected(report):
    with pytest.raises(ValueError, match="Duplicate configuration"):
        summary.summarize([report, report])
    other = deepcopy(report)
    other["profiles"][0]["draw_order_sha256"] = "different-permutation"
    with pytest.raises(ValueError, match="differ"):
        summary.summarize([report, other])
