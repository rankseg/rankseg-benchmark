"""The final report must distinguish complete cohorts, OOM subsets and failures."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("acceptance_summary", Path(__file__).resolve().parents[2]
                                              / "scripts/nnunet/summarize_screening_acceptance.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture():
    from .test_screening_audit import fixture as reference_fixture
    report, _, _ = reference_fixture()
    for record in report["records"]:
        for method, value in record["methods"].items():
            value["peak_incremental_mib"] = 8 if method == "screened" else 16
            value["objective"]["status"] = "passed"
    return report


def test_macro_and_complete_coverage():
    report = fixture()
    original = deepcopy(report)
    summary.verify_metrics(report)
    actual = summary.aggregate(report["records"], "screened")
    assert actual["dice"] == actual["iou"] == .5  # Not the .75 case-first macro.
    assert actual["completed"] == 2 and actual["full_coverage"]
    assert actual["mean_ms"] == 2
    assert actual["mean_peak_mib"] == actual["max_peak_mib"] == 8
    assert actual["objective_passed"] == 2
    paired = summary.comparison(report["records"], "full")
    assert paired["cases"] == 2 and paired["speedup"] == 1
    assert paired["mean_peak_reduction_percent"] == 50
    assert report == original


def test_subset_verification_is_explicit_and_does_not_relabel_full_cohort():
    report = fixture()
    report["records"] = report["records"][:1]
    report["selected_case_ids"] = [report["records"][0]["case_id"]]
    report["subset"] = True
    with pytest.raises(ValueError, match="cohort"):
        summary.verify_metrics(report)
    before = deepcopy(report)
    summary.verify_metrics(report, allow_subset=True)
    assert report == before
    report["subset"] = False
    with pytest.raises(ValueError, match="cohort"):
        summary.verify_metrics(report, allow_subset=True)


def test_missing_full_result_not_silently_replaced_by_optimized_full():
    report = fixture()
    report["records"][1]["methods"]["full"] = {"status": "cuda_oom"}
    summary.verify_metrics(report)
    methods = {name: summary.aggregate(report["records"], name) for name in summary.METHODS}
    assert methods["full"]["completed"] == 1 and methods["full"]["oom"] == 1
    assert not methods["full"]["full_coverage"]
    assert methods["full_optimized"]["full_coverage"]
    paired = summary.comparison(report["records"], "full")
    assert paired["cases"] == 1 and paired["screened"]["completed"] == 1


def test_failed_and_missing_objective_never_passed():
    report = fixture()
    objective = report["records"][0]["methods"]["screened"]["objective"]
    objective.update(status="failed", max_regret=5 * 2**-23, max_regret_eps=5.)
    del report["records"][1]["methods"]["screened"]["objective"]
    summary.verify_metrics(report)
    actual = summary.aggregate(report["records"], "screened")
    assert actual["objective_failed"] == actual["objective_unverified"] == 1
    assert actual["objective_passed"] == 0
    result = summary.summarize(report, {"audit": "not_passed"}, None)
    assert not result["screened_numerical_checks_passed"]
    assert len(result["objective_failures"]) == 1


@pytest.mark.parametrize("mutation", ["incomplete", "duplicate", "selection", "count", "metric",
                                      "timing", "objective_status", "objective_eps",
                                      "memory_negative", "memory_nonfinite", "labels_missing",
                                      "labels_duplicate", "reference_duplicate"])
def test_rejects_inconsistent_report(mutation):
    report = fixture()
    result = report["records"][0]["methods"]["screened"]
    if mutation == "incomplete":
        report["complete"] = False
    elif mutation == "duplicate":
        report["records"][1]["case_id"] = report["records"][0]["case_id"]
    elif mutation == "selection":
        report["selected_case_ids"].append("extra")
    elif mutation == "count":
        result["metrics"][0]["tp"] = -1
    elif mutation == "metric":
        result["metrics"][0]["dice"] = .2
    elif mutation == "timing":
        result["median_ms"] = .1
    elif mutation == "objective_status":
        result["objective"]["status"] = "failed"
    elif mutation == "objective_eps":
        result["objective"]["max_regret_eps"] = 999
    elif mutation.startswith("memory"):
        result["peak_incremental_mib"] = -1 if mutation == "memory_negative" else float("nan")
    elif mutation == "labels_missing":
        result["metrics"].pop()
    elif mutation == "labels_duplicate":
        result["metrics"].append(deepcopy(result["metrics"][0]))
    else:
        rows = report["records"][0]["methods"]["argmax"]["metrics"]
        rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):
        summary.verify_metrics(report)


def test_renderer_separates_incomplete_full_from_complete_methods():
    report = fixture()
    report["records"][1]["methods"]["full"] = {"status": "cuda_oom"}
    result = summary.summarize(report, {"metrics_and_timing_audit": "passed"}, None)
    rendered = summary.render({"datasets": {"test": result}})
    assert "| Full (screening off) | 1/2 | — | — | — | OOM: 1 |" in rendered
    assert "| Screened (production) | 2/2 | 50.00 | 50.00 |" in rendered
    assert "| test | Default full | 1 |" in rendered
    assert "| test | Optimized full | 2 |" in rendered


def test_all_empty_foreground_remains_undefined():
    report = fixture()
    for record in report["records"]:
        for method in record["methods"].values():
            for row in method["metrics"]:
                row.update(tp=0, fp=0, fn=0, dice=None, iou=None)
    summary.verify_metrics(report)
    assert summary.aggregate(report["records"], "screened")["dice"] is None
    assert summary.comparison(report["records"], "full")["dice_delta_pp"] is None


@pytest.mark.parametrize("failed_check", [None, "all_screened_numerical_checks_passed",
                                        "all_metrics_timing_audits_passed", "all_screening_diagnostics_complete"])
def test_document_never_promotes_incomplete_checks_to_passed(failed_check):
    report = {"datasets": {}, "all_screened_numerical_checks_passed": True,
              "all_metrics_timing_audits_passed": True, "all_screening_diagnostics_complete": True}
    if failed_check:
        report[failed_check] = False
    template = "# Acceptance\n\n" + summary.PENDING_STATUS + "\n\n## Protocol\n\nPreserved.\n"
    result = summary.completed_document(template, report)
    assert ("acceptance is NOT fully passed" in result) == bool(failed_check)
    assert "## Protocol\n\nPreserved." in result
    assert "## Completed measurements" in result
    assert "unresolved baseline limitations" in result
    with pytest.raises(ValueError, match="refusing"):
        summary.completed_document(result, report)


def test_wait_does_not_accept_running_or_transient_status(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text('{"complete": false}')
    with pytest.raises(TimeoutError, match="no final report"):
        summary.wait_for_completion(path)
    path.write_text('{"complete":')
    with pytest.raises(TimeoutError, match="no final report"):
        summary.wait_for_completion(path)
    sleeps = []
    def finish(seconds):
        sleeps.append(seconds)
        path.write_text('{"complete": true}')
    monkeypatch.setattr(summary.time, "sleep", finish)
    assert summary.wait_for_completion(path, 60) == {"complete": True}
    assert sleeps == [20]
    with pytest.raises(ValueError, match="nonnegative"):
        summary.wait_for_completion(path, -1)
