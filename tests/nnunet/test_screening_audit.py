from copy import deepcopy

import pandas as pd
import pytest

from rankseg_benchmark.nnunet.screening_audit import METHODS, audit_full_report


def fixture():
    reference, records = [], []
    # Label 2 is empty in the second case: label macro = .5, case macro = .75.
    for i in range(2):
        metrics = [{"label": 1, "tp": 1, "fp": 0, "fn": 0, "dice": 1., "iou": 1.},
                   {"label": 2, "tp": 0, "fp": 0, "fn": 1 - i,
                    "dice": 0. if i == 0 else None, "iou": 0. if i == 0 else None}]
        methods = {m: {"status": "ok", "samples_ms": [1., 2., 3.], "median_ms": 2.,
                       "objective": {"max_regret": 0., "max_regret_eps": 0.},
                       "metrics": deepcopy(metrics)} for m in METHODS}
        records.append({"case_id": str(i), "objective_check": "passed", "methods": methods})
        reference.extend({"case_id": str(i), "method": m, **row} for m in ("argmax", "rankseg") for row in metrics)
    report = {"complete": True, "subset": False, "cases_per_fold": 0, "dataset": "test", "available_cases": 2,
              "selected_case_ids": ["0", "1"], "records": records, "repeats": 3,
              "summary": {"paired_complete_cases": 2, "attempted_cases": 2,
                          "methods": {m: {"dice": .5, "iou": .5, "mean_ms": 2.} for m in METHODS}}}
    published = {"dataset": {"id": "test", "cases": 2, "foreground_labels": [1, 2]},
                 "metrics": {"macro": {m: {"foreground_mean_dice": .5, "foreground_mean_iou": .5}
                                       for m in ("argmax", "rankseg")}}}
    return report, pd.DataFrame(reference), published


def test_independent_audit_and_no_mutation():
    report, reference, published = fixture()
    before = deepcopy(report)
    result = audit_full_report(report, reference, published)
    assert result["audit"] == "passed" and result["cases"] == 2
    assert result["methods"]["screened"]["dice"] == .5
    assert report == before
    # Actual case coverage proves completeness, not the selection CLI spelling.
    report["cases_per_fold"] = 4
    assert audit_full_report(report, reference, published)["audit"] == "passed"


def test_explicit_three_way_audit_keeps_every_case_when_original_full_ooms():
    report, reference, published = fixture()
    report["records"][1]["methods"]["full"] = {"status": "cuda_oom"}
    report["summary"]["paired_complete_cases"] = 1
    with pytest.raises(ValueError, match="did not complete"):
        audit_full_report(report, reference, published)
    result = audit_full_report(report, reference, published, ("argmax", "full_optimized", "screened"))
    assert result["cases"] == 2 and result["four_path_complete_cases"] == 1
    assert result["other_methods_status"]["full"] == {"ok": 1, "cuda_oom": 1}
    assert result["methods"]["screened"]["dice"] == .5
    # Requested methods must still succeed on every case.
    report["records"][1]["methods"]["screened"] = {"status": "cuda_oom"}
    report["summary"]["paired_complete_cases"] = 1
    with pytest.raises(ValueError, match="did not complete"):
        audit_full_report(report, reference, published, ("argmax", "full_optimized", "screened"))


def test_explicit_failure_reporting_keeps_full_cohort_and_never_claims_passed():
    report, reference, published = fixture()
    record = report["records"][0]
    record["objective_check"] = "failed"
    record["methods"]["full"]["objective"] = {
        "status": "failed", "max_regret": 5 * 2**-23, "max_regret_eps": 5.,
    }
    with pytest.raises(ValueError, match="oracle not passed"):
        audit_full_report(report, reference, published)
    before = deepcopy(report)
    for methods in (METHODS, ("argmax", "full_optimized", "screened")):
        result = audit_full_report(report, reference, published, methods, report_objective_failures=True)
        assert result["audit"] == "failed_objective"
        assert result["metrics_and_timing_audit"] == "passed"
        assert result["cases"] == 2 and result["methods"]["screened"]["dice"] == .5
        assert result["objective_failures"] == [{"case_id": "0", "method": "full", "max_regret_eps": 5.}]
    assert report == before


@pytest.mark.parametrize("mutation", ["missing_status", "wrong_eps", "too_small", "nan", "oom",
                                      "falsely_passed", "hidden_failure", "bad_counts"])
def test_failure_reporting_still_rejects_invalid_evidence(mutation):
    report, reference, published = fixture()
    record = report["records"][0]
    record["objective_check"] = "failed"
    objective = {"status": "failed", "max_regret": 5 * 2**-23, "max_regret_eps": 5.}
    record["methods"]["screened"]["objective"] = objective
    if mutation == "missing_status":
        del objective["status"]
    elif mutation == "wrong_eps":
        objective["max_regret_eps"] = 1.
    elif mutation == "too_small":
        objective.update(max_regret=0., max_regret_eps=0.)
    elif mutation == "nan":
        objective["max_regret"] = float("nan")
    elif mutation == "oom":
        record["objective_oom"] = True
    elif mutation == "falsely_passed":
        record["objective_check"] = "passed"
    elif mutation == "hidden_failure":
        record["objective_check"] = "passed"
        objective["status"] = "passed"
    elif mutation == "bad_counts":
        record["methods"]["argmax"]["metrics"][0]["tp"] = 5
    with pytest.raises(ValueError):
        audit_full_report(report, reference, published, report_objective_failures=True)


@pytest.mark.parametrize("mutation", ["partial", "subset", "missing_case", "duplicate_case", "bad_counts",
                                      "wrong_aggregation", "bad_median", "oom", "oracle_failed", "oracle_missing",
                                      "regret", "wrong_label", "published", "reference_duplicate"])
def test_rejects_invalid_report(mutation):
    report, reference, published = fixture()
    result = report["records"][0]["methods"]["screened"]
    if mutation == "partial":
        report["complete"] = False
    elif mutation == "subset":
        report["subset"] = True
    elif mutation == "missing_case":
        report["records"].pop()
    elif mutation == "duplicate_case":
        report["records"][1]["case_id"] = "0"
    elif mutation == "bad_counts":
        report["records"][0]["methods"]["argmax"]["metrics"][0]["tp"] = 2
    elif mutation == "wrong_aggregation":
        report["summary"]["methods"]["screened"]["dice"] = .75
    elif mutation == "bad_median":
        result["median_ms"] = 4
    elif mutation == "oom":
        result["status"] = "cuda_oom"
    elif mutation == "oracle_failed":
        report["records"][0]["objective_check"] = "failed"
    elif mutation == "oracle_missing":
        report["records"][0]["objective_check"] = "cuda_oom_not_verified"
    elif mutation == "regret":
        result["objective"]["max_regret"] = 1e-3
    elif mutation == "wrong_label":
        result["metrics"][0]["label"] = 0
    elif mutation == "published":
        published["metrics"]["macro"]["rankseg"]["foreground_mean_dice"] = .9
    else:
        reference = pd.concat([reference, reference.iloc[:1]])
    with pytest.raises(ValueError):
        audit_full_report(report, reference, published)
