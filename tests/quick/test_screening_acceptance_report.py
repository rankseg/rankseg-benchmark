"""Post-run audit must reject incorrect acceptance claims or aggregates."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from .test_screening_acceptance import bench, summary_fixture

spec = importlib.util.spec_from_file_location("mini_report", Path(__file__).resolve().parents[2]
                                              / "scripts/summarize_screening_acceptance.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


@pytest.fixture(scope="module")
def helpers():
    root = Path(__file__).resolve().parents[3]
    return bench.load_helpers(Path(__import__("rankseg").__file__).resolve().parents[1], Path(__file__).resolve().parents[2])


@pytest.fixture
def report(helpers):
    _, counts = helpers
    value = summary_fixture()
    value.update(complete=True, repeats=1)
    value["sources"][0]["fold"] = None
    record = value["records"][0]
    record["shape"] = [1, 2, 1, 3]
    for method in record["methods"].values():
        method["samples_ms"] = [1.]
        method["objective"]["max_regret"] = 0.
    value["summary"] = bench.summarize(value, counts)
    return value


def test_valid_report_is_read_only(report):
    before = deepcopy(report)
    reporter.audit(report)
    assert report == before


@pytest.mark.parametrize("kind", ["incomplete", "missing", "duplicate", "timing", "memory", "counts",
                                  "objective", "objective_summary", "flag", "metric_summary", "partition"])
def test_rejects_invalid_report(report, kind):
    row = report["records"][0]
    screened = row["methods"]["screened"]
    if kind == "incomplete":
        report["complete"] = False
    elif kind == "missing":
        report["records"] = []
    elif kind == "duplicate":
        report["records"].append(deepcopy(row))
    elif kind == "timing":
        screened["samples_ms"] = [2.]
    elif kind == "memory":
        screened["peak_mib"] = float("nan")
    elif kind == "counts":
        screened["counts"]["tp"][0] = -1
    elif kind == "objective":
        screened["objective"]["max_regret_eps"] = 999.
    elif kind == "objective_summary":
        report["summary"]["methods"]["screened"]["objective_passed"] = 100
    elif kind == "flag":
        report["summary"]["screened_checks_passed"] = False
    elif kind == "metric_summary":
        report["summary"]["methods"]["screened"]["dice"] = .1
    else:
        row["screening_counts"]["rows"][0]["forced_positive"] = 2
    with pytest.raises(ValueError):
        reporter.audit(report)


def test_finite_objective_failure_remains_a_failure(report, helpers):
    _, counts = helpers
    report["records"][0]["methods"]["screened"]["objective"] = {
        "status": "failed", "max_regret": 5 * 2**-23, "max_regret_eps": 5.}
    report["summary"] = bench.summarize(report, counts)
    reporter.audit(report)
    summary = {"all_screened_checks_passed": False, "all_forced_screening_checks_passed": True,
               "datasets": {"test": report["summary"]}}
    assert "NOT all passed" in reporter.render(summary)
