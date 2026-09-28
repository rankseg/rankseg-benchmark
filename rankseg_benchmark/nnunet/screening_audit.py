# SPDX-License-Identifier: Apache-2.0
# Relocated from rankseg-nnunet-benchmark; see LICENSES/Apache-2.0.txt.
"""Read-only audit of a full-cohort screening report against published evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

METHODS = ("argmax", "full", "full_optimized", "screened")
FLOAT32_EPS = 2 ** -23


def _equal(actual, expected, context):
    a = math.nan if actual is None else float(actual)
    b = math.nan if expected is None else float(expected)
    if not ((math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"Mismatch at {context}: {actual} != {expected}")


def audit_full_report(report, reference, published, methods=METHODS, *, report_objective_failures=False):
    """Recompute counts/aggregation; do not run inference or modify inputs."""
    if not report.get("complete") or report.get("subset") is not False:
        raise ValueError("Expected a completed full-cohort report")
    if (len(methods) < 2 or len(set(methods)) != len(methods) or "argmax" not in methods
            or not set(methods).issubset(METHODS)):
        raise ValueError("Choose argmax and at least one distinct RankSEG method")
    if report["dataset"] != published["dataset"]["id"]:
        raise ValueError("Published dataset identity mismatch")
    labels = set(published["dataset"]["foreground_labels"])
    reference = reference[reference.label.isin(labels)].copy()
    if reference.duplicated(["case_id", "method", "label"]).any():
        raise ValueError("Duplicate reference case/method/label")
    ref = reference.set_index(["case_id", "method", "label"])
    expected_cases = set(reference.case_id)
    records = report["records"]
    observed = [r["case_id"] for r in records]
    selected = report["selected_case_ids"]
    if (len(observed) != len(set(observed)) or set(observed) != expected_cases
            or len(selected) != len(set(selected)) or set(selected) != expected_cases
            or len(observed) != report["available_cases"]
            or len(observed) != published["dataset"]["cases"]):
        raise ValueError("Case cohort differs from the published full cohort")
    if any((case, method, label) not in ref.index for case in expected_cases
           for method in ("argmax", "rankseg") for label in labels):
        raise ValueError("Reference lacks a case/method/label")
    four_way_cases = {r["case_id"] for r in records
                      if all(r["methods"][m].get("status") == "ok" for m in METHODS)}
    if (report["summary"]["paired_complete_cases"] != len(four_way_cases)
            or report["summary"]["attempted_cases"] != len(records)):
        raise ValueError("Incorrect paired cohort count")

    rows = []
    failed_checks = []
    max_regret = {m: 0.0 for m in methods if m != "argmax"}
    for record in records:
        case = record["case_id"]
        record_failed = record.get("objective_check") == "failed"
        if record.get("objective_oom") or (record.get("objective_check") != "passed"
                and not (report_objective_failures and record_failed)):
            raise ValueError(f"Numerical oracle not passed: {case}")
        recorded_failures = [m for m, result in record["methods"].items()
                             if result.get("objective", {}).get("status") == "failed"]
        if record_failed != bool(recorded_failures):
            raise ValueError(f"Inconsistent case/method objective status: {case}")
        # Keep failures in omitted controls visible too. OOM controls have no
        # objective, but completed controls with a failed check must not vanish.
        for method in recorded_failures:
            objective = record["methods"][method]["objective"]
            regret = objective.get("max_regret")
            if not isinstance(regret, (int, float)) or not math.isfinite(regret) or regret <= 4 * FLOAT32_EPS:
                raise ValueError(f"Invalid recorded objective failure: {case}/{method}")
            _equal(objective["max_regret_eps"], regret / FLOAT32_EPS, f"{case}/{method}/eps")
            failed_checks.append({"case_id": case, "method": method, "max_regret_eps": regret / FLOAT32_EPS})
        for method in methods:
            result = record["methods"][method]
            if result.get("status") != "ok":
                raise ValueError(f"Method did not complete: {case}/{method}")
            samples = result["samples_ms"]
            if len(samples) != report["repeats"] or not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError(f"Missing/invalid timing samples: {case}/{method}")
            _equal(result["median_ms"], statistics.median(samples), f"{case}/{method}/median")
            if method != "argmax":
                objective = result["objective"]
                regret = objective["max_regret"]
                if not math.isfinite(regret) or regret < 0:
                    raise ValueError(f"Invalid objective regret: {case}/{method}")
                exceeds_budget = regret > 4 * FLOAT32_EPS
                if exceeds_budget and not (report_objective_failures and method in recorded_failures):
                    raise ValueError(f"Objective outside regression budget: {case}/{method}")
                if objective.get("status", "passed") != ("failed" if exceeds_budget else "passed"):
                    raise ValueError(f"Incorrect objective status: {case}/{method}")
                _equal(objective["max_regret_eps"], regret / FLOAT32_EPS, f"{case}/{method}/eps")
                max_regret[method] = max(max_regret[method], objective["max_regret_eps"])
            metrics = result["metrics"]
            if len(metrics) != len(labels) or {r["label"] for r in metrics} != labels:
                raise ValueError(f"Incorrect foreground label set: {case}/{method}")
            for row in metrics:
                counts = [row[k] for k in ("tp", "fp", "fn")]
                if any(type(x) is not int or x < 0 for x in counts):
                    raise ValueError(f"Invalid confusion counts: {case}/{method}")
                tp, fp, fn = counts
                dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else math.nan
                iou = tp / (tp + fp + fn) if tp + fp + fn else math.nan
                _equal(row["dice"], dice, f"{case}/{method}/Dice")
                _equal(row["iou"], iou, f"{case}/{method}/IoU")
                if method == "argmax":
                    prior = ref.loc[(case, "argmax", row["label"])]
                    if any(row[k] != int(prior[k]) for k in ("tp", "fp", "fn")):
                        raise ValueError(f"Argmax counts differ from original benchmark: {case}/{row['label']}")
                rows.append({"case_id": case, "method": method, "label": row["label"], "dice": dice, "iou": iou})

    # Independent pandas aggregation, not the benchmark's metric_summary function.
    frame = pd.DataFrame(rows)
    current = frame.groupby(["method", "label"])[["dice", "iou"]].mean().groupby("method").mean()
    paired = frame[frame.case_id.isin(four_way_cases)]
    paired_macro = paired.groupby(["method", "label"])[["dice", "iou"]].mean().groupby("method").mean()
    original = reference.groupby(["method", "label"])[["dice", "iou"]].mean().groupby("method").mean()
    for method in ("argmax", "rankseg"):
        for metric in ("dice", "iou"):
            _equal(original.loc[method, metric], published["metrics"]["macro"][method][f"foreground_mean_{metric}"],
                   f"published/{method}/{metric}")
    timings = {}
    for method in methods:
        summary = report["summary"]["methods"][method]
        for metric in ("dice", "iou"):
            value = paired_macro.loc[method, metric] if four_way_cases else math.nan
            _equal(value, summary[metric], f"summary/{method}/{metric}")
        paired_times = [r["methods"][method]["median_ms"] for r in records if r["case_id"] in four_way_cases]
        _equal(summary["mean_ms"], statistics.mean(paired_times) if paired_times else None,
               f"summary/{method}/mean_ms")
        timings[method] = {"mean_ms": statistics.mean(r["methods"][method]["median_ms"] for r in records)}
    return {"dataset": report["dataset"], "cases": len(records),
            "audit": "failed_objective" if failed_checks else "passed",
            "metrics_and_timing_audit": "passed", "objective_failures": failed_checks,
            "audited_methods": list(methods), "four_path_complete_cases": len(four_way_cases),
            "other_methods_status": {m: dict(Counter(r["methods"][m]["status"] for r in records))
                                     for m in METHODS if m not in methods},
            "argmax_confusion_counts": "identical to original benchmark for every case/foreground label",
            "methods": current.to_dict("index"), "original_published": original.to_dict("index"),
            "timing": timings, "max_objective_regret_eps": max_regret}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--published-summary", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                        help="Require these methods on EVERY case; never silently drop OOM cases")
    parser.add_argument("--report-objective-failures", action="store_true",
                        help="Audit complete measurements but report recorded numerical failures as failed, not passed")
    args = parser.parse_args()
    result = audit_full_report(json.loads(args.report.read_text()), pd.read_csv(args.reference_csv),
                               json.loads(args.published_summary.read_text()), tuple(args.methods),
                               report_objective_failures=args.report_objective_failures)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
