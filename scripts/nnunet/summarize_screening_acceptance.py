"""Independent aggregation and final report for the complete-cohort acceptance.

Never promotes an incomplete/OOM method to full-cohort coverage, never merges
different methods into one baseline, and never treats recorded failures as passed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

METHODS = ("argmax", "full", "full_optimized", "screened")
NAMES = {"argmax": "Argmax", "full": "Full (screening off)",
         "full_optimized": "Optimized full (auxiliary)", "screened": "Screened (production)"}
COHORTS = {"liver": 131, "pancreas": 281, "hepaticvessel": 303, "lung": 63}


def finite_mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(values) if values else None


def aggregate(records, method, case_ids=None):
    selected = [r for r in records if case_ids is None or r["case_id"] in case_ids]
    successful = [r for r in selected if r["methods"][method]["status"] == "ok"]
    rows = [r["methods"][method] for r in successful]
    labels = sorted({m["label"] for row in rows for m in row["metrics"]})
    by_label = {}
    for label in labels:
        metrics = [m for row in rows for m in row["metrics"] if m["label"] == label]
        by_label[str(label)] = {key: finite_mean(m[key] for m in metrics) for key in ("dice", "iou")}
    objective = [row.get("objective", {}) for row in rows] if method != "argmax" else []
    return dict(attempted=len(selected), completed=len(rows), full_coverage=len(rows) == len(selected),
                oom=sum(r["methods"][method]["status"] == "cuda_oom" for r in selected),
                dice=finite_mean(v["dice"] for v in by_label.values()),
                iou=finite_mean(v["iou"] for v in by_label.values()), per_label=by_label,
                mean_ms=finite_mean(r["median_ms"] for r in rows),
                mean_peak_mib=finite_mean(r["peak_incremental_mib"] for r in rows),
                max_peak_mib=max((r["peak_incremental_mib"] for r in rows), default=None),
                objective_passed=sum(r.get("status") == "passed" for r in objective),
                objective_failed=sum(r.get("status") == "failed" for r in objective),
                objective_unverified=sum(r.get("status") not in ("passed", "failed") for r in objective),
                max_regret_eps=max((r["max_regret_eps"] for r in objective if "max_regret_eps" in r), default=None))


def comparison(records, baseline):
    paired = {r["case_id"] for r in records
              if all(r["methods"][m]["status"] == "ok" for m in (baseline, "screened"))}
    before, after = (aggregate(records, m, paired) for m in (baseline, "screened"))
    changes = []
    for r in records:
        if r["case_id"] not in paired:
            continue
        a = {m["label"]: m for m in r["methods"][baseline]["metrics"]}
        b = {m["label"]: m for m in r["methods"]["screened"]["metrics"]}
        for label in a:
            if a[label]["dice"] is not None and b[label]["dice"] is not None:
                changes.append(dict(case_id=r["case_id"], label=label,
                                    dice_delta_pp=100 * (b[label]["dice"] - a[label]["dice"])))
    return dict(cases=len(paired), baseline=before, screened=after,
                speedup=before["mean_ms"] / after["mean_ms"] if paired else None,
                mean_peak_reduction_percent=100 * (1 - after["mean_peak_mib"] / before["mean_peak_mib"])
                    if paired and before["mean_peak_mib"] else None,
                dice_delta_pp=100 * (after["dice"] - before["dice"])
                    if before["dice"] is not None and after["dice"] is not None else None,
                iou_delta_pp=100 * (after["iou"] - before["iou"])
                    if before["iou"] is not None and after["iou"] is not None else None,
                worst_case_label=min(changes, key=lambda r: r["dice_delta_pp"], default=None))


def verify_metrics(report, *, allow_subset=False):
    records = report["records"]
    ids = [r["case_id"] for r in records]
    if (not report["complete"] or not ids
            or (not allow_subset and (report["subset"] or len(ids) != report["available_cases"]))
            or (allow_subset and (len(ids) > report["available_cases"]
                                  or report["subset"] != (len(ids) < report["available_cases"])))
            or len(set(ids)) != len(ids) or len(report["selected_case_ids"]) != len(ids)
            or set(ids) != set(report["selected_case_ids"])):
        raise ValueError("Incomplete or inconsistent cohort")
    for r in records:
        reference_labels = [m["label"] for m in r["methods"]["argmax"]["metrics"]]
        if not reference_labels or len(reference_labels) != len(set(reference_labels)):
            raise ValueError("Invalid reference labels")
        for name in METHODS:
            method = r["methods"][name]
            if method["status"] == "cuda_oom":
                continue
            if method["status"] != "ok":
                raise ValueError("Unknown method status")
            times = method["samples_ms"]
            if len(times) != report["repeats"] or any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError("Invalid timing samples")
            if statistics.median(times) != method["median_ms"]:
                raise ValueError("Invalid timing median")
            peak = method["peak_incremental_mib"]
            if not math.isfinite(peak) or peak < 0:
                raise ValueError("Invalid memory measurement")
            labels = [m["label"] for m in method["metrics"]]
            if len(labels) != len(reference_labels) or set(labels) != set(reference_labels):
                raise ValueError("Inconsistent foreground labels")
            objective = method.get("objective")
            if objective is not None:
                regret = objective["max_regret"]
                expected = "passed" if regret <= 4 * 2**-23 else "failed"
                if (not math.isfinite(regret) or regret < 0 or objective.get("status") != expected
                        or objective["max_regret_eps"] != regret / 2**-23):
                    raise ValueError("Invalid objective accounting")
            for m in method["metrics"]:
                tp, fp, fn = (m[k] for k in ("tp", "fp", "fn"))
                if any(type(v) is not int or v < 0 for v in (tp, fp, fn)):
                    raise ValueError("Invalid confusion counts")
                for metric, numerator, denominator in (("dice", 2 * tp, 2 * tp + fp + fn),
                                                       ("iou", tp, tp + fp + fn)):
                    expected = numerator / denominator if denominator else None
                    if expected != m[metric]:
                        raise ValueError("Metric disagrees with confusion counts")


def summarize(report, audit, proportions, *, allow_subset=False):
    verify_metrics(report, allow_subset=allow_subset)
    methods = {m: aggregate(report["records"], m) for m in METHODS}
    failures = [{"case_id": r["case_id"], "method": m, **v["objective"]}
                for r in report["records"] for m, v in r["methods"].items()
                if v.get("objective", {}).get("status") == "failed"]
    screened = methods["screened"]
    result = dict(cases=len(report["records"]), methods=methods,
                  versus_default_full=comparison(report["records"], "full"),
                  versus_optimized_full=comparison(report["records"], "full_optimized"),
                  versus_argmax=comparison(report["records"], "argmax"),
                  objective_failures=failures, independent_audit=audit,
                  screening_proportions=proportions["summary"] if proportions else None,
                  screened_numerical_checks_passed=screened["full_coverage"]
                    and screened["objective_passed"] == len(report["records"]),
                  oracle_unverified_cases=[r["case_id"] for r in report["records"] if r.get("objective_oom")],
                  raw_changed_cases_vs_default_full=sum(
                      r["methods"]["screened"].get("raw_differing_voxels_vs_full", 0) > 0
                      for r in report["records"] if r["methods"]["screened"]["status"] == "ok"))
    return result


def format_number(value, digits=2, scale=1):
    return "—" if value is None else f"{value * scale:.{digits}f}"


def render(summary):
    lines = ["## Completed measurements", "",
             "Each complete method is aggregated over the entire cohort. Dashes mark an incomplete",
             "default-full control; its available subset is compared separately below. Memory is",
             "additional peak allocated MiB (mean of case peaks / largest case peak).", "",
             "| Dataset | Method | Completed | Dice (%) | IoU (%) | Mean decoder ms | Peak MiB: mean / max |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset, result in summary["datasets"].items():
        for name, m in result["methods"].items():
            metrics = [format_number(m[k], scale=100) for k in ("dice", "iou")] if m["full_coverage"] else ["—", "—"]
            ms = format_number(m["mean_ms"]) if m["full_coverage"] else "—"
            memory = f'{m["mean_peak_mib"]:.1f} / {m["max_peak_mib"]:.1f}' if m["full_coverage"] else f'OOM: {m["oom"]}'
            lines.append(f'| {dataset} | {NAMES[name]} | {m["completed"]}/{m["attempted"]} | '
                         + " | ".join(metrics) + f' | {ms} | {memory} |')
    lines += ["", "### Matched comparisons", "",
              "Default-full rows use only cases completed by both methods and explicitly show the",
              "smaller paired cohort. Optimized full is a distinct auxiliary control, not a",
              "replacement silently relabeled as normal screening-off behavior.", "",
              "| Dataset | Comparator | Paired volumes | Screened speedup | Mean peak reduction | Dice Δ (pp) | IoU Δ (pp) |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset, result in summary["datasets"].items():
        for key, label in (("versus_default_full", "Default full"), ("versus_optimized_full", "Optimized full")):
            c = result[key]
            lines.append(f'| {dataset} | {label} | {c["cases"]} | {format_number(c["speedup"])}× | '
                         f'{format_number(c["mean_peak_reduction_percent"])}% | '
                         f'{format_number(c["dice_delta_pp"], 5)} | {format_number(c["iou_delta_pp"], 5)} |')
    lines += ["", "### Screening proportion versus measured memory reduction", "",
              "Screening proportions exclude pruned whole classes and count class-probability entries.",
              "Sorting avoidance includes workspace fallbacks. Neither is a whole-memory percentage.", "",
              "| Dataset | H+L / active entries | Effective sorting avoidance | Class-pruned / all entries | Peak reduction vs optimized full |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for dataset, result in summary["datasets"].items():
        p = result["screening_proportions"] or {}
        c = result["versus_optimized_full"]
        lines.append(f'| {dataset} | {format_number(p.get("screening_fraction_active"), scale=100)}% | '
                     f'{format_number(p.get("effective_sort_avoidance_active"), scale=100)}% | '
                     f'{format_number(p.get("class_pruned_fraction_all"), scale=100)}% | '
                     f'{format_number(c["mean_peak_reduction_percent"])}% |')
    lines += ["", "### Numerical checks", "",
              "All recorded failures remain failures. Completing measurements does not waive the",
              "four-eps budget. Baseline failures and screened-path checks are listed separately.", "",
              "| Dataset | Method | Passed / failed / unverified completed cases | Maximum regret (float32 eps) |",
              "| --- | --- | ---: | ---: |"]
    for dataset, result in summary["datasets"].items():
        for method in METHODS[1:]:
            m = result["methods"][method]
            lines.append(f'| {dataset} | {NAMES[method]} | {m["objective_passed"]} / '
                         f'{m["objective_failed"]} / {m["objective_unverified"]} | '
                         f'{format_number(m["max_regret_eps"], 6)} |')
    for dataset, result in summary["datasets"].items():
        for failure in result["objective_failures"]:
            lines.append(f'\nRecorded failure: `{dataset}/{failure["case_id"]}/{failure["method"]}`, '
                         f'{failure["max_regret_eps"]:.6f} eps.')
    return "\n".join(lines) + "\n"


PENDING_STATUS = "**Status: preparing/running; no new complete-cohort results are claimed yet.**"


def completed_document(template, summary):
    """Publish a generated report only into the explicitly pending document."""
    if template.count(PENDING_STATUS) != 1 or "## Completed measurements" in template:
        raise ValueError("Expected the pending acceptance document; refusing to replace other work")
    ready = (summary["all_screened_numerical_checks_passed"]
             and summary["all_metrics_timing_audits_passed"]
             and summary["all_screening_diagnostics_complete"])
    status = ("**Status: measurements complete; screened-path acceptance checks passed on these cohorts.**"
              if ready else "**Status: measurements finished; acceptance is NOT fully passed. See the checks below.**")
    notes = [status, "", "## Acceptance outcome", "",
             f'- Screened numerical checks: **{"passed" if summary["all_screened_numerical_checks_passed"] else "NOT passed"}**.',
             f'- Independent metrics/timing audits: **{"passed" if summary["all_metrics_timing_audits_passed"] else "NOT passed"}**.',
             f'- Complete screening diagnostics: **{"yes" if summary["all_screening_diagnostics_complete"] else "NO"}**.',
             "- Argmax is included as a reference, with every foreground confusion count compared against the original published benchmark.",
             "- This is empirical acceptance of the frozen implementation on the stated cohorts, not proof for every possible input.",
             "- Screening and unscreened decoding need not be bitwise identical; binary objective tolerance and ground-truth metrics are reported separately.",
             "- Any baseline OOM or objective-budget failures below remain unresolved baseline limitations, not passed tests.", ""]
    return template.replace(PENDING_STATUS, "\n".join(notes), 1).rstrip() + "\n\n" + render(summary)


def wait_for_completion(path, timeout=0):
    """Wait only on the tiny status file, not per-case benchmark logs or tensors."""
    if timeout < 0:
        raise ValueError("Wait timeout must be nonnegative")
    deadline = time.monotonic() + timeout
    while True:
        try:
            state = json.loads(path.read_text())
        except json.JSONDecodeError:
            state = None  # The launcher may be in the middle of writing status.
        if state is not None and state.get("complete"):
            return state
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Acceptance sequence has not finished; no final report generated")
        time.sleep(min(20, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--document", type=Path,
                        help="Generate the final report into its explicitly pending document")
    parser.add_argument("--wait-timeout", type=int, default=0,
                        help="Optionally wait this many seconds for the serial acceptance to finish")
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    state = wait_for_completion(artifacts / "run_status.json", args.wait_timeout)
    if (set(state["datasets"]) != set(COHORTS)
            or any(state["datasets"][name]["expected_cases"] != count for name, count in COHORTS.items())):
        raise ValueError("Final acceptance must contain all four complete cohorts")
    for name, digest in state["source_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {name}")
    summary = dict(complete=True, datasets={})
    for short, status in state["datasets"].items():
        report_path = artifacts / (short + ".json")
        report = json.loads(report_path.read_text())
        if report.get("force_screening") is not False or report.get("oracle") != "bounded":
            raise ValueError("Final acceptance requires production dispatch and the bounded oracle")
        if len(report["records"]) != status["expected_cases"]:
            raise ValueError("Case count mismatch")
        audit_path = artifacts / (short + ".audit.json")
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"audit": "not_passed"}
        path = artifacts / (short + ".proportions.json")
        proportions = json.loads(path.read_text()) if path.exists() else None
        if proportions is not None:
            if (not proportions["complete"] or proportions["summary"]["cases"] != status["expected_cases"]
                    or proportions["benchmark_sha256"] != hashlib.sha256(report_path.read_bytes()).hexdigest()):
                raise ValueError("Screening diagnostics incomplete or report changed")
        summary["datasets"][short] = summarize(report, audit, proportions)
    summary["all_screened_numerical_checks_passed"] = all(
        r["screened_numerical_checks_passed"] for r in summary["datasets"].values())
    summary["all_metrics_timing_audits_passed"] = all(
        r["independent_audit"].get("metrics_and_timing_audit") == "passed" for r in summary["datasets"].values())
    summary["all_screening_diagnostics_complete"] = all(
        r["screening_proportions"] is not None for r in summary["datasets"].values())
    document = completed_document(args.document.read_text(), summary) if args.document else None
    with (artifacts / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with (artifacts / "RESULTS.md").open("x") as stream:
        stream.write(render(summary))
    if document is not None:
        args.document.write_text(document)
    print(json.dumps({k: v for k, v in summary.items() if k != "datasets"}, indent=2))


if __name__ == "__main__":
    main()
