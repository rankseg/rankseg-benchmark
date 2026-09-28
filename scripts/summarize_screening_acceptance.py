"""Read-only numerical/metric/provenance audit, then generate a new Markdown report."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mini_acceptance", ROOT / "scripts/benchmark_screening_acceptance.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def audit(report):
    if not report["complete"]:
        raise ValueError("Dataset not complete")
    sources = report["sources"]
    expected = {(s["fold"], i) for s in sources for i in range(s["rows"])}
    observed = [(r["fold"], r["source_row"]) for r in report["records"]]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Missing or duplicate cache rows")
    for r in report["records"]:
        target_counts = None
        for method in bench.METHODS:
            row = r["methods"][method]
            times = row["samples_ms"]
            if (len(times) != report["repeats"] or any(not math.isfinite(t) or t <= 0 for t in times)
                    or statistics.median(times) != row["median_ms"]):
                raise ValueError("Invalid timing")
            if not math.isfinite(row["peak_mib"]) or row["peak_mib"] < 0:
                raise ValueError("Invalid memory measurement")
            counts = row["counts"]
            classes = r["shape"][1]
            if any(len(counts[k]) != classes or any(type(v) is not int or v < 0 for v in counts[k])
                   for k in ("tp", "fp", "fn")):
                raise ValueError("Invalid confusion counts")
            totals = [tp + fn for tp, fn in zip(counts["tp"], counts["fn"])]
            if target_counts is not None and totals != target_counts:
                raise ValueError("Methods have different ground-truth counts")
            target_counts = totals
            if sum(counts["fp"]) != sum(counts["fn"]):
                raise ValueError("Multiclass confusion counts inconsistent")
            if method != "argmax":
                obj = row["objective"]
                loss = obj["max_regret"]
                if (not math.isfinite(loss) or loss < 0 or obj["max_regret_eps"] != loss / 2**-23
                        or obj["status"] != ("passed" if loss <= 4 * 2**-23 else "failed")):
                    raise ValueError("Invalid objective accounting")
        for row in r["screening_counts"]["rows"]:
            if (row["forced_positive"] + row["forced_negative"] + row["undecided"] != row["active_entries"]
                    or row["active_entries"] + row["class_pruned_entries"] != row["total_entries"]):
                raise ValueError("Invalid screening partition")
    for method in bench.METHODS:
        result = report["summary"]["methods"][method]
        independent = bench.independent_metrics(report["records"], method, report["dataset"] == "kits")
        for k, name in (("dice", "mDice"), ("iou", "mIoU")):
            if result[k] != independent[k] or not math.isclose(
                    result[k], report["repository_metrics"][method][name], abs_tol=1e-12, rel_tol=0):
                raise ValueError("Independent metric aggregation failed")
        rows = [r["methods"][method] for r in report["records"]]
        if (result["mean_ms"] != statistics.mean(r["median_ms"] for r in rows)
                or result["mean_peak_mib"] != statistics.mean(r["peak_mib"] for r in rows)
                or result["max_peak_mib"] != max(r["peak_mib"] for r in rows)):
            raise ValueError("Timing/memory aggregation failed")
        if method != "argmax":
            objectives = [r["objective"] for r in rows]
            if (result["objective_passed"] != sum(o["status"] == "passed" for o in objectives)
                    or result["objective_failed"] != sum(o["status"] == "failed" for o in objectives)
                    or result["max_regret_eps"] != max(o["max_regret_eps"] for o in objectives)):
                raise ValueError("Objective aggregation failed")
    for key, method in (("screened_checks_passed", "screened"),
                        ("forced_screening_checks_passed", "screened_forced")):
        if report["summary"][key] != all(r["methods"][method]["objective"]["status"] == "passed"
                                        for r in report["records"]):
            raise ValueError("Incorrect numerical acceptance status")


def render(summary):
    passed = summary["all_screened_checks_passed"] and summary["all_forced_screening_checks_passed"]
    lines = ["# RankSEG mini-benchmark screening acceptance — 2026-09-22", "",
             f'**Status: measurements complete; numerical checks {"passed" if passed else "NOT all passed"}.**', "",
             "## Scope and protocol", "",
             "- VOC, Cityscapes and ADE20K: the first 100 stored images each, selected without looking at outcomes; **subsets**, not their full-dataset README results.",
             "- KiTS: **all 5,712 stored slices in all five folds**. RankSEG receives only the foreground channel in multilabel mode. Argmax retains both original channels.",
             "- All caches use Hugging Face revision `1884f0766268cdc62730e696f24dcc913d551b35`. The missing natural-image prefixes were restored from this same revision; KiTS reused existing local files.",
             "- MONAI Pancreas/Spleen are **not included**: their local probability manifests were unavailable. No substitute nnU-Net caches, checkpoint downloads, or new inference were used.",
             "- Same frozen RankSEG snapshot as the 778-volume nnU-Net acceptance. No core algorithm edits, training, checkpoint changes, or parameter fitting.",
             "- RTX 3090, Torch 2.11.0+cu130, Triton 3.6.0, float32, batch size 1, four CPU threads. Two warm-ups and seven synchronized repeats in rotating method order.",
             "- Timings include benchmark channel routing and output conversion; exclude loading/transfers, compilation, metrics, oracles and screening diagnostics. Means of per-input median decoder wall times, not inference speedups.",
             "- Memory is extra peak PyTorch allocated GPU memory above resident original probabilities, including routing temporaries and output, excluding reserved memory and CUDA context. Tables show means of sample peaks.",
             "- Natural-image scores: mean over GT-present classes per image, then mean over images. KiTS: foreground slice metrics, then mean within case, across cases within fold, and across folds, preserving the repository's empty-slice convention. **Not whole-volume Dice.**",
             "- Natural-image ground-truth values outside [0, C) are excluded, in addition to ignore_index=255, matching the existing accumulator (e.g. VOC caches contain void label 21). KiTS retains its separate existing label-clamping convention. Labels never affect decoding or screening.",
             "- Dice/RMA, smooth=0, pruning_prob=0.5, max_score. IoU is measured but not separately optimized.", "",
             "## Methods", "",
             "`full` is normal public `safe_screening=False`; `screened` is normal public `safe_screening=True`.",
             "`full_optimized` is a separate direct-argmax full-sort auxiliary control; `screened_forced` overrides only the small-input bypass.",
             "The ordinary on/off paths do not pay for dispatch monkeypatching. Auxiliary controls use scoped overrides restored after each call.", "",
             "## Quality, decoder runtime and memory", "",
             "| Dataset | Method | Dice (%) | IoU (%) | Mean ms/input | Mean peak MiB |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, r in summary["datasets"].items():
        for method, v in r["methods"].items():
            lines.append(f'| {name} | {method} | {100*v["dice"]:.4f} | {100*v["iou"]:.4f} | '
                         f'{v["mean_ms"]:.4f} | {v["mean_peak_mib"]:.4f} |')
    lines += ["", "## Production on/off comparison", "",
              "All paired samples are retained. Speedup is full / screened; negative memory reduction is a regression.", "",
              "| Dataset | Samples | Speedup | Mean peak reduction | Dice Δ (pp) | IoU Δ (pp) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in summary["datasets"].items():
        a, b = (r["methods"][m] for m in ("full", "screened"))
        lines.append(f'| {name} | {r["samples"]} | {a["mean_ms"]/b["mean_ms"]:.3f}× | '
                     f'{100*(1-b["mean_peak_mib"]/a["mean_peak_mib"]):.3f}% | '
                     f'{100*(b["dice"]-a["dice"]):.6f} | {100*(b["iou"]-a["iou"]):.6f} |')
    lines += ["", "## Screening proportion versus actual dispatch", "",
              "Potential H+L counts are measured in a separate untimed forced-certificate call. They exclude class-pruned rows and count class-probability entries, not unique pixels. Production sorting avoidance respects both bypass and workspace fallback.", "",
              "| Dataset | Potential H+L / active | Production sort avoidance / active | Small-input bypass samples | Observed screening calls | Workspace fallback rows (forced) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in summary["datasets"].items():
        c = r["forced_screening_certificate"]
        lines.append(f'| {name} | {100*c["screening_fraction_active"]:.4f}% | '
                     f'{100*r["production_effective_sort_avoidance_active"]:.4f}% | '
                     f'{r["production_small_input_bypass_samples"]}/{r["samples"]} | '
                     f'{r["production_screening_calls"]} | {c["fallback_rows"]} |')
    lines += ["", "## Correctness and observed differences", "",
              "Every sample/method is checked against the same exhaustive full-prefix float64 binary-objective oracle, with the unchanged four-float32-eps budget. No epsilon-based full-sort retries are added. This is a binary-objective regression check, not a guarantee of identical final multiclass labels or ground-truth scores.", "",
              "| Dataset | Method | Passed / failed | Maximum regret (float32 eps) |",
              "| --- | --- | ---: | ---: |"]
    for name, r in summary["datasets"].items():
        for method, v in r["methods"].items():
            if method != "argmax":
                lines.append(f'| {name} | {method} | {v["objective_passed"]} / {v["objective_failed"]} | {v["max_regret_eps"]:.6f} |')
    for name, r in summary["datasets"].items():
        for comparator, change in r["differences_vs_screened"].items():
            worst = change["worst_unit_dice_delta_pp"]
            lines.append(f'\n{name}, screened vs {comparator}: {change["changed_samples"]}/{r["samples"]} changed masks; '
                         f'{change["different_pixels"]} differing pixels; worst scored image/slice Dice Δ '
                         f'{worst[0]:.6f} pp (fold={worst[1]}, source row={worst[2]}, case={worst[3]}).')
    lines += ["", "## Validation and artifacts", "",
              "- Full benchmark repository CPU/CUDA regression tests, including end-to-end report generation and out-of-range/void-label handling, pass. Static checks for the new harness/tests pass; run logs retain exact test counts.",
              "- Confusion counts and metric aggregation are independently computed using NumPy and compared with the unchanged repository accumulators. Per-sample timing medians, memory aggregates, numerical status, cohort completeness and source/cache hashes are audited again after the run.",
              "- Shared input tensors are checked for exact non-mutation on every sample. Untimed hooks verify actual production dispatch, and are removed before timing.",
              "- The real-cache runs complete normally or stop visibly on structural errors/OOM; no CPU retry, resized input, or silent dropped sample is allowed. Numerical budget failures remain recorded failures.",
              "- This is empirical acceptance on these inputs, not a proof for all possible inputs. Natural-image subsets cannot replace the published full-dataset evaluation.", "",
              "Raw reports, per-sample JSONL counts/timings and hashes are in `artifacts/screening-final-acceptance-2026-09-22-v2/` (local, ignored by Git). The initial attempt stopped at VOC's first sample because the new independent audit rejected void label 21. Only the audit was corrected; the complete v2 run starts afresh, and no measurements from the failed attempt are reused.",
              "Reproduce using fresh output paths:", "", "```bash",
              "../env/bin/python -B scripts/benchmark_screening_acceptance.py \\",
              "  --rankseg-path ../rankseg-nnunet-benchmark/outputs/screening-final-acceptance-2026-09-22/snapshot \\",
              "  --output-dir artifacts/screening-acceptance-new --warmup 2 --repeats 7",
              "CUDA_VISIBLE_DEVICES='' ../env/bin/python -B scripts/summarize_screening_acceptance.py \\",
              "  artifacts/screening-acceptance-new --document docs/SCREENING_ACCEPTANCE_NEW.md",
              "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.artifacts / "summary.json").read_text())
    if not summary["complete"] or set(summary["datasets"]) != {"pascal_voc", "cityscapes", "ade20k", "kits"}:
        raise ValueError("All four datasets must be completed")
    if any(v["samples"] != (5712 if name == "kits" else 100) for name, v in summary["datasets"].items()):
        raise ValueError("Incorrect fixed sample counts")
    if args.document.exists():
        raise FileExistsError("Refusing to overwrite existing document")
    for path, digest in summary["source_hashes"].items():
        if bench.sha256(path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    for name, value in summary["datasets"].items():
        report = json.loads((args.artifacts / f"{name}.json").read_text())
        if report["summary"] != value:
            raise ValueError("Dataset summary differs from combined summary")
        audit(report)
        for source in report["sources"]:
            if bench.sha256(source["path"]) != source["sha256"]:
                raise ValueError("Probability cache changed")
    for key, dataset_key in (("all_screened_checks_passed", "screened_checks_passed"),
                             ("all_forced_screening_checks_passed", "forced_screening_checks_passed")):
        if summary[key] != all(r[dataset_key] for r in summary["datasets"].values()):
            raise ValueError("Incorrect overall acceptance status")
    args.document.parent.mkdir(parents=True, exist_ok=True)
    with args.document.open("x") as stream:
        stream.write(render(summary))
    print(json.dumps({"audit": "passed", "screened_numerical_checks": summary["all_screened_checks_passed"],
                      "forced_screening_numerical_checks": summary["all_forced_screening_checks_passed"],
                      "samples": sum(r["samples"] for r in summary["datasets"].values()),
                      "document": str(args.document)}, indent=2))


if __name__ == "__main__":
    main()
