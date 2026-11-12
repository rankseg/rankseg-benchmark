"""Rebuild the complete TIAToolbox 2.1.3 table using only NumPy and CSV evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent
ARTIFACTS = PACKAGE.parents[1] / "artifacts" / "tiatoolbox_v2"
EVIDENCE = PACKAGE / "evidence"
COHORTS = {
    "micronet_consep": ("MicroNet / CoNSeP", 14, ("argmax", "foreground", "multiclass")),
    "micronet_kumar": ("MicroNet / Kumar", 14, ("argmax", "rankseg", "multiclass")),
    "hovernet_consep": ("HoVer-Net CoNSeP / CoNSeP", 14, ("argmax", "rankseg")),
    "hovernet_kumar": ("HoVer-Net Kumar / Kumar", 14, ("argmax", "rankseg")),
    "bcss": ("FCN ResNet50 UNet / BCSS", 45, ("argmax", "rankseg", "multilabel")),
}
CLASSES = ("tumor", "stroma", "inflammatory", "necrosis", "other")


def read(path: Path, expected: int) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise ValueError(f"Incomplete/duplicate evidence: {path}")
    for row in rows:
        for key in row:
            if key not in ("id", "split"):
                value = float(row[key])
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite score: {path}, {key}")
    return rows


def mean(rows: list[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def statistics(rows: list[dict], methods: tuple[str, ...]) -> dict:
    rng = np.random.default_rng(20261026)
    metrics = [k[len("argmax_") :] for k in rows[0] if k.startswith("argmax_")]
    result = {}
    for metric in metrics:
        baseline = np.array([float(r[f"argmax_{metric}"]) for r in rows])
        result[metric] = {}
        for method in methods:
            values = np.array([float(r[f"{method}_{metric}"]) for r in rows])
            if np.any((values < 0) | (values > 1)):
                raise ValueError(f"Score outside [0,1]: {metric}")
            delta = values - baseline
            boot = rng.choice(delta, size=(10000, len(rows)), replace=True).mean(1)
            result[metric][method] = {
                "mean": float(values.mean()),
                "gain": float(delta.mean()),
                "paired_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "wins": int((delta > 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
            }
    return result


def row(label: str, metric: str, summary: dict, method: str, n: int) -> str:
    base, rank = summary["argmax"], summary[method]
    lo, hi = np.array(rank["paired_ci95"]) * 100
    return (
        f"| {label} | {metric} | {n} | {base['mean']:.6f} | {rank['mean']:.6f} "
        f"| {rank['gain'] * 100:+.3f} | {lo:+.3f} to {hi:+.3f} | {rank['wins']}/{n} |"
    )


HEADER = [
    "| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | Paired 95% CI (pp) | Improved |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]


def render(*, from_artifacts: bool = False) -> str:
    cases, summaries = {}, {}
    for name, (_, n, methods) in COHORTS.items():
        file = ARTIFACTS / name / "per_case.csv" if from_artifacts else EVIDENCE / f"{name}_v2.csv"
        cases[name] = read(file, n)
        historical = read(EVIDENCE / f"{name}.csv", n)
        if {r["id"] for r in cases[name]} != {r["id"] for r in historical}:
            raise ValueError(f"Cohort differs from historical evidence: {name}")
        summaries[name] = statistics(cases[name], methods)
    for r in cases["bcss"]:
        for m in COHORTS["bcss"][2]:
            if abs(np.mean([float(r[f"{m}_dice_{k}"]) for k in CLASSES]) - float(r[f"{m}_dice"])) > 1e-12:
                raise ValueError("BCSS macro does not match its five classes")
    text = [
        "# TIAToolbox 2.1.3: all five paired comparisons",
        "",
        "RankSEG 0.0.8, Dice/RMA, `safe_screening=True`, `pruning_prob=0.5`,",
        "`smooth=0`. Each pair uses identical saved model probabilities and",
        "unchanged official postprocessing. No retraining, scaling, calibration",
        "or threshold tuning. Scores are computed per image/ROI, then averaged.",
        "",
        "## Primary comparisons",
        "",
        *HEADER,
    ]
    for name, (label, n, methods) in COHORTS.items():
        endpoint = "Five-class ROI macro Dice" if name == "bcss" else "Nucleus Dice"
        text.append(row(label, endpoint, summaries[name]["dice"], methods[1], n))
    text += [
        "",
        "The two CoNSeP rows share 14 images; the two Kumar rows share another",
        "14 images. MicroNet uses the CoNSeP checkpoint for both cohorts, so",
        "its Kumar row is a cross-dataset evaluation. Nucleus Dice is measured",
        "on `inst_map > 0` after official instance postprocessing. HoVer-Net's",
        "baseline thresholds its nucleus probability at 0.5; the column is",
        "labelled Argmax for consistency with the other comparisons.",
        "",
        "## Comparison with the historical 1.6.0 runs",
        "",
        "| Checkpoint / cohort | 1.6 Argmax | 2.1.3 Argmax | 1.6 RankSEG gain (pp) | 2.1.3 RankSEG gain (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, (label, n, methods) in COHORTS.items():
        old = read(EVIDENCE / f"{name}.csv", n)
        a, b = (
            ("argmax_macro", "rankseg_macro")
            if name == "bcss"
            else ("post_argmax", "post_foreground" if name.startswith("micronet") else "post_rankseg")
        )
        s = summaries[name]["dice"]
        text.append(
            f"| {label} | {mean(old, a):.6f} | {s['argmax']['mean']:.6f} "
            f"| {(mean(old, b) - mean(old, a)) * 100:+.3f} | {s[methods[1]]['gain'] * 100:+.3f} |"
        )
    text += [
        "",
        "These columns describe different inference pipelines. The old engine",
        "uses black edge padding and the new engine uses white. BCSS additionally",
        "changes its registered stride from 256 to 450 pixels. MicroNet also",
        "changes its returned instance map's hole filling. See the",
        "[migration audit](V2_MIGRATION.md) and the detailed",
        "[MicroNet/CoNSeP report](V2_MICRONET_CONSEP.md).",
        "",
        "## Instance metrics",
        "",
        *HEADER,
    ]
    for name, (label, n, methods) in COHORTS.items():
        if name != "bcss":
            for metric in ("aji", "pq"):
                text.append(row(label, metric.upper(), summaries[name][metric], methods[1], n))
    text += [
        "",
        "AJI and PQ evaluate instances; a foreground Dice increase need not",
        "improve either metric. Both methods retain the same official instance",
        "postprocessor. RankSEG does not optimize AJI or PQ directly.",
        "",
        "## BCSS: all five classes",
        "",
        *HEADER,
    ]
    for k in CLASSES:
        text.append(row("BCSS", k.title(), summaries["bcss"][f"dice_{k}"], "rankseg", 45))
    text += [
        "",
        "All 45 ROIs contribute to every class mean. Raw labels 0 and 7 are",
        "excluded only during scoring. Other is included. A both-empty class",
        "scores 1; a predicted class with no ground-truth pixels scores 0.",
        "The necrosis gain is largely driven by three absent-class Dice switches",
        "from 0 to 1; the [presence audit](V2_MIGRATION.md#bcss-class-presence-audit)",
        "separates this effect from scores on ROIs with annotated necrosis.",
        "",
        "## Secondary decoder configurations",
        "",
        *HEADER,
    ]
    for name in ("micronet_consep", "micronet_kumar"):
        label, n, _ = COHORTS[name]
        text.append(row(label, "Two-channel multiclass Dice", summaries[name]["dice"], "multiclass", n))
    text.append(row("BCSS", "Independent-class multilabel macro Dice", summaries["bcss"]["dice"], "multilabel", 45))
    for k in CLASSES:
        text.append(row("BCSS multilabel", k.title(), summaries["bcss"][f"dice_{k}"], "multilabel", 45))
    text += [
        "",
        "BCSS multilabel masks can overlap or leave pixels unassigned. They are",
        "a diagnostic and do not form the single-label output required for",
        "ordinary BCSS semantic segmentation.",
        "",
        "## Reproduce this table",
        "",
        "```bash",
        "python -m rankseg_benchmark.tiatoolbox.table_v2 --output rankseg_benchmark/tiatoolbox/RESULTS_V2.md",
        "```",
        "",
        "Only NumPy and the tracked evidence CSVs are needed (including the",
        "historical CSVs for the old/new comparison).",
        "Paired bootstrap intervals use 10,000 image/ROI resamples and seed",
        "20261026, preserving each evaluator's metric and method call order.",
        "Use `--from-artifacts --check-artifacts` to verify the original run.",
        "Full inference instructions and limitations are in [V2_MIGRATION.md](V2_MIGRATION.md).",
        "",
    ]
    return "\n".join(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-artifacts", action="store_true")
    parser.add_argument("--check-artifacts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check_artifacts:
        for name in COHORTS:
            if (ARTIFACTS / name / "per_case.csv").read_bytes() != (EVIDENCE / f"{name}_v2.csv").read_bytes():
                raise ValueError(f"Evidence differs from original artifacts: {name}")
    report = render(from_artifacts=args.from_artifacts)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
