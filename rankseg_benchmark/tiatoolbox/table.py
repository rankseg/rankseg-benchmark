"""Rebuild the TIAToolbox comparison table from case-level scores.

The tracked CSV files are small, reviewable outputs of the inference/evaluation
modules beside this file. ``--from-artifacts`` reads the ignored run outputs
instead, so the published table can be checked after a full rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
EVIDENCE = PACKAGE / "evidence"
ARTIFACTS = ROOT / "artifacts"
FILES = {
    "micronet_consep": ("tiatoolbox_micronet_consep", "per_case.csv"),
    "micronet_kumar": ("tiatoolbox_micronet_kumar", "per_case.csv"),
    "hovernet_consep": ("tiatoolbox_hovernet_consep", "per_case.csv"),
    "hovernet_kumar": ("tiatoolbox_hovernet_kumar", "per_case.csv"),
    "bcss": ("tiatoolbox_bcss", "per_case.csv"),
    "bcss_multilabel": ("tiatoolbox_bcss", "multilabel_per_case.csv"),
}
CLASSES = ("tumor", "stroma", "inflammatory", "necrosis", "other")
EXPECTED_N = {name: (45 if name.startswith("bcss") else 14) for name in FILES}
INSTANCE_FIELDS = (
    "id",
    "argmax_aji",
    "foreground_aji",
    "multiclass_aji",
    "argmax_pq",
    "foreground_pq",
    "multiclass_pq",
)


def read_cases(path: Path, expected_n: int) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames and "id" in reader.fieldnames, path
        rows = list(reader)
    ids = [row["id"] for row in rows]
    assert len(rows) == expected_n and len(set(ids)) == expected_n, path
    return rows


def load_cases(*, from_artifacts: bool = False) -> dict[str, list[dict[str, str]]]:
    cases = {}
    for name, (folder, filename) in FILES.items():
        path = ARTIFACTS / folder / filename if from_artifacts else EVIDENCE / f"{name}.csv"
        cases[name] = read_cases(path, EXPECTED_N[name])
    for left, right in (
        ("micronet_consep", "hovernet_consep"),
        ("micronet_kumar", "hovernet_kumar"),
        ("bcss", "bcss_multilabel"),
    ):
        assert {row["id"] for row in cases[left]} == {row["id"] for row in cases[right]}
    cases["micronet_kumar_instances"] = (
        instance_cases_from_artifacts() if from_artifacts else read_cases(EVIDENCE / "micronet_kumar_instances.csv", 14)
    )
    assert {row["id"] for row in cases["micronet_kumar_instances"]} == {row["id"] for row in cases["micronet_kumar"]}
    return cases


def instance_cases_from_artifacts() -> list[dict[str, str]]:
    source = ARTIFACTS / "tiatoolbox_micronet_kumar" / "results.json"
    recorded = json.loads(source.read_text())
    rows = []
    for case in recorded["cases"]:
        metrics = case["instance_metrics"]
        row = {"id": case["id"]}
        for metric in ("aji", "pq"):
            for method in ("argmax", "foreground", "multiclass"):
                row[f"{method}_{metric}"] = str(metrics[method][metric])
        rows.append(row)
    assert len(rows) == len({row["id"] for row in rows}) == 14
    return rows


def values(rows: list[dict[str, str]], column: str) -> np.ndarray:
    array = np.array([float(row[column]) for row in rows], dtype=np.float64)
    assert np.isfinite(array).all() and ((0 <= array) & (array <= 1)).all(), column
    return array


def paired(rows: list[dict[str, str]], baseline: str, candidate: str) -> dict[str, float | int]:
    first, second = values(rows, baseline), values(rows, candidate)
    delta = second - first
    return {
        "argmax": float(first.mean()),
        "rankseg": float(second.mean()),
        "delta": float(delta.mean()),
        "wins": int(np.count_nonzero(delta > 1e-12)),
    }


def bootstrap_interval(
    rows: list[dict[str, str]], calls: tuple[tuple[str, str], ...], *, seed: int, selected: int
) -> tuple[float, float]:
    """Replay the original evaluator's ordered RNG calls (10,000 pairs)."""
    rng = np.random.default_rng(seed)
    interval = None
    for index, (baseline, candidate) in enumerate(calls):
        delta = values(rows, candidate) - values(rows, baseline)
        boot = rng.choice(delta, size=(10_000, len(delta)), replace=True).mean(axis=1)
        if index == selected:
            interval = tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))
    assert interval is not None
    return interval


MICRONET_CONSEP_CALLS = (
    ("raw_argmax", "raw_argmax"),
    ("raw_argmax", "raw_foreground"),
    ("raw_argmax", "raw_multiclass"),
    ("post_argmax", "post_argmax"),
    ("post_argmax", "post_foreground"),
    ("post_argmax", "post_multiclass"),
)
MICRONET_KUMAR_CALLS = (
    ("raw_argmax", "raw_foreground"),
    ("raw_argmax", "raw_multiclass"),
    ("post_argmax", "post_foreground"),
    ("post_argmax", "post_multiclass"),
)
HOVERNET_CALLS = (("raw_argmax", "raw_rankseg"), ("post_argmax", "post_rankseg"))


def fmt_change(value: float) -> str:
    return f"{value * 100:+.3f}"


def format_row(
    model: str,
    target: str,
    rows: list[dict[str, str]],
    baseline: str,
    candidate: str,
    interval: tuple[float, float] | None,
) -> str:
    result = paired(rows, baseline, candidate)
    ci = "—" if interval is None else f"{fmt_change(interval[0])} to {fmt_change(interval[1])}"
    return (
        f"| {model} | {target} | {len(rows)} | {result['argmax']:.4f} | "
        f"{result['rankseg']:.4f} | {fmt_change(result['delta'])} | {ci} | "
        f"{result['wins']}/{len(rows)} |"
    )


def validate_bcss(rows: list[dict[str, str]]) -> None:
    for row in rows:
        for prefix, macro in (("argmax", "argmax_macro"), ("rankseg", "rankseg_macro")):
            class_scores = np.array([float(row[f"{prefix}_{klass}"]) for klass in CLASSES])
            assert abs(class_scores.mean() - float(row[macro])) < 1e-12, row["id"]


def render(cases: dict[str, list[dict[str, str]]]) -> str:
    for rows in cases.values():
        assert rows
    validate_bcss(cases["bcss"])
    c = cases
    primary = [
        format_row(
            "`micronet-consep` / CoNSeP Test",
            "Nucleus Dice",
            c["micronet_consep"],
            "post_argmax",
            "post_foreground",
            bootstrap_interval(c["micronet_consep"], MICRONET_CONSEP_CALLS, seed=20261018, selected=4),
        ),
        format_row(
            "`micronet-consep` / Kumar test",
            "Nucleus Dice",
            c["micronet_kumar"],
            "post_argmax",
            "post_foreground",
            bootstrap_interval(c["micronet_kumar"], MICRONET_KUMAR_CALLS, seed=20261018, selected=2),
        ),
        format_row(
            "`hovernet_original-consep` / CoNSeP Test",
            "Nucleus Dice",
            c["hovernet_consep"],
            "post_argmax",
            "post_rankseg",
            bootstrap_interval(c["hovernet_consep"], HOVERNET_CALLS, seed=20261018, selected=1),
        ),
        format_row(
            "`hovernet_original-kumar` / Kumar test",
            "Nucleus Dice",
            c["hovernet_kumar"],
            "post_argmax",
            "post_rankseg",
            bootstrap_interval(c["hovernet_kumar"], HOVERNET_CALLS, seed=20261018, selected=1),
        ),
        format_row(
            "`fcn_resnet50_unet-bcss` / BCSS test",
            "Five-class ROI macro Dice",
            c["bcss"],
            "argmax_macro",
            "rankseg_macro",
            bootstrap_interval(c["bcss"], (("argmax_macro", "rankseg_macro"),), seed=20261017, selected=0),
        ),
    ]
    primary.extend(
        format_row("↳ BCSS", klass.title(), c["bcss"], f"argmax_{klass}", f"rankseg_{klass}", None) for klass in CLASSES
    )
    instance_rows = c["micronet_kumar_instances"]
    assert {row["id"] for row in instance_rows} == {row["id"] for row in c["micronet_kumar"]}
    secondary = [
        format_row(
            "`micronet-consep` / Kumar test", "AJI (exploratory)", instance_rows, "argmax_aji", "foreground_aji", None
        ),
        format_row(
            "`micronet-consep` / Kumar test", "PQ (exploratory)", instance_rows, "argmax_pq", "foreground_pq", None
        ),
        format_row(
            "`fcn_resnet50_unet-bcss` / BCSS test",
            "Independent-class multilabel Dice",
            c["bcss_multilabel"],
            "argmax_macro",
            "multilabel_macro",
            bootstrap_interval(
                c["bcss_multilabel"], (("argmax_macro", "multilabel_macro"),), seed=20261017, selected=0
            ),
        ),
    ]
    return "\n".join(
        [
            "# TIAToolbox paired-decoder results",
            "",
            "TIAToolbox 1.6.0; RankSEG 0.0.8 decoder implementation, Dice/RMA, `safe_screening=True`, default "
            "`pruning_prob=0.5`. Each pair uses the same saved model probabilities. No retraining, "
            "probability scaling, or threshold tuning. Scores are calculated per image or ROI, then averaged.",
            "",
            "## Primary single-mask comparisons",
            "",
            "| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | "
            "Paired 95% bootstrap interval (pp) | Improved |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *primary,
            "",
            "For nucleus models, Dice measures binary `inst_map > 0` after identical official instance "
            'postprocessing. MicroNet uses the nucleus foreground channel with `output_mode="multilabel"` '
            "to produce one binary mask; HoVer-Net uses the nucleus probability with the same output mode "
            'while keeping its HV/type maps unchanged. BCSS uses `output_mode="multiclass"` for one '
            "five-class mask. BCSS raw labels 0 and 7 are ignored for both methods; both-empty class "
            "Dice is 1 and an absent ground-truth class with predicted pixels scores 0. The BCSS "
            "per-class rows each average over all 45 ROIs, including class-absent ROIs. Bootstrap "
            "intervals resample paired images/ROIs 10,000 times.",
            "",
            "## Secondary diagnostics",
            "",
            "| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | "
            "Paired 95% bootstrap interval (pp) | Improved |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *secondary,
            "",
            "AJI/PQ use the same TIAToolbox MicroNet instance maps, scored by a local implementation of "
            "the HoVer-Net matching definitions; they are secondary checks. BCSS multilabel masks "
            "may overlap or leave pixels unassigned, so that row is not a valid single-label BCSS output.",
            "",
            "## Interpretation limits",
            "",
            "- The MicroNet CoNSeP argmax baseline (0.7376) does not reproduce the approximately "
            "0.80 Dice quoted in [TIAToolbox's MicroNet documentation]"
            "(https://tia-toolbox.readthedocs.io/en/latest/_autosummary/"
            "tiatoolbox.models.architecture.micronet.MicroNet.html). The cause is unresolved. CoNSeP was "
            "obtained from a Kaggle mirror whose byte identity with the authors' archive was not verified.",
            "- The [TIAToolbox 1.6.0 BCSS table]"
            "(https://github.com/TissueImageAnalytics/tiatoolbox/blob/v1.6.0/"
            "tiatoolbox/models/engine/semantic_segmentor.py) reports 0.763 overall; its exact aggregation is "
            "not reproduced here. Our ROI-wise argmax macro is 0.5928, while our pooled-pixel "
            "class macro is 0.7444. These are distinct metrics. The BCSS multiclass paired "
            "result shows no overall gain under our ROI-wise rule.",
            "- The same CoNSeP images are used for both CoNSeP checkpoints, and the same Kumar "
            "images for both Kumar rows. MicroNet/Kumar is a cross-dataset evaluation of the "
            "CoNSeP checkpoint. The 14-image nucleus cohorts are small.",
            "",
            "Regenerate this file with `python -m rankseg_benchmark.tiatoolbox.table --output "
            "rankseg_benchmark/tiatoolbox/RESULTS.md`. Use `--from-artifacts --check-artifacts` "
            "after full inference/evaluation to verify the tracked per-case evidence against the ignored run outputs.",
            "",
        ]
    )


def check_artifacts() -> None:
    for name, (folder, filename) in FILES.items():
        source = ARTIFACTS / folder / filename
        target = EVIDENCE / f"{name}.csv"
        assert source.read_bytes() == target.read_bytes(), f"Tracked evidence differs from {source}"
    recorded = json.loads((ARTIFACTS / "tiatoolbox_micronet_kumar" / "results.json").read_text())
    rows = {row["id"]: row for row in read_cases(EVIDENCE / "micronet_kumar_instances.csv", 14)}
    for case in recorded["cases"]:
        for metric in ("aji", "pq"):
            for method in ("argmax", "foreground", "multiclass"):
                assert (
                    abs(float(rows[case["id"]][f"{method}_{metric}"]) - case["instance_metrics"][method][metric])
                    < 1e-12
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-artifacts", action="store_true", help="Read ignored run outputs instead of tracked CSVs"
    )
    parser.add_argument("--check-artifacts", action="store_true", help="Check tracked CSVs against local run outputs")
    parser.add_argument(
        "--refresh-evidence", action="store_true", help="Copy validated case CSVs from ignored run outputs"
    )
    parser.add_argument("--output", type=Path, help="Write Markdown instead of printing it")
    args = parser.parse_args()
    if args.refresh_evidence:
        artifact_cases = load_cases(from_artifacts=True)
        for name, (folder, filename) in FILES.items():
            shutil.copyfile(ARTIFACTS / folder / filename, EVIDENCE / f"{name}.csv")
        with (EVIDENCE / "micronet_kumar_instances.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=INSTANCE_FIELDS)
            writer.writeheader()
            writer.writerows(artifact_cases["micronet_kumar_instances"])
    if args.check_artifacts:
        check_artifacts()
    markdown = render(load_cases(from_artifacts=args.from_artifacts)).rstrip("\n")
    if args.output:
        args.output.write_text(markdown + "\n")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
