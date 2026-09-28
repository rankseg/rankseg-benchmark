"""Auto-default acceptance on a predeclared subset, never a full-cohort claim.

Two lexicographic cases per held-out fold, plus the existing liver_43 large
index regression. No checkpoints, inference, resampling or outcome selection.
"""
import argparse
import csv
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from rankseg_benchmark.nnunet.paths import workspace_root
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("Task003_Liver", "Task007_Pancreas", "Task008_HepaticVessel", "Task006_Lung")


def select_with_regression(prepare, config, per_fold):
    cases, audit, available = prepare(config, per_fold)
    if config.dataset_id == "Task003_Liver":
        all_cases, _, _ = prepare(config, 0)
        extra = [case for case in all_cases if case.case_id == "liver_43"]
        if len(extra) != 1:
            raise ValueError("Missing/duplicate liver_43 regression input")
        if not any(case.case_id == "liver_43" for case in cases):
            cases = [*cases, extra[0]]
    return cases, audit, available


def audit_argmax(report, reference_csv):
    with reference_csv.open() as stream:
        reference = {(r["case_id"], int(r["label"])): r for r in csv.DictReader(stream)
                     if r["method"] == "argmax"}
    for record in report["records"]:
        for row in record["methods"]["argmax"]["metrics"]:
            original = reference[record["case_id"], row["label"]]
            if any(row[k] != int(original[k]) for k in ("tp", "fp", "fn")):
                raise AssertionError("Argmax counts differ from the published case evidence")
    return dict(scope="selected subset only", argmax_confusion_counts="identical to published case evidence",
                metrics_and_timing_audit="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankseg-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sys.path[:0] = [str(args.rankseg_path), str(ROOT)]
    import numpy as np
    import torch
    from rankseg import _rankseg_algo as algo
    from rankseg_benchmark.nnunet import screening_benchmark as bench
    from rankseg_benchmark.nnunet.config import load_dataset_config
    from rankseg_benchmark.nnunet.io import load_probabilities, restore_cropped_probabilities
    spec = importlib.util.spec_from_file_location("subset_counts", ROOT / "scripts/nnunet/collect_screening_proportions.py")
    counts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(counts)
    spec = importlib.util.spec_from_file_location("subset_summary", ROOT / "scripts/nnunet/summarize_screening_acceptance.py")
    summarizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summarizer)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU substitution")
    args.output_dir.mkdir(parents=True)
    paths = [*args.rankseg_path.joinpath("rankseg").rglob("*.py"), Path(__file__), Path(bench.__file__),
             ROOT / "scripts/nnunet/collect_screening_proportions.py", ROOT / "scripts/nnunet/summarize_screening_acceptance.py"]
    frozen = {str(p): bench.sha256(p) for p in paths}
    status = dict(complete=False, scope="subset plus liver_43", source_hashes=frozen, datasets={},
                  started_utc=datetime.now(timezone.utc).isoformat())
    def save_status():
        (args.output_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    def check_sources():
        if any(bench.sha256(Path(p)) != digest for p, digest in frozen.items()):
            raise RuntimeError("Frozen source changed")
    save_status()
    prepare = bench.prepare_cases
    for task in TASKS:
        check_sources()
        config = load_dataset_config(ROOT / "configs/nnunet" / f"{task}_ensemble_oof.yaml")
        cases, _, available = select_with_regression(prepare, config, 2)
        output = args.output_dir / f"{task}.json"
        status["datasets"][task] = dict(state="running", cases=len(cases), available=available)
        save_status()
        options = SimpleNamespace(rankseg_path=args.rankseg_path, output=output,
                                  manifest=ROOT / "configs/nnunet" / f"{task}_ensemble_oof.yaml",
                                  cases_per_fold=2, repeats=5, warmup=2, threads=4,
                                  production_dispatch=True, screening_mode="auto", oracle="bounded",
                                  objective_failure_policy="record")
        with (args.output_dir / f"{task}.log").open("x") as log:
            with redirect_stdout(log), redirect_stderr(log), patch.object(
                    bench, "prepare_cases", lambda cfg, n: select_with_regression(prepare, cfg, n)):
                bench.run(options)
        report = json.loads(output.read_text())
        if not report["complete"] or report["selected_case_ids"] != [c.case_id for c in cases]:
            raise AssertionError("Missing or changed selected cases")
        report["selection"] = "first two lexicographic cases per held-out fold; additionally liver_43 for Liver"
        report["selection_driver_sha256"] = bench.sha256(Path(__file__))
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        summarizer.verify_metrics(report, allow_subset=True)
        audit = audit_argmax(report, workspace_root(ROOT) / "outputs" / f"{task}_ensemble_oof" / "case_label_metrics.csv")
        status["datasets"][task]["state"] = "counting"
        save_status()
        # Untimed, separately loaded certificate diagnostics; unchanged data.
        rows = []
        for case, old in zip(cases, report["records"]):
            if case.case_id != old["case_id"] or bench.sha256(case.probabilities) != old["probability_sha256"]:
                raise AssertionError("Probability cache changed")
            p, _ = load_probabilities(case.probabilities, config.probability_key)
            p, restoration = restore_cropped_probabilities(
                p, probability_path=case.probabilities, target_shape=tuple(old["shape"][2:]),
                background_channel=config.channel_labels.index(0))
            if restoration != old["restoration"]:
                raise AssertionError("Restored shape changed")
            tensor = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))[None].cuda()
            value = counts.screening_counts(tensor, config.channel_labels)
            rows.append(dict(case_id=case.case_id, auto_screens=algo._rma_dice_use_screening(tensor, "auto"),
                             **value))
            del tensor, p
            torch.cuda.empty_cache()
        proportions = dict(complete=True, subset=True, timed=False, records=rows,
                           summary=counts.summarize(rows), benchmark_sha256=bench.sha256(output))
        with (args.output_dir / f"{task}.proportions.json").open("x") as stream:
            json.dump(proportions, stream, indent=2, allow_nan=False)
        summary = summarizer.summarize(report, audit, proportions, allow_subset=True)
        with (args.output_dir / f"{task}.summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        check_sources()
        status["datasets"][task].update(state="measurements_complete", objective_check_counts=report["summary"]["objective_check_counts"])
        save_status()
    status.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    save_status()


if __name__ == "__main__":
    main()
