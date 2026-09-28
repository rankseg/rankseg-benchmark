"""Paired implementation regression on existing, held-out probability caches.

This is an engineering SUBSET, not a rerun/replacement of complete cohorts.
No model inference or checkpoint changes; I/O, metrics and independent oracles
are excluded from decoder timing. Execute serially on an otherwise idle GPU.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from rankseg_benchmark.nnunet.paths import workspace_root
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT.parent / "rankseg"), str(ROOT), str(ROOT)]

import torch

from scripts.benchmark_screening_pipeline import current, load_baseline, measure, source_hashes


def nnunet_cases(args, baseline):
    import numpy as np
    from rankseg_benchmark.nnunet.config import load_dataset_config
    from rankseg_benchmark.nnunet.io import load_probabilities, load_segmentation_with_voxel_volume, restore_cropped_probabilities
    from rankseg_benchmark.nnunet.metrics import overlap_metrics
    from rankseg_benchmark.nnunet.postprocessing import OfficialNnUNetPostprocessor
    from rankseg_benchmark.nnunet.screening_benchmark import prepare_cases, sha256

    for task in ("Task007_Pancreas", "Task008_HepaticVessel", "Task006_Lung", "Task003_Liver"):
        config = load_dataset_config(ROOT / "configs/nnunet" / f"{task}_ensemble_oof.yaml")
        cases, audit, available = prepare_cases(config, args.cases_per_fold)
        if task == "Task003_Liver":
            all_cases, _, _ = prepare_cases(config, 0)
            cases += [c for c in all_cases if c.case_id == "liver_43" and c not in cases]
        postprocessor = OfficialNnUNetPostprocessor(config.postprocessing_file) if config.postprocessing_file else None
        for case in cases:
            p, _ = load_probabilities(case.probabilities, config.probability_key)
            target, voxel_volume = load_segmentation_with_voxel_volume(case.label, reader=config.segmentation_reader)
            p, restoration = restore_cropped_probabilities(p, probability_path=case.probabilities,
                                                           target_shape=target.shape, background_channel=0)
            probs = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32)).unsqueeze(0).cuda()
            record, predictions = measure(probs, baseline, "multiclass", repeats=args.repeats)
            record.update(dataset=task, case_id=case.case_id, n_samples=1, probability_sha256=sha256(case.probabilities),
                          label_sha256=sha256(case.label), restoration=restoration, available_cases=available)
            for name, prediction in predictions.items():
                pred = np.asarray(config.channel_labels)[prediction[0].numpy()]
                if postprocessor is not None:
                    pred = postprocessor(pred, volume_per_voxel=voxel_volume)
                record["methods"][name]["metrics"] = overlap_metrics(pred, target, config.foreground_labels,
                                                                     ignore_label=config.ignore_label)
            yield record
            del probs, predictions, p, target
            torch.cuda.empty_cache()


def mini_cases(args, baseline):
    from rankseg_benchmark.quick.datasets import get_spec
    from rankseg_benchmark.quick.metrics import ConfusionAccumulator
    path = workspace_root(ROOT) / "outputs/screening-engineering/natural-kits/run_rankseg_real_screening.py"
    spec = importlib.util.spec_from_file_location("original_cache_reader", path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    if not args.use_dispatch:
        for module in (baseline, current):
            module._RMA_CUDA_SCREENING_MIN_SINGLE_CHANNEL_DIM = 0
            module._RMA_CUDA_SCREENING_MIN_FEW_ROWS_DIM = 0
    for dataset in args.datasets:
        dataset_spec = get_spec(dataset)
        cached = Path("/tmp/rankseg-real-screening-expanded") / f"{dataset}-first100.parquet"
        options = SimpleNamespace(artifact_dir=None, cached_subset=cached if dataset != "kits" else None,
                                  limit=args.mini_limit if dataset != "kits" else args.kits_per_fold,
                                  batch_size=args.batch_size)
        for index, (probs, labels, metadata) in enumerate(reader.batches(dataset_spec, options)):
            if dataset == "kits":
                probs = probs[:, 1:2]
            record, predictions = measure(probs, baseline, "multilabel" if dataset == "kits" else "multiclass",
                                          repeats=args.repeats)
            record.update(dataset=dataset, batch=index, n_samples=probs.shape[0], metadata=metadata,
                          forced_screening=not args.use_dispatch,
                          selection="fixed first cached examples; KiTS first slices per fold")
            for name, prediction in predictions.items():
                accumulator = ConfusionAccumulator(dataset_spec.num_classes, "multiclass", dataset_spec.ignore_index,
                                                    dataset_spec.evaluation_class_ids)
                accumulator.update(prediction[:, 0].long() if dataset == "kits" else prediction, labels)
                result = accumulator.summary()
                record["methods"][name]["metrics"] = {k: float(result[k]) for k in ("mDice", "mIoU")}
            yield record
            del probs, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--suite", choices=["mini", "nnunet"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-fold", type=int, default=1)
    parser.add_argument("--mini-limit", type=int, default=100)
    parser.add_argument("--kits-per-fold", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--datasets", nargs="+", choices=["pascal_voc", "cityscapes", "ade20k", "kits"],
                        default=["pascal_voc", "cityscapes", "ade20k", "kits"])
    parser.add_argument("--use-dispatch", action="store_true", help="retain production small-input bypasses")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.cases_per_fold < 0 or min(args.mini_limit, args.kits_per_fold, args.batch_size, args.repeats) < 1:
        parser.error("sample limits, batch size and repeats must be positive; cases-per-fold may be zero for all cases")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("dataset names must be unique")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    report = {"complete": False, "suite": args.suite, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "baseline": source_hashes(args.baseline), "current": source_hashes(ROOT.parent / "rankseg"), "records": []}
    baseline = load_baseline(args.baseline)
    iterator = nnunet_cases(args, baseline) if args.suite == "nnunet" else mini_cases(args, baseline)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    for record in iterator:
        report["records"].append(record)
        args.output.write_text(json.dumps(report, indent=2))
    report["summary"] = {}
    for dataset in sorted({r["dataset"] for r in report["records"]}):
        rows = [r for r in report["records"] if r["dataset"] == dataset]
        n = sum(r["n_samples"] for r in rows)
        times = {m: sum(r["methods"][m]["median_ms"] for r in rows) / n for m in ("before", "after")}
        report["summary"][dataset] = {"samples": n, "ms_per_sample": times, "speedup": times["before"] / times["after"],
                                       "different_pixels": sum(r["different_pixels"] for r in rows),
                                       "max_regret_eps": max(r["methods"][m]["objective"]["max_regret_eps"]
                                                             for r in rows for m in ("before", "after"))}
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
