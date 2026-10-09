"""Pilot a frozen official TotalSegmentator v2 checkpoint on AMOS22 CT validation.

The default, fixed 10-case pilot is every tenth ID of the official 100-case
validation list. It is descriptive, not a full-cohort result. No outcomes are
used in case selection. Source model training subjects are not fully listed,
so patient-level separation cannot be established independently.
"""

from __future__ import annotations

import argparse
import json
import statistics
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from rankseg_benchmark.monai.cache import _atomic_json_save, _hash_file
from rankseg_benchmark.monai.small_organs import (
    ORGANS,
    _measure,
    _scores,
    native_organ_volumes,
    resolve_cases,
    size_baseline_strata,
    summarize,
    validate_native_pair,
)

DATASET_ID = "amos22_ct_official_validation_total_segmentator_v2_pilot"
MODEL_CHANNELS = {"left_adrenal": 9, "right_adrenal": 8, "gallbladder": 4,
                  "pancreas": 7, "spleen": 1, "liver": 5}
OFFICIAL_MODEL_ARCHIVE_SHA256 = "2cde8e6bcfb5b6a02183648a7036c3d6b28bc854576f8e1c34fbd6b4dd5c6c5b"


def select_cases(cases: list[tuple[str, Path, Path]], stride: int) -> list[tuple[str, Path, Path]]:
    if stride < 1:
        raise ValueError("stride must be positive")
    selected = cases[::stride]
    if stride == 10 and (len(cases) != 100 or len(selected) != 10):
        raise ValueError("The fixed pilot must select exactly ten of the official 100 cases")
    return selected


@torch.inference_mode()
def run(dataset_root: Path, model_dir: Path, output: Path, device: torch.device,
        *, stride: int = 10, warmup: int = 2, repeats: int = 9) -> dict:
    if stride < 1 or warmup < 1 or repeats < 1:
        raise ValueError("stride, warmup, and repeats must be positive")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This pinned TotalSegmentator v2 pilot requires CUDA")
    cases = resolve_cases(dataset_root)
    selected = select_cases(cases, stride)
    checkpoint = model_dir / "fold_0/checkpoint_final.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model_sha = _hash_file(checkpoint)
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from rankseg import RankSEG

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                use_mirroring=False, perform_everything_on_device=True,
                                device=device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), [0],
                                                    checkpoint_name="checkpoint_final.pth")
    labels = predictor.dataset_json["labels"]
    if len(labels) != 25 or any(labels.get(name) != channel for name, channel in (
        ("adrenal_gland_left", 9), ("adrenal_gland_right", 8), ("gallbladder", 4),
        ("pancreas", 7), ("spleen", 1), ("liver", 5)
    )):
        raise ValueError("Official TotalSegmentator v2 label/channel mapping changed")
    reader = predictor.plans_manager.image_reader_writer_class()
    decoder = RankSEG(metric="dice", solver="RMA", output_mode="multilabel",
                      safe_screening=True)
    provenance = {"study": DATASET_ID, "stride": stride, "warmup": warmup,
                  "repeats": repeats, "selected_case_ids": [case_id for case_id, _, _ in selected],
                  "rankseg_version": version("rankseg"), "nnunetv2_version": version("nnunetv2"),
                  "model_checkpoint_sha256": model_sha,
                  "model_dataset_json_sha256": _hash_file(model_dir / "dataset.json"),
                  "model_plans_json_sha256": _hash_file(model_dir / "plans.json"),
                  "model_archive_sha256": OFFICIAL_MODEL_ARCHIVE_SHA256,
                  "model_training_subject_overlap_audited": False,
                  "method": "native-grid, one-vs-rest p(target) versus 1-p(target)",
                  "device": str(device)}
    rows = []
    prior = None
    if output.exists():
        prior = json.loads(output.read_text())
        if any(prior.get(key) != value for key, value in provenance.items()):
            raise ValueError("Existing pilot output has different provenance or timing design")
        rows = prior["case_organ_results"]
        completed = {row["case_id"] for row in rows}
        if any(sum(row["case_id"] == case_id for row in rows) != len(ORGANS) for case_id in completed):
            raise ValueError("Existing pilot output has incomplete case-organ rows")
        if not completed.issubset({case_id for case_id, _, _ in selected}):
            raise ValueError("Existing pilot output has a case outside the fixed selection")
        for case_id, image_path, label_path in selected:
            if case_id not in completed:
                continue
            existing = next(row for row in rows if row["case_id"] == case_id)
            if existing["image_sha256"] != _hash_file(image_path) or (
                existing["label_sha256"] != _hash_file(label_path)
            ):
                raise ValueError(f"Existing pilot source changed: {case_id}")
    else:
        completed = set()

    for index, (case_id, image_path, label_path) in enumerate(selected, 1):
        if case_id in completed:
            print(f"[{index}/{len(selected)}] Reusing {case_id}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] Inferring {case_id}", flush=True)
        validate_native_pair(image_path, label_path)
        native = native_organ_volumes(label_path)
        image, image_props = reader.read_images([str(image_path)])
        label, label_props = reader.read_seg(str(label_path))
        if image.shape[1:] != label.shape[1:] or label.shape[0] != 1 or not np.allclose(
            image_props["spacing"], label_props["spacing"], rtol=0, atol=1e-5
        ):
            raise ValueError(f"nnU-Net reoriented image/label grid mismatch: {case_id}")
        if not np.all(np.isfinite(label)) or not np.all(label == np.rint(label)):
            raise ValueError(f"Invalid AMOS label: {case_id}")
        predicted, all_probs = predictor.predict_single_npy_array(
            image, image_props, save_or_return_probabilities=True)
        if all_probs.shape != (25, *label.shape[1:]) or all_probs.dtype != np.float32:
            raise ValueError(f"Unexpected TotalSegmentator probability grid: {case_id}")
        if not np.array_equal(predicted, np.argmax(all_probs, axis=0)):
            raise ValueError(f"TotalSegmentator native argmax does not match probabilities: {case_id}")
        truth_labels = torch.from_numpy(label[0].astype(np.uint8, copy=False)).to(device)
        source_hashes = {"image_sha256": _hash_file(image_path),
                         "label_sha256": _hash_file(label_path)}
        case_rows = []
        for organ in ORGANS:
            p = torch.from_numpy(np.ascontiguousarray(all_probs[MODEL_CHANNELS[organ.name]])).to(device)
            if not bool(torch.isfinite(p).all()) or float(p.min()) < 0 or float(p.max()) > 1:
                raise ValueError(f"Invalid probabilities for {case_id} / {organ.name}")
            p = p[None, None]
            truth = truth_labels[None] == organ.amos_label

            def argmax_call(probability=p):
                return probability[:, 0] > 0.5

            def rankseg_call(probability=p):
                return decoder.predict(probability)[:, 0]

            argmax, rankseg_mask = argmax_call().bool(), rankseg_call().bool()
            arg_dice, arg_iou, arg_tp, arg_size, gt_size = _scores(argmax, truth)
            rank_dice, rank_iou, rank_tp, rank_size, _ = _scores(rankseg_mask, truth)
            if gt_size == 0 and native[organ.name]["voxels"] > 0:
                raise ValueError(f"Reorientation erased {organ.name} in {case_id}")
            for _ in range(warmup):
                _measure(argmax_call, device)
                _measure(rankseg_call, device)
            timing = {"argmax": [], "rankseg": []}
            for repetition in range(repeats):
                calls = (("argmax", argmax_call), ("rankseg", rankseg_call))
                if (index + repetition) % 2:
                    calls = calls[::-1]
                for method, call in calls:
                    timing[method].append(_measure(call, device))
            case_rows.append({"case_id": case_id, "organ": organ.name,
                              "size_group": organ.size_group,
                              **source_hashes,
                              "gt_native_voxels": native[organ.name]["voxels"],
                              "gt_native_mm3": native[organ.name]["volume_mm3"],
                              "gt_model_voxels": gt_size, "argmax_voxels": arg_size,
                              "rankseg_voxels": rank_size, "argmax_tp": arg_tp,
                              "rankseg_tp": rank_tp, "argmax_dice": arg_dice,
                              "rankseg_dice": rank_dice, "argmax_iou": arg_iou,
                              "rankseg_iou": rank_iou,
                              "argmax_ms": statistics.median(timing["argmax"]),
                              "rankseg_ms": statistics.median(timing["rankseg"])})
            del argmax_call, rankseg_call, p, truth, argmax, rankseg_mask
        del all_probs, predicted, image, label, truth_labels
        rows.extend(case_rows)
        result = {**provenance, "complete": len(rows) == len(selected) * len(ORGANS),
                  "case_organ_results": rows}
        if result["complete"]:
            result["summary"] = summarize(rows)
            result["size_baseline_strata"] = size_baseline_strata(rows)
        _atomic_json_save(result, output)
        print(f"[{index}/{len(selected)}] Saved {case_id}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args(argv)
    result = run(args.dataset_root, args.model_dir, args.output,
                 torch.device(args.device), stride=args.stride,
                 warmup=args.warmup, repeats=args.repeats)
    print(json.dumps(result.get("summary", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
