"""Pilot RankSEG decoding inside TotalSegmentator's liver_vessels pipeline.

Run with the TotalSegmentator source tree and released RankSEG 0.0.7 on
PYTHONPATH. The first pass runs both official models and caches the target
model's probability export. The second pass replays the exact same model
outputs, replacing only the target model's hard segmentation with RankSEG.
TotalSegmentator applies its own remaining postprocessing to both outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


TARGET_TASK_ID = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(pred: np.ndarray, truth: np.ndarray) -> dict:
    p = pred == 2
    y = truth == 2
    tp = int(np.count_nonzero(p & y))
    predicted = int(np.count_nonzero(p))
    actual = int(np.count_nonzero(y))
    dice = 1.0 if predicted + actual == 0 else 2 * tp / (predicted + actual)
    union = predicted + actual - tp
    iou = 1.0 if union == 0 else tp / union
    return {
        "dice": dice,
        "iou": iou,
        "true_positive_voxels": tp,
        "false_positive_voxels": predicted - tp,
        "false_negative_voxels": actual - tp,
        "predicted_voxels": predicted,
        "ground_truth_voxels": actual,
    }


def run_case(case_id: int, dataset_root: Path, output_root: Path, device: str,
             robust_crop: bool) -> dict:
    from totalsegmentator.config import get_weights_dir, setup_nnunet

    setup_nnunet()
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from rankseg import RankSEG
    from totalsegmentator import nnunet as ts_nnunet
    from totalsegmentator.python_api import totalsegmentator

    image_path = dataset_root / "imagesTr" / f"liver_{case_id}.nii.gz"
    label_path = dataset_root / "labelsTr" / f"liver_{case_id}.nii.gz"
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing Task03 image or label for liver_{case_id}")

    case_dir = output_root / f"liver_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    probabilities_path = case_dir / "probabilities.npz"
    native_path = case_dir / "native_argmax.nii.gz"
    rankseg_path = case_dir / "rankseg.nii.gz"
    coarse_raw = case_dir / "coarse_raw.nii.gz"
    target_raw = case_dir / "target_argmax_raw.nii.gz"
    rankseg_raw = case_dir / "target_rankseg_raw.nii.gz"
    coarse_task_id = 297 if robust_crop else 298
    original_predict = ts_nnunet.nnUNetv2_predict
    mode = "infer"
    seen: list[int] = []

    def intercept(dir_in, dir_out, task_id, *args, **kwargs):
        tid = int(task_id)
        if tid not in (coarse_task_id, TARGET_TASK_ID):
            raise RuntimeError(f"Unexpected TotalSegmentator model id: {tid}")
        out_file = Path(dir_out) / "s01.nii.gz"
        seen.append(tid)
        if mode == "infer":
            original_predict(dir_in, dir_out, task_id, *args, **kwargs)
            shutil.copy2(out_file, coarse_raw if tid == coarse_task_id else target_raw)
        else:
            cached = coarse_raw if tid == coarse_task_id else rankseg_raw
            shutil.copy2(cached, out_file)

    ts_nnunet.nnUNetv2_predict = intercept
    try:
        started = time.perf_counter()
        baseline = totalsegmentator(
            image_path,
            native_path,
            ml=True,
            task="liver_vessels",
            device=device,
            quiet=True,
            save_probabilities=probabilities_path,
            robust_crop=robust_crop,
        )
        baseline_seconds = time.perf_counter() - started
        if seen != [coarse_task_id, TARGET_TASK_ID]:
            raise RuntimeError(f"Unexpected inference stages: {seen}")
        seen.clear()

        target_model = get_weights_dir() / "Dataset008_HepaticVessel" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        with (target_model / "plans.json").open() as stream:
            plans = PlansManager(json.load(stream))
        reader_writer = plans.image_reader_writer_class()
        with probabilities_path.with_suffix(".pkl").open("rb") as stream:
            properties = pickle.load(stream)
        with np.load(probabilities_path) as archive:
            probabilities = archive["probabilities"].astype(np.float32)
        if probabilities.shape[0] != 3:
            raise RuntimeError(f"Expected background, vessel, tumor; got {probabilities.shape}")
        native_raw, _ = reader_writer.read_seg(str(target_raw))
        exact_native_voxels = bool(np.array_equal(probabilities.argmax(axis=0), native_raw[0]))
        if not exact_native_voxels:
            raise RuntimeError("Probability argmax differs from TotalSegmentator's raw target mask")

        predictor = RankSEG(metric="dice", solver="RMA", output_mode="multiclass", safe_screening=True)
        tensor = torch.from_numpy(probabilities).unsqueeze(0).to(
            "cuda" if device.startswith("gpu") else "cpu"
        )
        if tensor.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        decoded = predictor(tensor)[0].to("cpu").numpy().astype(np.uint8)
        if tensor.is_cuda:
            torch.cuda.synchronize()
        decoder_seconds = time.perf_counter() - started
        if decoded.shape != native_raw[0].shape:
            raise RuntimeError("RankSEG output shape differs from the native target mask")
        reader_writer.write_seg(decoded, str(rankseg_raw), properties)
        del tensor, probabilities, decoded, native_raw

        mode = "replay"
        started = time.perf_counter()
        alternative = totalsegmentator(
            image_path,
            rankseg_path,
            ml=True,
            task="liver_vessels",
            device=device,
            quiet=True,
            robust_crop=robust_crop,
        )
        replay_seconds = time.perf_counter() - started
        if seen != [coarse_task_id, TARGET_TASK_ID]:
            raise RuntimeError(f"Unexpected replay stages: {seen}")
    finally:
        ts_nnunet.nnUNetv2_predict = original_predict

    reference = nib.load(label_path)
    if baseline.shape != reference.shape or alternative.shape != reference.shape:
        raise RuntimeError("TotalSegmentator outputs differ from the original image grid")
    if not np.allclose(baseline.affine, reference.affine, atol=1e-4):
        raise RuntimeError("Baseline output affine differs from reference")
    if not np.allclose(alternative.affine, reference.affine, atol=1e-4):
        raise RuntimeError("RankSEG output affine differs from reference")
    truth = np.asanyarray(reference.dataobj)
    native = np.asanyarray(baseline.dataobj)
    rankseg = np.asanyarray(alternative.dataobj)
    native_score = _score(native, truth)
    rankseg_score = _score(rankseg, truth)
    return {
        "case_id": f"liver_{case_id}",
        "image_sha256": _sha256(image_path),
        "label_sha256": _sha256(label_path),
        "probabilities_sha256": _sha256(probabilities_path),
        "shape": list(reference.shape),
        "probability_argmax_matches_native_raw_mask": exact_native_voxels,
        "argmax": native_score,
        "rankseg": rankseg_score,
        "delta_dice": rankseg_score["dice"] - native_score["dice"],
        "delta_iou": rankseg_score["iou"] - native_score["iou"],
        "timings_seconds": {
            "full_native_inference": baseline_seconds,
            "rankseg_decoder_single_call": decoder_seconds,
            "full_pipeline_replay_without_inference": replay_seconds,
        },
    }


def validate_default_case(case_id: int, dataset_root: Path, output_root: Path,
                          device: str, robust_crop: bool) -> dict:
    """Check the normal no-probability-export TotalSegmentator output separately."""
    from totalsegmentator.python_api import totalsegmentator

    image_path = dataset_root / "imagesTr" / f"liver_{case_id}.nii.gz"
    label_path = dataset_root / "labelsTr" / f"liver_{case_id}.nii.gz"
    case_dir = output_root / f"liver_{case_id}"
    default_path = case_dir / "official_default.nii.gz"
    started = time.perf_counter()
    default_image = totalsegmentator(
        image_path, default_path, ml=True, task="liver_vessels", device=device,
        quiet=True, robust_crop=robust_crop,
    )
    elapsed = time.perf_counter() - started
    reference = nib.load(label_path)
    exported = nib.load(case_dir / "native_argmax.nii.gz")
    if default_image.shape != reference.shape or not np.allclose(default_image.affine, reference.affine, atol=1e-4):
        raise RuntimeError("Official default output differs from the reference image grid")
    default_array = np.asanyarray(default_image.dataobj)
    export_array = np.asanyarray(exported.dataobj)
    return {
        "tumor_score": _score(default_array, np.asanyarray(reference.dataobj)),
        "different_voxels_vs_probability_export_argmax": int(np.count_nonzero(default_array != export_array)),
        "full_inference_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-ids", type=int, nargs="+", default=[0, 1, 2, 3, 32])
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--robust-crop", action="store_true",
                        help="Use TotalSegmentator's 3 mm coarse crop model instead of 6 mm")
    args = parser.parse_args()

    from totalsegmentator.config import set_config_key, setup_totalseg

    setup_totalseg()
    set_config_key("send_usage_stats", False)
    set_config_key("statistics_disclaimer_shown", True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        if results["requested_case_ids"] != args.case_ids:
            raise RuntimeError("Existing result file belongs to a different case list")
        if results.get("robust_crop", False) != args.robust_crop:
            raise RuntimeError("Existing result file belongs to a different crop mode")
    else:
        results = {
            "task": "TotalSegmentator liver_vessels (Dataset008 HepaticVessel)",
            "coarse_crop_model": "Dataset297 TotalSegmentator 3mm" if args.robust_crop
                                 else "Dataset298 TotalSegmentator 6mm",
            "robust_crop": args.robust_crop,
            "cohort": "MSD Task03 Liver imagesTr; tumor label only",
            "requested_case_ids": args.case_ids,
            "rankseg_version": importlib.metadata.version("rankseg"),
            "nnunetv2_version": importlib.metadata.version("nnunetv2"),
            "torch_version": torch.__version__,
            "device": args.device,
            "cases": [],
        }
    if results["rankseg_version"] != "0.0.7":
        raise RuntimeError("This pilot requires released RankSEG 0.0.7")
    completed = {case["case_id"] for case in results["cases"]}
    for case_id in args.case_ids:
        if f"liver_{case_id}" in completed:
            continue
        case = run_case(case_id, args.dataset_root, args.output_root, args.device,
                        args.robust_crop)
        results["cases"].append(case)
        tmp_path = results_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(results, indent=2) + "\n")
        tmp_path.replace(results_path)
        print(f"{case['case_id']}: argmax {case['argmax']['dice']:.4f}, "
              f"RankSEG {case['rankseg']['dice']:.4f}, delta {case['delta_dice']:+.4f}", flush=True)

    for case in results["cases"]:
        if "official_default" in case:
            continue
        case_id = int(case["case_id"].split("_")[1])
        case["official_default"] = validate_default_case(case_id, args.dataset_root,
                                                          args.output_root, args.device,
                                                          args.robust_crop)
        tmp_path = results_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(results, indent=2) + "\n")
        tmp_path.replace(results_path)
        print(f"{case['case_id']}: official default differs at "
              f"{case['official_default']['different_voxels_vs_probability_export_argmax']} voxels", flush=True)


if __name__ == "__main__":
    main()
