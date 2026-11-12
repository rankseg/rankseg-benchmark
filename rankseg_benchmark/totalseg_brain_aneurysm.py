"""Pilot RankSEG decoding in TotalSegmentator's single-model brain_aneurysm task.

Use a fixed TopAneu center-2 TOF-MRA cohort. The official TotalSegmentator
pipeline runs once with a probability export. A second call replays its raw
nnU-Net output with only the decoder changed to RankSEG 0.0.7. The official
pipeline performs all other postprocessing in both calls.
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


TASK_ID = 615
MODEL_NAME = "Dataset615_MAXIMUS"
DEFAULT_CASE_IDS = ("002", "004", "007", "010", "011")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(pred: np.ndarray, truth: np.ndarray) -> dict:
    p = pred > 0
    y = truth > 0
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


def _paths(dataset_root: Path, case_id: str) -> tuple[Path, Path]:
    stem = f"topaneu_center2_mr_{case_id}"
    image = dataset_root / "images" / f"{stem}_0000.nii.gz"
    label = dataset_root / "location_masks" / f"{stem}.nii.gz"
    if not image.is_file() or not label.is_file():
        raise FileNotFoundError(f"Missing image or label for {stem}")
    return image, label


def _check_grid(pred: nib.Nifti1Image, ref: nib.Nifti1Image) -> None:
    if pred.shape != ref.shape or not np.allclose(pred.affine, ref.affine, atol=1e-4):
        raise RuntimeError("Prediction and reference are on different image grids")


def run_case(case_id: str, dataset_root: Path, output_root: Path, device: str) -> dict:
    from totalsegmentator.config import get_weights_dir, setup_nnunet

    setup_nnunet()
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from rankseg import RankSEG
    from totalsegmentator import nnunet as ts_nnunet
    from totalsegmentator.python_api import totalsegmentator

    image_path, label_path = _paths(dataset_root, case_id)
    case_dir = output_root / f"topaneu_center2_mr_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    probabilities_path = case_dir / "probabilities.npz"
    baseline_path = case_dir / "native_argmax.nii.gz"
    rankseg_path = case_dir / "rankseg_foreground.nii.gz"
    multiclass_path = case_dir / "rankseg_multiclass.nii.gz"
    target_raw = case_dir / "target_argmax_raw.nii.gz"
    rankseg_raw = case_dir / "target_rankseg_foreground_raw.nii.gz"
    multiclass_raw = case_dir / "target_rankseg_multiclass_raw.nii.gz"

    original_predict = ts_nnunet.nnUNetv2_predict
    mode = "infer"
    replay_raw = rankseg_raw
    seen: list[int] = []

    def intercept(dir_in, dir_out, task_id, *args, **kwargs):
        tid = int(task_id)
        if tid != TASK_ID:
            raise RuntimeError(f"Unexpected TotalSegmentator model id: {tid}")
        seen.append(tid)
        out_file = Path(dir_out) / "s01.nii.gz"
        if mode == "infer":
            original_predict(dir_in, dir_out, task_id, *args, **kwargs)
            shutil.copy2(out_file, target_raw)
        else:
            shutil.copy2(replay_raw, out_file)

    ts_nnunet.nnUNetv2_predict = intercept
    try:
        started = time.perf_counter()
        baseline = totalsegmentator(
            image_path, baseline_path, ml=True, task="brain_aneurysm",
            device=device, quiet=True, save_probabilities=probabilities_path,
        )
        baseline_seconds = time.perf_counter() - started
        if seen != [TASK_ID]:
            raise RuntimeError(f"Expected one model, saw {seen}")
        seen.clear()

        model_root = get_weights_dir() / MODEL_NAME
        model_dirs = list(model_root.glob("*__nnUNetPlans__3d_fullres"))
        if len(model_dirs) != 1:
            raise RuntimeError(f"Expected one nnU-Net model directory at {model_root}")
        with (model_dirs[0] / "plans.json").open() as stream:
            plans = PlansManager(json.load(stream))
        reader_writer = plans.image_reader_writer_class()
        with probabilities_path.with_suffix(".pkl").open("rb") as stream:
            properties = pickle.load(stream)
        with np.load(probabilities_path) as archive:
            probabilities = archive["probabilities"].astype(np.float32)
        if probabilities.shape[0] != 2:
            raise RuntimeError(f"Expected background and aneurysm channels, got {probabilities.shape}")
        native_raw, _ = reader_writer.read_seg(str(target_raw))
        exact_raw = bool(np.array_equal(probabilities.argmax(axis=0), native_raw[0]))
        if not exact_raw:
            raise RuntimeError("Exported probability argmax differs from the official raw mask")

        foreground_predictor = RankSEG(metric="dice", solver="RMA", output_mode="multilabel", safe_screening=True)
        multiclass_predictor = RankSEG(metric="dice", solver="RMA", output_mode="multiclass", safe_screening=True)
        tensor = torch.from_numpy(probabilities).unsqueeze(0).to("cuda" if device.startswith("gpu") else "cpu")
        if tensor.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        decoded = foreground_predictor(tensor[:, 1:2])[0, 0].to("cpu").numpy().astype(np.uint8)
        if tensor.is_cuda:
            torch.cuda.synchronize()
        decoder_seconds = time.perf_counter() - started
        if decoded.shape != native_raw[0].shape:
            raise RuntimeError("RankSEG output shape differs from native target mask")
        reader_writer.write_seg(decoded, str(rankseg_raw), properties)
        if tensor.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        decoded_multiclass = multiclass_predictor(tensor)[0].to("cpu").numpy().astype(np.uint8)
        if tensor.is_cuda:
            torch.cuda.synchronize()
        multiclass_decoder_seconds = time.perf_counter() - started
        if decoded_multiclass.shape != native_raw[0].shape:
            raise RuntimeError("Multiclass RankSEG output shape differs from native target mask")
        reader_writer.write_seg(decoded_multiclass, str(multiclass_raw), properties)
        del tensor, probabilities, decoded, native_raw

        mode = "replay"
        started = time.perf_counter()
        alternative = totalsegmentator(
            image_path, rankseg_path, ml=True, task="brain_aneurysm",
            device=device, quiet=True,
        )
        replay_seconds = time.perf_counter() - started
        if seen != [TASK_ID]:
            raise RuntimeError(f"Expected one replayed model, saw {seen}")
        seen.clear()
        replay_raw = multiclass_raw
        multiclass_alternative = totalsegmentator(
            image_path, multiclass_path, ml=True, task="brain_aneurysm",
            device=device, quiet=True,
        )
        if seen != [TASK_ID]:
            raise RuntimeError(f"Expected one replayed multiclass model, saw {seen}")
    finally:
        ts_nnunet.nnUNetv2_predict = original_predict

    reference = nib.load(label_path)
    _check_grid(baseline, reference)
    _check_grid(alternative, reference)
    _check_grid(multiclass_alternative, reference)
    truth = np.asanyarray(reference.dataobj)
    argmax_score = _score(np.asanyarray(baseline.dataobj), truth)
    rankseg_score = _score(np.asanyarray(alternative.dataobj), truth)
    multiclass_score = _score(np.asanyarray(multiclass_alternative.dataobj), truth)
    return {
        "case_id": f"topaneu_center2_mr_{case_id}",
        "image_sha256": _sha256(image_path),
        "label_sha256": _sha256(label_path),
        "probabilities_sha256": _sha256(probabilities_path),
        "shape": list(reference.shape),
        "probability_argmax_matches_native_raw_mask": exact_raw,
        "argmax": argmax_score,
        "rankseg": rankseg_score,
        "rankseg_multiclass": multiclass_score,
        "delta_dice": rankseg_score["dice"] - argmax_score["dice"],
        "delta_iou": rankseg_score["iou"] - argmax_score["iou"],
        "timings_seconds": {
            "full_native_inference": baseline_seconds,
            "rankseg_decoder_single_call": decoder_seconds,
            "rankseg_multiclass_decoder_single_call": multiclass_decoder_seconds,
            "full_pipeline_replay_without_inference": replay_seconds,
        },
    }


def validate_default(case_id: str, dataset_root: Path, output_root: Path, device: str) -> dict:
    """Compare against an unmodified no-probability-export API call."""
    from totalsegmentator.python_api import totalsegmentator

    image_path, label_path = _paths(dataset_root, case_id)
    case_dir = output_root / f"topaneu_center2_mr_{case_id}"
    started = time.perf_counter()
    default = totalsegmentator(
        image_path, case_dir / "official_default.nii.gz", ml=True,
        task="brain_aneurysm", device=device, quiet=True,
    )
    elapsed = time.perf_counter() - started
    reference = nib.load(label_path)
    _check_grid(default, reference)
    exported = nib.load(case_dir / "native_argmax.nii.gz")
    return {
        "aneurysm_score": _score(np.asanyarray(default.dataobj), np.asanyarray(reference.dataobj)),
        "different_voxels_vs_probability_export_argmax": int(np.count_nonzero(
            np.asanyarray(default.dataobj) != np.asanyarray(exported.dataobj))),
        "full_inference_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASE_IDS))
    parser.add_argument("--all-center2", action="store_true",
                        help="Evaluate every downloaded TopAneu center-2 MR image (40 cases)")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.all_center2:
        pattern = "topaneu_center2_mr_*_0000.nii.gz"
        args.case_ids = sorted(path.name.split("_")[3] for path in
                               (args.dataset_root / "images").glob(pattern))
        if len(args.case_ids) != 40:
            raise RuntimeError(f"Expected 40 center-2 MR images, got {len(args.case_ids)}")

    from totalsegmentator.config import set_config_key, setup_totalseg

    setup_totalseg()
    set_config_key("send_usage_stats", False)
    set_config_key("statistics_disclaimer_shown", True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        previous = results["requested_case_ids"]
        if previous != args.case_ids and previous != args.case_ids[:len(previous)]:
            raise RuntimeError("Existing results belong to a different case list")
        if previous != args.case_ids:
            results["requested_case_ids"] = args.case_ids
            results["cohort"] = "TopAneu 2026 center-2 MR, all 40 numerical IDs"
            results_path.write_text(json.dumps(results, indent=2) + "\n")
    else:
        results = {
            "task": "TotalSegmentator brain_aneurysm (Dataset615 MAXIMUS)",
            "cohort": "TopAneu 2026 center-2 MR, all 40 numerical IDs" if args.all_center2
                      else "TopAneu 2026 center-2 MR, first five numerical IDs",
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
        if f"topaneu_center2_mr_{case_id}" in completed:
            continue
        row = run_case(case_id, args.dataset_root, args.output_root, args.device)
        results["cases"].append(row)
        tmp = results_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, indent=2) + "\n")
        tmp.replace(results_path)
        print(f"{row['case_id']}: argmax {row['argmax']['dice']:.4f}, "
              f"RankSEG {row['rankseg']['dice']:.4f}, delta {row['delta_dice']:+.4f}", flush=True)

    for row in results["cases"]:
        if "official_default" in row:
            continue
        case_id = row["case_id"].split("_")[-1]
        row["official_default"] = validate_default(case_id, args.dataset_root,
                                                     args.output_root, args.device)
        tmp = results_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, indent=2) + "\n")
        tmp.replace(results_path)
        print(f"{row['case_id']}: default differs at "
              f"{row['official_default']['different_voxels_vs_probability_export_argmax']} voxels", flush=True)


if __name__ == "__main__":
    main()
