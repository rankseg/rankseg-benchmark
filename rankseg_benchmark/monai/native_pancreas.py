"""Native-grid MSD and CPTAC expert-label tests of the MONAI DiNTS Bundle.

The expert pancreas/tumor masks were published after the MSD challenge and are
kept separate from the 281 training images. Both decoders operate after the
Bundle's softmax and inverse transform, in the native test-image grid.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from .matched_official import _hash_file, _summary, score_foreground
from .native_prostate import _write_json


@dataclass(frozen=True)
class PancreasCase:
    case_id: str
    image: Path
    pancreas: Path
    tumor: Path


def resolve_expert_msd_test_cases(dataset_root: Path, expert_root: Path) -> list[PancreasCase]:
    image_train = {path.name for path in (dataset_root / "imagesTr").glob("pancreas_*.nii.gz")}
    image_test = {path.name for path in (dataset_root / "imagesTs").glob("pancreas_*.nii.gz")}
    if len(image_train) != 281 or len(image_test) != 139 or image_train & image_test:
        raise ValueError("MSD Task07 train/test image split is incomplete or overlapping")
    pancreas_dir = expert_root / "Pancreas_segmentations"
    tumor_dir = expert_root / "Tumor_segmentations"
    pancreas_files = sorted(pancreas_dir.glob("pancreas_*_pancreas.nii.gz"))
    if len(pancreas_files) != 49:
        raise ValueError("Expected 49 post-hoc expert masks for MSD imagesTs")
    cases = []
    for pancreas in pancreas_files:
        case_id = pancreas.name.removesuffix("_pancreas.nii.gz")
        image = dataset_root / "imagesTs" / f"{case_id}.nii.gz"
        tumor = tumor_dir / f"{case_id}_tumor.nii.gz"
        if image.name not in image_test or image.name in image_train or not tumor.is_file():
            raise ValueError(f"Missing or train-overlapping image/tumor pair: {case_id}")
        images = [nib.load(str(path)) for path in (image, pancreas, tumor)]
        if any(scan.shape != images[0].shape or not np.allclose(scan.affine, images[0].affine, atol=1e-4)
               for scan in images[1:]):
            raise ValueError(f"Image/mask native grid mismatch: {case_id}")
        cases.append(PancreasCase(case_id, image, pancreas, tumor))
    expected_tumor_names = {f"{case.case_id}_tumor.nii.gz" for case in cases}
    actual_tumor_names = {path.name for path in tumor_dir.glob("pancreas_*_tumor.nii.gz")}
    if actual_tumor_names != expected_tumor_names:
        raise ValueError("Pancreas and tumor expert mask sets do not match")
    return cases


def resolve_expert_cptac_cases(volume_root: Path, expert_root: Path) -> list[PancreasCase]:
    """Pair the 42 CPTAC CTs (underscores) with expert masks (hyphens)."""
    pancreas_dir = expert_root / "Pancreas_segmentations"
    tumor_dir = expert_root / "Tumor_segmentations"
    pancreas_files = sorted(pancreas_dir.glob("C3*_pancreas.nii.gz"))
    image_files = list(volume_root.glob("C3*.nii.gz"))
    if len(pancreas_files) != 42 or len(image_files) != 42:
        raise ValueError("Expected exactly 42 external CPTAC images and mask pairs")
    cases = []
    for pancreas in pancreas_files:
        case_id = pancreas.name.removesuffix("_pancreas.nii.gz")
        image = volume_root / f"{case_id.replace('-', '_')}.nii.gz"
        tumor = tumor_dir / f"{case_id}_tumor.nii.gz"
        if not image.is_file() or not tumor.is_file():
            raise ValueError(f"Missing CPTAC image or tumor mask: {case_id}")
        scans = [nib.load(str(path)) for path in (image, pancreas, tumor)]
        if any(scan.shape != scans[0].shape or
               not np.allclose(scan.affine, scans[0].affine, atol=1e-4)
               for scan in scans[1:]):
            raise ValueError(f"CPTAC image/mask native grid mismatch: {case_id}")
        cases.append(PancreasCase(case_id, image, pancreas, tumor))
    expected_image_names = {case.image.name for case in cases}
    expected_tumor_names = {case.tumor.name for case in cases}
    if {path.name for path in image_files} != expected_image_names or \
       {path.name for path in tumor_dir.glob("C3*_tumor.nii.gz")} != expected_tumor_names:
        raise ValueError("CPTAC image/pancreas/tumor sets do not match exactly")
    return cases


def native_expert_label(case: PancreasCase) -> torch.Tensor:
    pancreas = np.asarray(nib.load(str(case.pancreas)).dataobj)
    tumor = np.asarray(nib.load(str(case.tumor)).dataobj)
    if not np.all(np.isin(pancreas, (0, 1))) or not np.all(np.isin(tumor, (0, 1))):
        raise ValueError(f"Expert masks are not binary: {case.case_id}")
    label = np.zeros(pancreas.shape, dtype=np.uint8)
    label[pancreas == 1] = 1
    label[tumor == 1] = 2  # tumor overrides overlapping whole-pancreas mask
    if not np.any(label == 1) or not np.any(label == 2):
        raise ValueError(f"Missing pancreas or tumor foreground: {case.case_id}")
    return torch.from_numpy(label.astype(np.int64))


def _components(bundle_root: Path):
    from monai.bundle import ConfigParser
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped,
                                  LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd)

    config_path = bundle_root / "configs" / "inference.yaml"
    config = yaml.safe_load(config_path.read_text())
    entries = config["preprocessing"]["transforms"]
    if [step["_target_"] for step in entries] != [
        "LoadImaged", "EnsureChannelFirstd", "Orientationd", "Spacingd",
        "ScaleIntensityRanged", "EnsureTyped"
    ]:
        raise ValueError("Unexpected DiNTS Bundle preprocessing")
    post = config["postprocessing"]["transforms"]
    if [step["_target_"] for step in post] != [
        "Activationsd", "Invertd", "AsDiscreted", "SaveImaged"
    ] or post[0]["softmax"] is not True or post[1]["nearest_interp"] is not False or post[2]["argmax"] is not True:
        raise ValueError("Unexpected DiNTS Bundle postprocessing")
    intensity = entries[4]
    preprocessing = Compose([
        LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes=entries[2]["axcodes"]),
        Spacingd(keys="image", pixdim=tuple(entries[3]["pixdim"]), mode=entries[3]["mode"]),
        ScaleIntensityRanged(keys="image", a_min=intensity["a_min"], a_max=intensity["a_max"],
                             b_min=intensity["b_min"], b_max=intensity["b_max"], clip=intensity["clip"]),
        EnsureTyped(keys="image"),
    ])
    parser = ConfigParser()
    parser.read_config(config_path)
    parser["bundle_root"] = str(bundle_root)
    parser["device"] = torch.device("cuda")
    architecture_path = bundle_root / "models" / "search_code_18590.pt"
    parser["arch_ckpt"] = torch.load(architecture_path, map_location="cuda", weights_only=False)
    parser["dints_space#device"] = "cuda"
    model = parser.get_parsed_content("network_def").to("cuda")
    checkpoint_path = bundle_root / "models" / "model.pt"
    weights = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    model.load_state_dict(weights.get("model", weights), strict=True)
    model.eval()
    inferer_spec = config["inferer"]
    inferer = SlidingWindowInferer(
        roi_size=tuple(inferer_spec["roi_size"]),
        sw_batch_size=int(inferer_spec["sw_batch_size"]), overlap=float(inferer_spec["overlap"]),
    )
    return config, preprocessing, model, inferer, config_path, checkpoint_path, architecture_path


def _saved_prediction(prediction: torch.Tensor, probability_meta, case: PancreasCase,
                      save_dir: Path) -> torch.Tensor:
    from monai.data import MetaTensor
    from monai.transforms import SaveImaged

    data = {"pred": MetaTensor(prediction[None].cpu(), meta=dict(probability_meta))}
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=str(save_dir),
                output_ext=".nii.gz", output_dtype=np.float32,
                output_postfix="trans", separate_folder=True)(data)
    image_id = case.image.name.removesuffix(".nii.gz")
    saved_path = save_dir / image_id / f"{image_id}_trans.nii.gz"
    saved = nib.load(str(saved_path))
    native_image = nib.load(str(case.image))
    if saved.shape != native_image.shape or not np.allclose(saved.affine, native_image.affine, atol=1e-4):
        raise ValueError(f"Saved DiNTS prediction is not in the native test grid: {case.case_id}")
    saved_values = np.asarray(saved.dataobj)
    if not np.array_equal(saved_values, prediction.cpu().numpy()):
        raise ValueError(f"Saved DiNTS prediction differs from decoded labels: {case.case_id}")
    return torch.from_numpy(saved_values.astype(np.int64))


@torch.inference_mode()
def run(dataset_root: Path, expert_root: Path, bundle_root: Path, output: Path,
        *, cohort: str = "msd_imagesTs", limit: int | None = None) -> dict:
    if version("rankseg") != "0.0.7" or version("monai") != "1.4.0":
        raise RuntimeError("Pin RankSEG 0.0.7 and MONAI 1.4.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MONAI Bundle benchmark")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if cohort == "msd_imagesTs":
        cases = resolve_expert_msd_test_cases(dataset_root, expert_root)
        cohort_audit = {"imagesTr": 281, "imagesTs": 139,
                        "posthoc_expert_labeled_imagesTs": 49, "training_id_overlap": 0}
        task = "msd_task07_imagesTs_expert49_native"
    elif cohort == "cptac_pda":
        cases = resolve_expert_cptac_cases(dataset_root, expert_root)
        volume_archive = dataset_root.parent / "Volume_files.zip"
        archive_md5 = _hash_file(volume_archive, "md5")
        if archive_md5 != "6068ad9fc389c56d6c448f4370af6051":
            raise ValueError("CPTAC volume archive differs from the publisher's MD5")
        cohort_audit = {"external_cptac_pda_cases": 42,
                        "external_case_ids_absent_from_msd": True,
                        "volume_archive_md5": archive_md5}
        task = "cptac_pda_expert42_native"
    else:
        raise ValueError(f"Unknown cohort: {cohort}")
    if limit is not None:
        cases = cases[:limit]
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.benchmark = True
    config, preprocessing, model, inferer, config_path, checkpoint_path, architecture_path = _components(bundle_root)
    from monai.data import MetaTensor
    from monai.transforms import Invertd
    from rankseg import RankSEG
    rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass", safe_screening=True)
    metadata = {
        "task": task, "bundle": "pancreas_ct_dints_segmentation",
        "bundle_version": "0.5.2", "rankseg_version": version("rankseg"),
        "monai_version": version("monai"), "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(), "seed": 123,
        "bundle_config_sha256": _hash_file(config_path),
        "checkpoint_sha256": _hash_file(checkpoint_path),
        "architecture_sha256": _hash_file(architecture_path),
        "cohort": cohort_audit,
        "rankseg": {"metric": "dice", "solver": "RMA", "output_mode": "multiclass",
                    "safe_screening": True},
        "inference": {"amp": bool(config["evaluator"]["amp"]),
                      "sw_batch_size": int(config["inferer"]["sw_batch_size"]),
                      "coordinate_space": "native_image_grid_after_official_inverse",
                      "label_combination": "tumor class 2 overrides whole-pancreas class 1"},
        "timing": {"scope": "decoder_only", "calls_per_case": 1,
                   "method": "synchronized CUDA wall time"},
        "complete": False,
    }
    records = []
    for index, case in enumerate(cases):
        item = preprocessing({"image": str(case.image)})
        image = item["image"].as_tensor()[None].to("cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", enabled=bool(config["evaluator"]["amp"])):
            logits = inferer(image, model)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000
        probs = logits.float().softmax(dim=1)[0]
        inverted = Invertd(keys="pred", transform=preprocessing, orig_keys="image",
                           meta_key_postfix="meta_dict", nearest_interp=False, to_tensor=True)(
            {"image": item["image"], "pred": MetaTensor(probs.cpu())}
        )
        native_probs = inverted["pred"]
        if tuple(native_probs.shape[1:]) != nib.load(str(case.image)).shape:
            raise ValueError(f"Inverse probability shape mismatch: {case.case_id}")
        native_probs_tensor = native_probs.as_tensor().to("cuda")
        if not bool(torch.isfinite(native_probs_tensor).all()):
            raise ValueError(f"Non-finite native probabilities: {case.case_id}")
        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        argmax = native_probs_tensor.argmax(dim=0)
        torch.cuda.synchronize()
        argmax_ms = (time.perf_counter() - decode_start) * 1000
        decode_start = time.perf_counter()
        rma = rankseg.predict(native_probs_tensor[None].contiguous())[0]
        torch.cuda.synchronize()
        rma_ms = (time.perf_counter() - decode_start) * 1000
        label = native_expert_label(case)
        scores = {}
        for method, prediction in (("argmax", argmax), ("rma", rma)):
            prediction_dir = ("monai_native_pancreas_predictions_007" if cohort == "msd_imagesTs"
                              else "monai_native_cptac_predictions_007")
            saved = _saved_prediction(prediction, native_probs.meta, case,
                                      output.parent / prediction_dir / case.case_id / method)
            scores[method] = score_foreground(saved, label, (1, 2))
        records.append({
            "case_id": case.case_id, "native_shape": list(label.shape),
            "scores": {"expert": scores},
            "timing": {"argmax": {"single_call_ms": argmax_ms},
                       "rma": {"single_call_ms": rma_ms}},
            "inference_ms": inference_ms, "changed_native_voxels": int((argmax != rma).sum()),
            "source_sha256": {"image": _hash_file(case.image),
                              "pancreas": _hash_file(case.pancreas), "tumor": _hash_file(case.tumor)},
        })
        _write_json(output.with_suffix(".partial.json"), {**metadata, "cases": records})
        print(f"[{index + 1}/{len(cases)}] {case.case_id}", flush=True)
        del item, image, logits, probs, native_probs, native_probs_tensor, argmax, rma
        torch.cuda.empty_cache()
    # Reuse the score aggregation without claiming a one-call timing is a median.
    timing_projection = [
        {"scores": record["scores"],
         "timing": {method: {"median_ms": record["timing"][method]["single_call_ms"]}
                    for method in ("argmax", "rma")}}
        for record in records
    ]
    summary = _summary(timing_projection, ("expert",), (1, 2))
    summary["mean_single_call_ms"] = summary.pop("mean_of_case_median_ms")
    result = {**metadata, "complete": limit is None, "summary": summary,
              "cases": records}
    _write_json(output, result)
    output.with_suffix(".partial.json").unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("msd_imagesTs", "cptac_pda"), default="msd_imagesTs")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = run(args.dataset_root, args.expert_root, args.bundle_root, args.output,
                 cohort=args.cohort, limit=args.limit)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
