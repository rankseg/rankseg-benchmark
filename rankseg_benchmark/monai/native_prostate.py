"""Native-grid Prostate158 test of the released MONAI Bundle with RankSEG 0.0.7.

Both arms use the Bundle's preprocessing, model, largest-component filter and
inverse transform. Only the model-grid decoder changes.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .matched_official import (
    _bundle_components,
    _hash_file,
    _paired_times,
    _preprocessing,
    _summary,
    audit_prostate_extraction,
    audit_prostate_split_lists,
    resolve_prostate_cases,
    score_foreground,
)


def native_label(path: Path, image_path: Path) -> torch.Tensor:
    """Load a reference in the publisher's native image coordinate array."""
    reference = nib.load(str(path))
    image = nib.load(str(image_path))
    if reference.shape != image.shape or not np.allclose(reference.affine, image.affine, atol=1e-4):
        raise ValueError(f"Native image/reference grid mismatch: {path}")
    label = np.asarray(reference.dataobj)
    if not np.all(np.isin(label, (0, 1, 2))):
        raise ValueError(f"Unexpected native label IDs: {path}")
    return torch.as_tensor(label.astype(np.int64))


def postprocess_native(
    segmentation: torch.Tensor, item: dict, preprocessing, expected_affine: np.ndarray,
    save_dir: Path | None = None,
) -> tuple[torch.Tensor, int]:
    """Apply Bundle KLC -> inverse, then round its bilinear inverse for scoring.

    The official config sets Invertd(nearest_interp=False) *after* AsDiscreted,
    so its saved native-grid prediction can contain fractional label IDs.
    Rounding is applied only to the scoring copy, identically for both arms.
    """
    from monai.transforms import Invertd, KeepLargestConnectedComponentd, SaveImaged

    if segmentation.ndim != 3 or segmentation.dtype != torch.long:
        raise ValueError("Decoder must return one 3D integer label map")
    data = {"image": item["image"], "pred": segmentation[None].cpu()}
    data = KeepLargestConnectedComponentd(keys="pred", applied_labels=[1, 2])(data)
    data = Invertd(
        keys="pred", transform=preprocessing, orig_keys="image",
        meta_key_postfix="meta_dict", nearest_interp=False, to_tensor=True,
    )(data)
    inverse = data["pred"]
    affine = inverse.meta.get("affine")
    if affine is None or not np.allclose(np.asarray(affine), expected_affine, atol=1e-4):
        raise ValueError("Bundle inverse prediction affine differs from native image")
    values = inverse.as_tensor().squeeze(0)
    if save_dir is not None:
        # This is the final official Bundle step. Isolate each case/arm because
        # every Prostate158 source image is named t2.nii.gz.
        SaveImaged(keys="pred", resample=False, meta_keys="pred_meta_dict",
                    output_dir=str(save_dir))(data)
        saved_path = save_dir / "t2" / "t2_trans.nii.gz"
        saved = nib.load(str(saved_path))
        if saved.shape != tuple(values.shape) or not np.allclose(saved.affine, expected_affine, atol=1e-4):
            raise ValueError(f"Saved Bundle prediction grid mismatch: {saved_path}")
        saved_values = np.asarray(saved.dataobj)
        if not np.allclose(saved_values, values.numpy(), atol=1e-5):
            raise ValueError(f"Saved Bundle prediction differs from the scoring tensor: {saved_path}")
    rounded = values.round()
    fractional = int((values != rounded).sum())
    result = rounded.to(torch.long)
    if bool(((result < 0) | (result > 2)).any()):
        raise ValueError("Inverted prediction contains unexpected label IDs")
    return result, fractional


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def run(dataset_root: Path, training_list_root: Path, test_archive: Path,
        bundle_root: Path, output: Path, *, limit: int | None = None,
        warmup: int = 2, repeats: int = 7) -> dict:
    if version("rankseg") != "0.0.7":
        raise RuntimeError("This experiment is pinned to RankSEG 0.0.7")
    if version("monai") != "1.4.0":
        raise RuntimeError("This experiment is pinned to MONAI 1.4.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pinned MONAI benchmark")
    if warmup < 1 or repeats < 1 or (limit is not None and limit < 1):
        raise ValueError("warmup, repeats and limit must be positive")
    cases = resolve_prostate_cases(dataset_root)
    audit = audit_prostate_split_lists(cases, training_list_root)
    audit["test_zip_md5"] = _hash_file(test_archive, "md5")
    if audit["test_zip_md5"] != "403b3736cb12013ed7585e07cbda6978":
        raise ValueError("Prostate158 test archive MD5 differs from the publisher's checksum")
    audit["evaluated_files_verified_against_test_zip"] = audit_prostate_extraction(
        cases, dataset_root, test_archive
    )
    if limit:
        cases = cases[:limit]
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    # The Bundle's initialize block explicitly enables cuDNN benchmarking.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    config, model, _, config_path, checkpoint_path = _bundle_components(bundle_root, torch.device("cuda"))
    from monai.inferers import SlidingWindowInferer
    inferer_spec = config["inferer"]
    inferer = SlidingWindowInferer(
        roi_size=tuple(inferer_spec["roi_size"]),
        sw_batch_size=int(inferer_spec["sw_batch_size"]),
        overlap=float(inferer_spec["overlap"]),
    )
    assert config["network_def"]["out_channels"] == 3
    steps = config["postprocessing"]["transforms"]
    if [step["_target_"] for step in steps] != [
        "AsDiscreted", "KeepLargestConnectedComponentd", "Invertd", "SaveImaged"
    ] or steps[1]["applied_labels"] != [1, 2] or steps[2]["nearest_interp"] is not False:
        raise ValueError("Unexpected official Bundle postprocessing")
    preprocessing = _preprocessing(config, "prostate158_test", ())
    from rankseg import RankSEG
    rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass", safe_screening=True)
    full_sort = RankSEG(metric="dice", solver="RMA", output_mode="multiclass", safe_screening=False)
    records: list[dict] = []
    metadata = {
        "task": "prostate158_native_test", "bundle": "prostate_mri_anatomy",
        "bundle_version": "0.3.6", "bundle_config_sha256": _hash_file(config_path),
        "checkpoint_sha256": _hash_file(checkpoint_path), "rankseg_version": version("rankseg"),
        "monai_version": version("monai"), "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(), "seed": 123, "cohort_audit": audit,
        "cudnn_benchmark": True,
        "rankseg": {"metric": "dice", "solver": "RMA", "output_mode": "multiclass",
                    "safe_screening": True},
        "inference": {"amp": bool(config["evaluator"]["amp"]),
                      "sw_batch_size": int(inferer_spec["sw_batch_size"]),
                      "postprocess": "official KLC then Invertd(nearest_interp=False) then SaveImaged(resample=False)",
                      "scoring_quantization": "torch.round after official inverse; identical for both arms",
                      "coordinate_space": "publisher_native_image_grid"},
        "timing": {"warmup": warmup, "repeats": repeats,
                   "decoder": "synchronized CUDA, mean of case medians"},
        "complete": False,
    }
    for index, case in enumerate(cases):
        started = time.perf_counter()
        item = preprocessing({"image": str(case.image)})
        image = item["image"].as_tensor()[None].to("cuda")
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.autocast(device_type="cuda", enabled=bool(config["evaluator"]["amp"])):
            logits = inferer(image, model)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_started) * 1000
        probabilities = logits.float().softmax(dim=1).contiguous()
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError(f"Non-finite model output: {case.case_id}")
        base = probabilities.argmax(dim=1)[0]
        rma = rankseg.predict(probabilities)[0]
        # Screening is an experimental 0.0.7 path. Its output can differ from
        # full sorting; quantify that difference instead of silently assuming
        # bitwise identity (also observed in the earlier model-grid study).
        screening_changed = int((rma != full_sort.predict(probabilities)[0]).sum())
        timing = _paired_times(probabilities, rankseg, warmup=warmup, repeats=repeats, case_index=index)
        expected_affine = nib.load(str(case.image)).affine
        scores: dict = {name: {} for name in case.labels}
        fractional: dict = {}
        shapes: dict = {}
        post_ms: dict = {}
        for method, segmentation in (("argmax", base), ("rma", rma)):
            post_started = time.perf_counter()
            native, fractional_count = postprocess_native(
                segmentation, item, preprocessing, expected_affine,
                output.parent / "monai_native_prostate_predictions_007" / case.case_id / method,
            )
            post_ms[method] = (time.perf_counter() - post_started) * 1000
            fractional[method] = fractional_count
            shapes[method] = list(native.shape)
            for reader, label_path in case.labels.items():
                reference = native_label(label_path, case.image)
                scores[reader][method] = score_foreground(native, reference, (1, 2))
        record = {
            "case_id": case.case_id, "model_grid_shape": list(base.shape),
            "native_shape": shapes["argmax"], "scores": scores,
            "timing": timing, "inference_ms": inference_ms,
            "postprocess_ms": post_ms, "case_wall_ms": (time.perf_counter() - started) * 1000,
            "changed_model_grid_voxels": int((base != rma).sum()),
            "screening_changed_voxels_vs_full_sort": screening_changed,
            "fractional_voxels_after_official_inverse": fractional,
            "source_sha256": {"image": _hash_file(case.image),
                              "labels": {reader: _hash_file(path) for reader, path in case.labels.items()}},
        }
        if shapes["argmax"] != shapes["rma"]:
            raise ValueError("Native prediction shapes differ")
        records.append(record)
        _write_json(output.with_suffix(".partial.json"), {**metadata, "cases": records})
        print(f"[{index + 1}/{len(cases)}] {case.case_id}", flush=True)
        del item, image, logits, probabilities, base, rma
    result = {**metadata, "complete": limit is None, "summary": _summary(records, ("reader1", "reader2"), (1, 2)),
              "cases": records}
    _write_json(output, result)
    output.with_suffix(".partial.json").unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-list-root", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    result = run(args.dataset_root, args.training_list_root, args.test_archive,
                 args.bundle_root, args.output, limit=args.limit,
                 warmup=args.warmup, repeats=args.repeats)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
