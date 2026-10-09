"""AMOS22 CT validation: paired small/large-organ study with a frozen MONAI BTCV Bundle.

The official AMOS22 validation split is fixed before inference. Its public CT
mirror is pinned in the documentation; this module also accepts the original
extracted AMOS22 archive. Images, labels, and probability caches remain local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from rankseg_benchmark.monai.cache import (
    _atomic_json_save,
    _atomic_torch_save,
    _build_model,
    _build_preprocessing,
    _hash_file,
    _hash_json,
    _infer_case,
    _require_monai,
    _software_info,
    load_monai_specs,
)

DATASET_ID = "amos22_ct_official_validation"
EXPECTED_CASES = 100
CASE_LIST_SHA256 = "6f4c322228a29b8eee612ccea0056759bbac6e0e704f21ce0957c1b16dcf3891"
MIRROR_REPO = "MedOtter/amos22-ct-dataset"
MIRROR_REVISION = "c67f7c01e66277038d87975b03b73ece489a3035"
BTCV_SPEC_ID = "monai_btcv_swin_v058_msd_pancreas"


@dataclass(frozen=True)
class Organ:
    name: str
    amos_label: int
    model_channel: int
    size_group: str


ORGANS = (
    Organ("left_adrenal", 12, 13, "small"),
    Organ("right_adrenal", 11, 12, "small"),
    Organ("gallbladder", 4, 4, "small"),
    Organ("pancreas", 10, 11, "medium"),
    Organ("spleen", 1, 1, "large"),
    Organ("liver", 6, 6, "large"),
)


def resolve_cases(dataset_root: Path, *, expected_count: int = EXPECTED_CASES,
                  expected_digest: str = CASE_LIST_SHA256) -> list[tuple[str, Path, Path]]:
    """Use every labeled CT validation case, never a prediction-selected subset."""
    root = dataset_root / "valid" if (dataset_root / "valid").is_dir() else dataset_root
    images_dir, labels_dir = root / "imagesVa", root / "labelsVa"
    images = {path.name[:-7]: path for path in images_dir.glob("amos_*.nii.gz")}
    labels = {path.name[:-7]: path for path in labels_dir.glob("amos_*.nii.gz")}
    if images.keys() != labels.keys():
        raise ValueError(f"AMOS image/label case mismatch: images-only={sorted(images.keys()-labels.keys())}, "
                         f"labels-only={sorted(labels.keys()-images.keys())}")
    case_ids = sorted(images)
    if len(case_ids) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} labeled AMOS CT validation cases, found {len(case_ids)}")
    if any(not name.startswith("amos_") or not name[5:].isdigit() or int(name[5:]) >= 500 for name in case_ids):
        raise ValueError("AMOS validation cohort contains a non-CT or malformed case ID")
    digest = hashlib.sha256("\n".join(case_ids).encode()).hexdigest()
    if digest != expected_digest:
        raise ValueError(f"AMOS validation case-list fingerprint mismatch: {digest}")
    return [(name, images[name], labels[name]) for name in case_ids]


def validate_payload(payload: dict) -> None:
    probabilities, label = payload.get("probabilities"), payload.get("label")
    if not isinstance(probabilities, torch.Tensor) or not isinstance(label, torch.Tensor):
        raise TypeError("AMOS artifact requires probability and label tensors")
    if probabilities.dtype != torch.float32 or label.dtype != torch.uint8:
        raise TypeError("AMOS artifact requires float32 probabilities and uint8 labels")
    if probabilities.ndim != 4 or probabilities.shape[0] != len(ORGANS) or tuple(probabilities.shape[1:]) != tuple(label.shape):
        raise ValueError("AMOS artifact probability/label shape mismatch")
    if label.ndim != 3 or not bool(((label >= 0) & (label <= 15)).all()):
        raise ValueError("AMOS artifact contains invalid label IDs")
    if not bool(torch.isfinite(probabilities).all()) or float(probabilities.min()) < 0 or float(probabilities.max()) > 1:
        raise ValueError("AMOS artifact probabilities must be finite in [0, 1]")
    # Selected foreground channels are a subset of the 14-way softmax; their
    # sum is intentionally not one. Each target is evaluated one-vs-rest.


def native_organ_volumes(label_path: Path) -> dict[str, dict[str, float | int]]:
    import nibabel as nib

    nii = nib.load(str(label_path))
    if nii.header.get_xyzt_units()[0] != "mm":
        raise ValueError("AMOS native label spatial units must be millimeters")
    voxels = np.asanyarray(nii.dataobj)
    if not np.all(np.isfinite(voxels)) or not np.all(voxels == np.rint(voxels)):
        raise ValueError("AMOS native label contains non-finite or non-integral values")
    observed = set(int(value) for value in np.unique(voxels))
    if not observed.issubset(set(range(16))):
        raise ValueError(f"Unexpected AMOS source label IDs: {sorted(observed)}")
    voxel_mm3 = float(abs(np.linalg.det(nii.affine[:3, :3])))
    if not math.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError("Invalid AMOS native voxel volume")
    counts = {organ.name: int(np.count_nonzero(voxels == organ.amos_label)) for organ in ORGANS}
    return {organ.name: {"voxels": counts[organ.name], "volume_mm3": counts[organ.name] * voxel_mm3}
            for organ in ORGANS}


def validate_native_pair(image_path: Path, label_path: Path) -> None:
    """Reject labels that do not occupy the image's physical voxel grid."""
    import nibabel as nib

    image, label = nib.load(str(image_path)), nib.load(str(label_path))
    if image.header.get_xyzt_units()[0] != "mm" or label.header.get_xyzt_units()[0] != "mm":
        raise ValueError("AMOS image and label spatial units must be millimeters")
    if image.shape != label.shape or not np.allclose(image.affine, label.affine, rtol=0, atol=1e-5):
        raise ValueError(f"AMOS native image/label grids differ: {image_path}, {label_path}")


@torch.inference_mode()
def make_cache(dataset_root: Path, bundle_parent: Path, output_dir: Path, device: torch.device,
               *, limit: int | None = None) -> Path:
    cases = resolve_cases(dataset_root)
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    selected_cases = cases if limit is None else cases[:limit]
    spec = dict(load_monai_specs()[BTCV_SPEC_ID])
    spec.pop("output_transform")  # preserve all 14 model channels until the six targets are selected
    bundle = spec["bundle"]
    if DATASET_ID in {bundle["supervised_training_dataset_id"], *bundle["self_supervised_pretraining_dataset_ids"]}:
        raise ValueError("AMOS evaluation cases must not be a declared checkpoint training source")
    bundle_root = bundle_parent / bundle["name"]
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"Missing pinned MONAI Bundle: {bundle_root}")
    api = _require_monai()
    model, config_path, checkpoint_path = _build_model(spec, bundle_root, device, api)
    preprocessing = _build_preprocessing(spec, api)
    software = _software_info(api["monai"])
    fingerprint = _hash_json({"study_revision": 1, "case_list_sha256": CASE_LIST_SHA256,
                              "targets": [vars(organ) for organ in ORGANS], "model_spec": spec,
                              "config_sha256": _hash_file(config_path),
                              "checkpoint_sha256": _hash_file(checkpoint_path),
                              "device": str(device), "software": software})
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records = []
    for index, (case_id, image_path, label_path) in enumerate(selected_cases, 1):
        validate_native_pair(image_path, label_path)
        source_hashes = {"image_sha256": _hash_file(image_path), "label_sha256": _hash_file(label_path)}
        destination = output_dir / "cases" / f"{case_id}.pt"
        if destination.exists():
            payload = torch.load(destination, map_location="cpu", weights_only=True)
            validate_payload(payload)
            metadata = payload.get("metadata", {})
            if payload.get("case_id") != case_id or metadata.get("fingerprint") != fingerprint or any(
                metadata.get(key) != value for key, value in source_hashes.items()
            ):
                raise ValueError(f"Existing AMOS cache provenance differs: {destination}")
            print(f"[{index}/{len(selected_cases)}] Reusing {case_id}", flush=True)
        else:
            print(f"[{index}/{len(selected_cases)}] Inferring {case_id}", flush=True)
            native = native_organ_volumes(label_path)
            transformed = preprocessing({"image": str(image_path), "label": str(label_path)})
            image = transformed["image"].as_tensor()
            label = transformed["label"].as_tensor().squeeze(0)
            if not bool(((label >= 0) & (label <= 15) & (label == label.round())).all()):
                raise ValueError(f"AMOS transformed label has invalid class IDs in {case_id}")
            label = label.to(dtype=torch.uint8).cpu().contiguous()
            full_probs, inference_ms = _infer_case(model, image, spec, device, api)
            if full_probs.shape[0] != 14:
                raise ValueError("BTCV Bundle did not produce 14 softmax channels")
            probabilities = full_probs[[organ.model_channel for organ in ORGANS]].contiguous()
            del full_probs, image, transformed
            for organ in ORGANS:
                if native[organ.name]["voxels"] and not bool((label == organ.amos_label).any()):
                    raise ValueError(f"Resampling erased the {organ.name} mask in {case_id}")
            payload = {"schema_version": 1, "case_id": case_id, "probabilities": probabilities,
                       "label": label, "metadata": {"fingerprint": fingerprint, **source_hashes,
                       "native_organ_volumes": native, "inference_ms": inference_ms,
                       "coordinate_space": "resampled_model_space"}}
            validate_payload(payload)
            _atomic_torch_save(payload, destination)
        records.append({"case_id": case_id, "file": str(destination.relative_to(output_dir)),
                        "sha256": _hash_file(destination), "spatial_shape": list(payload["label"].shape)})
    manifest = {"schema_version": 1, "study": DATASET_ID, "complete": len(records) == EXPECTED_CASES,
                "expected_cases": EXPECTED_CASES, "case_list_sha256": CASE_LIST_SHA256,
                "dataset_source": {"official": "https://zenodo.org/records/7155725",
                                   "download_mirror": MIRROR_REPO, "mirror_revision": MIRROR_REVISION,
                                   "split": "labeled CT validation", "checkpoint_exposure": "none_by_declared_dataset_provenance",
                                   "used_for_checkpoint_training": False,
                                   "used_for_decoder_parameter_selection": False,
                                   "patient_overlap_audited": False},
                "bundle": {"name": bundle["name"], "version": bundle["version"],
                           "config_sha256": _hash_file(config_path), "checkpoint_sha256": _hash_file(checkpoint_path),
                           "training_dataset": "BTCV"},
                "targets": [vars(organ) for organ in ORGANS], "generation_fingerprint": fingerprint,
                "software": software, "cases": records}
    path = output_dir / "manifest.json"
    _atomic_json_save(manifest, path)
    return path


def _scores(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float | None, float | None, int, int, int]:
    gt = int(truth.sum())
    positive = int(pred.sum())
    tp = int((pred & truth).sum())
    if gt == 0:
        return None, None, tp, positive, gt
    return 2 * tp / (positive + gt), tp / (positive + gt - tp), tp, positive, gt


def _measure(call, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    del result
    return elapsed_ms


def summarize(rows: list[dict]) -> dict:
    """Foreground-positive cases only; absent targets are separately counted."""
    summary = {}
    for organ in ORGANS:
        selected = [row for row in rows if row["organ"] == organ.name]
        positive = [row for row in selected if row["gt_model_voxels"] > 0]
        absent = [row for row in selected if row["gt_model_voxels"] == 0]
        if not positive:
            raise ValueError(f"No ground-truth-positive cases for {organ.name}")
        baseline = statistics.mean(row["argmax_dice"] for row in positive)
        rankseg = statistics.mean(row["rankseg_dice"] for row in positive)
        summary[organ.name] = {
            "size_group": organ.size_group, "cases": len(selected), "positive_cases": len(positive),
            "absent_cases": len(selected) - len(positive),
            "absent_argmax_predicted_voxels": sum(row["argmax_voxels"] for row in absent),
            "absent_rankseg_predicted_voxels": sum(row["rankseg_voxels"] for row in absent),
            "absent_argmax_false_positive_cases": sum(row["argmax_voxels"] > 0 for row in absent),
            "absent_rankseg_false_positive_cases": sum(row["rankseg_voxels"] > 0 for row in absent),
            "argmax_dice": baseline, "rankseg_dice": rankseg, "delta_dice_pp": 100 * (rankseg - baseline),
            "argmax_iou": statistics.mean(row["argmax_iou"] for row in positive),
            "rankseg_iou": statistics.mean(row["rankseg_iou"] for row in positive),
            "mean_native_volume_mm3": statistics.mean(row["gt_native_mm3"] for row in positive),
            "mean_argmax_ms": statistics.mean(row["argmax_ms"] for row in selected),
            "mean_rankseg_ms": statistics.mean(row["rankseg_ms"] for row in selected),
            "wins": sum(row["rankseg_dice"] > row["argmax_dice"] for row in positive),
            "losses": sum(row["rankseg_dice"] < row["argmax_dice"] for row in positive),
            "ties": sum(row["rankseg_dice"] == row["argmax_dice"] for row in positive),
            "resampling_erased_cases": sum(row["gt_native_voxels"] > 0 and row["gt_model_voxels"] == 0 for row in selected),
        }
    return summary


def size_baseline_strata(rows: list[dict]) -> dict:
    """Descriptive within-organ 2×2 split; never selects cases by RankSEG outcome."""
    result = {}
    for organ in ORGANS:
        positive = [row for row in rows if row["organ"] == organ.name and row["gt_model_voxels"] > 0]
        if not positive:
            raise ValueError(f"No ground-truth-positive cases for {organ.name}")
        size_median = statistics.median(row["gt_native_mm3"] for row in positive)
        baseline_median = statistics.median(row["argmax_dice"] for row in positive)
        cells = {}
        for size_name, size_test in (("smaller", lambda value: value <= size_median),
                                     ("larger", lambda value: value > size_median)):
            for baseline_name, baseline_test in (("lower_baseline", lambda value: value <= baseline_median),
                                                 ("higher_baseline", lambda value: value > baseline_median)):
                subset = [row for row in positive if size_test(row["gt_native_mm3"])
                          and baseline_test(row["argmax_dice"])]
                cells[f"{size_name}__{baseline_name}"] = {
                    "cases": len(subset),
                    "mean_delta_dice_pp": statistics.mean(
                        100 * (row["rankseg_dice"] - row["argmax_dice"]) for row in subset
                    ) if subset else None,
                }
        result[organ.name] = {"native_volume_median_mm3": size_median,
                              "argmax_dice_median": baseline_median, "cells": cells}
    return result


def shared_positive_summary(rows: list[dict]) -> dict:
    """Secondary paired view restricted by GT only, never by predictions."""
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    complete_ids = {case_id for case_id, case_rows in by_case.items()
                    if len(case_rows) == len(ORGANS) and all(row["gt_model_voxels"] > 0 for row in case_rows)}
    if not complete_ids:
        return {"cases": 0, "organs": {}}
    summary = {}
    for organ in ORGANS:
        selected = [row for row in rows if row["organ"] == organ.name and row["case_id"] in complete_ids]
        baseline = statistics.mean(row["argmax_dice"] for row in selected)
        rankseg = statistics.mean(row["rankseg_dice"] for row in selected)
        summary[organ.name] = {"argmax_dice": baseline, "rankseg_dice": rankseg,
                               "delta_dice_pp": 100 * (rankseg - baseline)}
    return {"cases": len(complete_ids), "organs": summary}


@torch.inference_mode()
def evaluate(cache_dir: Path, device: torch.device, *, warmup: int = 2, repeats: int = 9,
             allow_partial: bool = False) -> dict:
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must both be >= 1")
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("study") != DATASET_ID or manifest.get("case_list_sha256") != CASE_LIST_SHA256:
        raise ValueError("Not a pinned AMOS22 CT validation cache")
    if manifest.get("expected_cases") != EXPECTED_CASES or (
        manifest.get("complete") and len(manifest.get("cases", [])) != EXPECTED_CASES
    ):
        raise ValueError("AMOS cache does not contain its declared complete cohort")
    if not manifest.get("complete") and not allow_partial:
        raise ValueError("Partial AMOS cache cannot produce a cohort result")
    if manifest.get("targets") != [vars(organ) for organ in ORGANS]:
        raise ValueError("AMOS cache target mapping differs from this study")
    case_ids = [entry["case_id"] for entry in manifest["cases"]]
    if case_ids != sorted(set(case_ids)) or (
        manifest["complete"] and hashlib.sha256("\n".join(case_ids).encode()).hexdigest() != CASE_LIST_SHA256
    ):
        raise ValueError("AMOS cache case list is duplicated, unordered, or differs from the fixed validation split")
    from rankseg import RankSEG
    import rankseg

    decoder = RankSEG(metric="dice", solver="RMA", output_mode="multilabel", safe_screening=True)
    rows = []
    for index, entry in enumerate(manifest["cases"], 1):
        path = cache_dir / entry["file"]
        if _hash_file(path) != entry["sha256"]:
            raise ValueError(f"AMOS case checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_payload(payload)
        if payload.get("case_id") != entry["case_id"] or payload.get("metadata", {}).get("fingerprint") != manifest["generation_fingerprint"]:
            raise ValueError(f"AMOS case provenance mismatch: {path}")
        probs = payload["probabilities"].to(device)
        label = payload["label"].to(device)
        for organ_index, organ in enumerate(ORGANS):
            p = probs[organ_index].unsqueeze(0).unsqueeze(0)
            binary = torch.cat((1 - p, p), dim=1)

            def argmax_call():
                return binary.argmax(dim=1)

            def rankseg_call():
                return decoder.predict(p)[:, 0]

            argmax = argmax_call().bool()
            rankseg_mask = rankseg_call().bool()
            truth = label.unsqueeze(0) == organ.amos_label
            arg_dice, arg_iou, arg_tp, arg_size, gt_size = _scores(argmax, truth)
            rank_dice, rank_iou, rank_tp, rank_size, _ = _scores(rankseg_mask, truth)
            for _ in range(warmup):
                _measure(argmax_call, device)
                _measure(rankseg_call, device)
            timings = {"argmax": [], "rankseg": []}
            for repetition in range(repeats):
                order = (("argmax", argmax_call), ("rankseg", rankseg_call))
                if (index + organ_index + repetition) % 2:
                    order = order[::-1]
                for method, call in order:
                    timings[method].append(_measure(call, device))
            native = payload["metadata"]["native_organ_volumes"][organ.name]
            if native["voxels"] > 0 and gt_size == 0:
                raise ValueError(f"Resampling erased {organ.name} in {entry['case_id']}")
            rows.append({"case_id": entry["case_id"], "organ": organ.name,
                         "size_group": organ.size_group, "gt_native_voxels": native["voxels"],
                         "gt_native_mm3": native["volume_mm3"], "gt_model_voxels": gt_size,
                         "argmax_voxels": arg_size, "rankseg_voxels": rank_size,
                         "argmax_tp": arg_tp, "rankseg_tp": rank_tp,
                         "argmax_dice": arg_dice, "rankseg_dice": rank_dice,
                         "argmax_iou": arg_iou, "rankseg_iou": rank_iou,
                         "max_target_probability": float(p.max()),
                         "expected_target_voxels": float(p.sum()),
                         "changed_voxels": int((argmax != rankseg_mask).sum()),
                         "error_corrected_voxels": int(((argmax != truth) & (rankseg_mask == truth)).sum()),
                         "error_introduced_voxels": int(((argmax == truth) & (rankseg_mask != truth)).sum()),
                         "argmax_ms": statistics.median(timings["argmax"]),
                         "rankseg_ms": statistics.median(timings["rankseg"])})
        print(f"[{index}/{len(manifest['cases'])}] {entry['case_id']}", flush=True)
    return {"study": DATASET_ID, "complete_cohort": bool(manifest["complete"]),
            "cases": len(manifest["cases"]), "rankseg_version": version("rankseg"),
            "rankseg_module": str(Path(rankseg.__file__).resolve()),
            "device": str(device), "warmup": warmup, "repeats": repeats,
            "method": "one-vs-rest per organ; 14-way MONAI softmax channel versus its complement",
            "summary": summarize(rows), "size_baseline_strata": size_baseline_strata(rows),
            "shared_positive": shared_positive_summary(rows),
            "case_organ_results": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache", help="Infer the pinned AMOS22 CT validation cases once")
    cache.add_argument("--dataset-root", type=Path, required=True)
    cache.add_argument("--bundle-dir", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--limit", type=int, help="Smoke testing only; never a complete cohort")
    eval_parser = sub.add_parser("evaluate", help="Compare Argmax and RMA Dice + screening")
    eval_parser.add_argument("--cache-dir", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--device", default="cuda")
    eval_parser.add_argument("--warmup", type=int, default=2)
    eval_parser.add_argument("--repeats", type=int, default=9)
    eval_parser.add_argument("--allow-partial", action="store_true", help="Smoke testing only")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.command == "cache":
        print(make_cache(args.dataset_root, args.bundle_dir, args.output_dir, device, limit=args.limit))
    else:
        result = evaluate(args.cache_dir, device, warmup=args.warmup, repeats=args.repeats,
                          allow_partial=args.allow_partial)
        _atomic_json_save(result, args.output)
        print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
