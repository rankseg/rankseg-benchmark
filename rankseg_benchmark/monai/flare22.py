"""Frozen MONAI BTCV checkpoint on the 50 public labeled FLARE22 training cases.

These cases are an external evaluation cohort for BTCV, not the FLARE22
challenge's hidden validation/test set. Four images overlap a previously
evaluated MSD Pancreas cohort, so the predeclared primary analysis uses the
other 46 cases; all 50 are also reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
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
from rankseg_benchmark.monai.small_organs import (
    BTCV_SPEC_ID,
    ORGANS,
    _measure,
    _scores,
    shared_positive_summary,
    size_baseline_strata,
    summarize,
    validate_native_pair,
    validate_payload,
)

DATASET_ID = "flare22_public_labeled50_external_to_btcv"
EXPECTED_CASES = 50
CASE_LIST_SHA256 = "b571037423bb9f5680871ea5005f090c31d9738d59a4c4aa8adb0e42fd1b032e"
MIRROR_REPO = "MedOtter/FLARE22"
MIRROR_REVISION = "ab0b99b53e2183fe59321b5888867c6c7cb0792a"
OFFICIAL_ARCHIVE_MD5 = "f6b1a257deb8ad98eb901328b0762cc4"

# Official FLARE22 labels differ from BTCV softmax channels. The Organ
# definitions imported above fix the BTCV channels and the shared endpoints.
SOURCE_LABELS = {
    "left_adrenal": 8,
    "right_adrenal": 7,
    "gallbladder": 9,
    "pancreas": 4,
    "spleen": 3,
    "liver": 1,
}

# Determined solely by SHA-256 of source CT bytes, before inference: these
# match the existing 20-case MSD Task07 BTCV cache. They are still included
# in the all-50 secondary analysis and in every case-level output.
PRIOR_MSD_OVERLAP = {
    "FLARE22_Tr_0005": "pancreas_016",
    "FLARE22_Tr_0011": "pancreas_056",
    "FLARE22_Tr_0020": "pancreas_102",
    "FLARE22_Tr_0023": "pancreas_107",
}
PRIOR_MSD_IMAGE_SHA256 = {
    "FLARE22_Tr_0005": "2be5019bfc7e805d5e24785bcd44ffe7720e13e38b2a3124ad25b454811b221c",
    "FLARE22_Tr_0011": "1e64295f3b0e2b4263cf0e32c3313fd7c46814777123ea0d7fff3ec10eca02d7",
    "FLARE22_Tr_0020": "a57b50ac7d4a7053c46a801de99a17607e61d35970d1f2dc17738008690a395b",
    "FLARE22_Tr_0023": "b6be578558284074ea945da67e008b278aacc33844e17e1610fa39f5cc168c1f",
}


def target_mapping() -> list[dict]:
    return [{"name": organ.name, "source_label": SOURCE_LABELS[organ.name],
             "model_channel": organ.model_channel, "size_group": organ.size_group}
            for organ in ORGANS]


def resolve_cases(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    """Select all 50 official labeled cases, regardless of model predictions."""
    root = dataset_root / "FLARE22Train" if (dataset_root / "FLARE22Train").is_dir() else dataset_root
    images = {path.name.removesuffix("_0000.nii.gz"): path
              for path in (root / "images").glob("FLARE22_Tr_*_0000.nii.gz")}
    labels = {path.name.removesuffix(".nii.gz"): path
              for path in (root / "labels").glob("FLARE22_Tr_*.nii.gz")}
    if images.keys() != labels.keys():
        raise ValueError(f"FLARE22 image/label mismatch: images-only={sorted(images.keys()-labels.keys())}, "
                         f"labels-only={sorted(labels.keys()-images.keys())}")
    case_ids = sorted(images)
    if len(case_ids) != EXPECTED_CASES or hashlib.sha256("\n".join(case_ids).encode()).hexdigest() != CASE_LIST_SHA256:
        raise ValueError("FLARE22 labeled 50-case list differs from the pinned official release")
    return [(case_id, images[case_id], labels[case_id]) for case_id in case_ids]


def native_organ_volumes(label_path: Path) -> dict[str, dict[str, float | int]]:
    import nibabel as nib

    nii = nib.load(str(label_path))
    if nii.header.get_xyzt_units()[0] != "mm":
        raise ValueError("FLARE22 label spatial units must be millimeters")
    voxels = np.asanyarray(nii.dataobj)
    if not np.all(np.isfinite(voxels)) or not np.all(voxels == np.rint(voxels)):
        raise ValueError("FLARE22 native label contains non-finite or non-integral values")
    observed = set(int(value) for value in np.unique(voxels))
    if not observed.issubset(set(range(14))):
        raise ValueError(f"Unexpected FLARE22 label IDs: {sorted(observed)}")
    voxel_mm3 = float(abs(np.linalg.det(nii.affine[:3, :3])))
    if not math.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError("Invalid FLARE22 native voxel volume")
    counts = {organ.name: int(np.count_nonzero(voxels == SOURCE_LABELS[organ.name]))
              for organ in ORGANS}
    return {organ.name: {"voxels": counts[organ.name],
                         "volume_mm3": counts[organ.name] * voxel_mm3}
            for organ in ORGANS}


@torch.inference_mode()
def make_cache(dataset_root: Path, bundle_parent: Path, output_dir: Path, device: torch.device,
               *, limit: int | None = None) -> Path:
    cases = resolve_cases(dataset_root)
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    selected = cases if limit is None else cases[:limit]
    spec = dict(load_monai_specs()[BTCV_SPEC_ID])
    spec.pop("output_transform")
    bundle = spec["bundle"]
    if "flare22" in {bundle["supervised_training_dataset_id"], *bundle["self_supervised_pretraining_dataset_ids"]}:
        raise ValueError("FLARE22 cannot be used to evaluate a checkpoint trained on FLARE22")
    bundle_root = bundle_parent / bundle["name"]
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"Missing pinned MONAI Bundle: {bundle_root}")
    api = _require_monai()
    model, config_path, checkpoint_path = _build_model(spec, bundle_root, device, api)
    preprocessing = _build_preprocessing(spec, api)
    software = _software_info(api["monai"])
    fingerprint = _hash_json({"study_revision": 1, "study": DATASET_ID,
                              "case_list_sha256": CASE_LIST_SHA256, "targets": target_mapping(),
                              "model_spec": spec, "config_sha256": _hash_file(config_path),
                              "checkpoint_sha256": _hash_file(checkpoint_path),
                              "device": str(device), "software": software})
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records = []
    for index, (case_id, image_path, label_path) in enumerate(selected, 1):
        validate_native_pair(image_path, label_path)
        source_hashes = {"image_sha256": _hash_file(image_path), "label_sha256": _hash_file(label_path)}
        if case_id in PRIOR_MSD_IMAGE_SHA256 and source_hashes["image_sha256"] != PRIOR_MSD_IMAGE_SHA256[case_id]:
            raise ValueError(f"Predeclared MSD overlap hash changed for {case_id}")
        destination = output_dir / "cases" / f"{case_id}.pt"
        if destination.exists():
            payload = torch.load(destination, map_location="cpu", weights_only=True)
            validate_payload(payload)
            metadata = payload.get("metadata", {})
            if payload.get("case_id") != case_id or metadata.get("fingerprint") != fingerprint or any(
                metadata.get(key) != value for key, value in source_hashes.items()
            ):
                raise ValueError(f"Existing FLARE22 cache provenance differs: {destination}")
            print(f"[{index}/{len(selected)}] Reusing {case_id}", flush=True)
        else:
            print(f"[{index}/{len(selected)}] Inferring {case_id}", flush=True)
            native = native_organ_volumes(label_path)
            transformed = preprocessing({"image": str(image_path), "label": str(label_path)})
            image = transformed["image"].as_tensor()
            label = transformed["label"].as_tensor().squeeze(0)
            if not bool(((label >= 0) & (label <= 13) & (label == label.round())).all()):
                raise ValueError(f"FLARE22 transformed label has invalid class IDs in {case_id}")
            label = label.to(dtype=torch.uint8).cpu().contiguous()
            full_probs, inference_ms = _infer_case(model, image, spec, device, api)
            if full_probs.shape[0] != 14:
                raise ValueError("BTCV Bundle did not produce 14 softmax channels")
            probabilities = full_probs[[organ.model_channel for organ in ORGANS]].contiguous()
            del full_probs, image, transformed
            for organ in ORGANS:
                if native[organ.name]["voxels"] and not bool((label == SOURCE_LABELS[organ.name]).any()):
                    raise ValueError(f"Resampling erased {organ.name} in {case_id}")
            payload = {"schema_version": 1, "case_id": case_id, "probabilities": probabilities,
                       "label": label, "metadata": {"fingerprint": fingerprint, **source_hashes,
                       "native_organ_volumes": native, "inference_ms": inference_ms,
                       "coordinate_space": "resampled_model_space"}}
            validate_payload(payload)
            _atomic_torch_save(payload, destination)
        records.append({"case_id": case_id, "file": str(destination.relative_to(output_dir)),
                        "sha256": _hash_file(destination), "spatial_shape": list(payload["label"].shape),
                        "prior_msd_overlap": case_id in PRIOR_MSD_OVERLAP})
    manifest = {"schema_version": 1, "study": DATASET_ID, "complete": len(records) == EXPECTED_CASES,
                "expected_cases": EXPECTED_CASES, "case_list_sha256": CASE_LIST_SHA256,
                "dataset_source": {"official": "https://zenodo.org/records/7860267",
                                   "official_archive_md5": OFFICIAL_ARCHIVE_MD5,
                                   "download_mirror": MIRROR_REPO, "mirror_revision": MIRROR_REVISION,
                                   "split": "public labeled training cases; external to BTCV checkpoint",
                                   "patient_overlap_audited": False},
                "prior_msd_overlap": PRIOR_MSD_OVERLAP,
                "bundle": {"name": bundle["name"], "version": bundle["version"],
                           "config_sha256": _hash_file(config_path), "checkpoint_sha256": _hash_file(checkpoint_path),
                           "training_dataset": "BTCV"},
                "targets": target_mapping(), "generation_fingerprint": fingerprint,
                "software": software, "cases": records}
    path = output_dir / "manifest.json"
    _atomic_json_save(manifest, path)
    return path


@torch.inference_mode()
def evaluate(cache_dir: Path, device: torch.device, *, warmup: int = 2, repeats: int = 9,
             allow_partial: bool = False) -> dict:
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must both be >= 1")
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("study") != DATASET_ID or manifest.get("case_list_sha256") != CASE_LIST_SHA256:
        raise ValueError("Not a pinned FLARE22 labeled-50 cache")
    if manifest.get("expected_cases") != EXPECTED_CASES or (
        manifest.get("complete") and len(manifest.get("cases", [])) != EXPECTED_CASES
    ):
        raise ValueError("FLARE22 cache does not contain its declared complete cohort")
    if not manifest.get("complete") and not allow_partial:
        raise ValueError("Partial FLARE22 cache cannot produce a cohort result")
    if manifest.get("targets") != target_mapping() or manifest.get("prior_msd_overlap") != PRIOR_MSD_OVERLAP:
        raise ValueError("FLARE22 target mapping or predeclared overlap differs from this study")
    case_ids = [entry["case_id"] for entry in manifest["cases"]]
    if case_ids != sorted(set(case_ids)) or (
        manifest["complete"] and hashlib.sha256("\n".join(case_ids).encode()).hexdigest() != CASE_LIST_SHA256
    ):
        raise ValueError("FLARE22 cache case list is duplicated, unordered, or differs from the fixed release")
    from rankseg import RankSEG
    import rankseg

    decoder = RankSEG(metric="dice", solver="RMA", output_mode="multilabel", safe_screening=True)
    rows = []
    for index, entry in enumerate(manifest["cases"], 1):
        path = cache_dir / entry["file"]
        if _hash_file(path) != entry["sha256"]:
            raise ValueError(f"FLARE22 case checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_payload(payload)
        if payload.get("case_id") != entry["case_id"] or payload.get("metadata", {}).get("fingerprint") != manifest["generation_fingerprint"]:
            raise ValueError(f"FLARE22 case provenance mismatch: {path}")
        if entry["prior_msd_overlap"] != (entry["case_id"] in PRIOR_MSD_OVERLAP):
            raise ValueError(f"FLARE22 predeclared overlap mismatch: {entry['case_id']}")
        probs, label = payload["probabilities"].to(device), payload["label"].to(device)
        for organ_index, organ in enumerate(ORGANS):
            p = probs[organ_index].unsqueeze(0).unsqueeze(0)
            binary = torch.cat((1 - p, p), dim=1)

            def argmax_call():
                return binary.argmax(dim=1)

            def rankseg_call():
                return decoder.predict(p)[:, 0]

            argmax, rankseg_mask = argmax_call().bool(), rankseg_call().bool()
            truth = label.unsqueeze(0) == SOURCE_LABELS[organ.name]
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
                         "size_group": organ.size_group, "prior_msd_overlap": entry["prior_msd_overlap"],
                         "gt_native_voxels": native["voxels"], "gt_native_mm3": native["volume_mm3"],
                         "gt_model_voxels": gt_size, "argmax_voxels": arg_size,
                         "rankseg_voxels": rank_size, "argmax_tp": arg_tp, "rankseg_tp": rank_tp,
                         "argmax_dice": arg_dice, "rankseg_dice": rank_dice,
                         "argmax_iou": arg_iou, "rankseg_iou": rank_iou,
                         "changed_voxels": int((argmax != rankseg_mask).sum()),
                         "error_corrected_voxels": int(((argmax != truth) & (rankseg_mask == truth)).sum()),
                         "error_introduced_voxels": int(((argmax == truth) & (rankseg_mask != truth)).sum()),
                         "argmax_ms": statistics.median(timings["argmax"]),
                         "rankseg_ms": statistics.median(timings["rankseg"])})
        print(f"[{index}/{len(manifest['cases'])}] {entry['case_id']}", flush=True)
    primary = [row for row in rows if not row["prior_msd_overlap"]]
    primary_cases = len({row["case_id"] for row in primary})
    if manifest["complete"] and primary_cases != EXPECTED_CASES - len(PRIOR_MSD_OVERLAP):
        raise ValueError("FLARE22 non-overlap primary cohort is incomplete")
    return {"study": DATASET_ID, "complete_cohort": bool(manifest["complete"]),
            "cases": len(manifest["cases"]), "primary_cases": primary_cases,
            "rankseg_version": version("rankseg"), "rankseg_module": str(Path(rankseg.__file__).resolve()),
            "device": str(device), "warmup": warmup, "repeats": repeats,
            "method": "one-vs-rest per organ; MONAI BTCV softmax channel versus its complement",
            "primary_summary": summarize(primary), "all50_summary": summarize(rows),
            "primary_size_baseline_strata": size_baseline_strata(primary),
            "primary_shared_positive": shared_positive_summary(primary),
            "case_organ_results": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache", help="Infer pinned FLARE22 labeled cases once")
    cache.add_argument("--dataset-root", type=Path, required=True)
    cache.add_argument("--bundle-dir", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--limit", type=int, help="Smoke testing only")
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
        print(json.dumps(result["primary_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
