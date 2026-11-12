"""Paired TotalSegmentator lung_vessels / RankSEG pilot on public HiPaS cases.

HiPaS provides CT arrays without original affines. At least case 001 contains
raw HU despite the repository README saying the archives are normalized. The
conversion below checks each case's value range and uses the per-case spacing
from metadata.xlsx. Consequently this is an exploratory external-data pilot,
not a native PARSE2022 validation or an official HiPaS benchmark result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import pickle
import shutil
import time
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
import torch


LABELS = {"artery": 3, "vein": 4}
TARGET_ID = 117
COARSE_ID = 297

def load_spacing(metadata_path: Path) -> dict[str, tuple[float, float, float]]:
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(metadata_path) as archive:
        strings_xml = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(cell.itertext()) for cell in strings_xml.findall("m:si", namespace)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    result = {}
    for row in sheet.findall(".//m:sheetData/m:row", namespace)[1:]:
        cells = {}
        for cell in row.findall("m:c", namespace):
            value = cell.find("m:v", namespace)
            if value is None:
                continue
            cells[cell.attrib["r"].rstrip("0123456789")] = (
                strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text)
        if "A" in cells and "B" in cells:
            spacing = tuple(float(x) for x in ast.literal_eval(cells["B"].strip()))
            if len(spacing) != 3 or min(spacing) <= 0:
                raise RuntimeError(f"Invalid HiPaS spacing: {cells}")
            result[Path(cells["A"]).stem] = spacing
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred_n = int(np.count_nonzero(pred))
    truth_n = int(np.count_nonzero(truth))
    tp = int(np.count_nonzero(pred & truth))
    dice = 1.0 if pred_n + truth_n == 0 else 2 * tp / (pred_n + truth_n)
    return {"dice": dice, "true_positive": tp, "predicted": pred_n,
            "reference": truth_n, "false_positive": pred_n - tp,
            "false_negative": truth_n - tp}


def convert(case_id: str, ct_dir: Path, annotation_dir: Path, case_dir: Path,
            spacing: tuple[float, float, float]) -> tuple[Path, np.ndarray, dict]:
    source = ct_dir / f"{case_id}.npz"
    arteries = annotation_dir / "artery" / f"{case_id}.npz"
    veins = annotation_dir / "vein" / f"{case_id}.npz"
    for path in (source, arteries, veins):
        if not path.is_file():
            raise FileNotFoundError(path)
    image_path = case_dir / "ct_hu_reconstructed.nii.gz"
    checksums = {p.stem + "_" + p.parent.name: sha256(p) for p in (source, arteries, veins)}
    with np.load(source) as archive:
        ct = archive["data"]
    if ct.ndim != 3:
        raise RuntimeError(f"Unexpected HiPaS CT shape: {ct.shape}")
    ct_range = [float(ct.min()), float(ct.max())]
    if ct_range[0] >= -1e-5 and ct_range[1] <= 1.00001:
        ct_units = "normalized_-1000_to_600_HU"
    elif ct_range[0] >= -2000 and ct_range[1] <= 5000:
        ct_units = "raw_HU"
    else:
        raise RuntimeError(f"Unexpected HiPaS CT range: {ct.shape}, {ct_range}")
    affine = np.diag([*spacing, 1.0])
    if not image_path.is_file():
        hu = np.asarray(ct, dtype=np.float32)
        if ct_units != "raw_HU":
            hu = hu * 1600.0 - 1000.0
        nib.save(nib.Nifti1Image(hu, affine), image_path)
        del hu
    elif not np.allclose(nib.load(image_path).affine, affine, atol=1e-5):
        raise RuntimeError("Existing converted CT has different spacing")
    shape = ct.shape
    del ct
    label_info = {}
    for kind, path in (("artery", arteries), ("vein", veins)):
        with np.load(path) as archive:
            label = archive["data"]
        if label.shape != shape or not np.isin(np.unique(label), [0, 1]).all():
            raise RuntimeError(f"Unexpected {kind} annotation: {label.shape}, {np.unique(label)}")
        label_info[kind] = int(np.count_nonzero(label))
        del label
    return image_path, affine, {"source_sha256": checksums, "shape": list(shape),
                                "spacing_mm": list(spacing),
                                "ct_units": ct_units, "ct_range": ct_range,
                                "annotation_voxels": label_info}


def evaluate_case(case_id: str, ct_dir: Path, annotation_dir: Path,
                  output_root: Path, device: str,
                  spacing: tuple[float, float, float]) -> dict:
    from totalsegmentator.config import get_weights_dir, setup_nnunet
    setup_nnunet()
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from rankseg import RankSEG
    from totalsegmentator import nnunet as ts_nnunet
    from totalsegmentator.python_api import totalsegmentator
    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path, affine, source_info = convert(case_id, ct_dir, annotation_dir,
                                              case_dir, spacing)
    native_path = case_dir / "argmax.nii.gz"
    rankseg_path = case_dir / "rankseg.nii.gz"
    probabilities_path = case_dir / "probabilities.npz"
    coarse_raw = case_dir / "coarse_raw.nii.gz"
    target_raw = case_dir / "target_argmax_raw.nii.gz"
    rankseg_raw = case_dir / "target_rankseg_raw.nii.gz"
    original_predict = ts_nnunet.nnUNetv2_predict
    mode = "infer"
    seen: list[int] = []

    def intercept(dir_in, dir_out, task_id, *args, **kwargs):
        tid = int(task_id)
        if tid not in (COARSE_ID, TARGET_ID):
            raise RuntimeError(f"Unexpected TotalSegmentator model: {tid}")
        out_file = Path(dir_out) / "s01.nii.gz"
        seen.append(tid)
        if mode == "infer":
            original_predict(dir_in, dir_out, task_id, *args, **kwargs)
            shutil.copy2(out_file, coarse_raw if tid == COARSE_ID else target_raw)
        else:
            shutil.copy2(coarse_raw if tid == COARSE_ID else rankseg_raw, out_file)

    ts_nnunet.nnUNetv2_predict = intercept
    try:
        started = time.perf_counter()
        baseline = totalsegmentator(image_path, native_path, task="lung_vessels",
                                    ml=True, device=device, quiet=True,
                                    save_probabilities=probabilities_path)
        infer_seconds = time.perf_counter() - started
        if seen != [COARSE_ID, TARGET_ID]:
            raise RuntimeError(f"Unexpected inference stages: {seen}")
        seen.clear()
        target_model = (get_weights_dir() / "Dataset117_lung_airways_arteries_veins_282subj"
                        / "nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres")
        with (target_model / "plans.json").open() as stream:
            plans = PlansManager(json.load(stream))
        reader_writer = plans.image_reader_writer_class()
        with probabilities_path.with_suffix(".pkl").open("rb") as stream:
            properties = pickle.load(stream)
        with np.load(probabilities_path) as archive:
            probabilities = archive["probabilities"].astype(np.float32)
        native_raw, _ = reader_writer.read_seg(str(target_raw))
        if probabilities.shape[0] != 5 or not np.array_equal(probabilities.argmax(axis=0), native_raw[0]):
            raise RuntimeError("Exported probabilities do not reproduce the native raw argmax mask")
        if not np.isfinite(probabilities).all() or np.min(probabilities) < -1e-5 or np.max(probabilities) > 1.00001:
            raise RuntimeError("Exported model output is not valid probabilities")
        probability_sum_range = [float(probabilities.sum(axis=0).min()),
                                 float(probabilities.sum(axis=0).max())]
        predictor = RankSEG(metric="dice", solver="RMA", output_mode="multiclass",
                            pruning_prob=0.5, safe_screening=True)
        tensor = torch.from_numpy(probabilities).unsqueeze(0).to(
            "cuda" if device == "gpu" else "cpu")
        if tensor.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        decoded = predictor(tensor)[0].cpu().numpy().astype(np.uint8)
        if tensor.is_cuda:
            torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - started
        if decoded.shape != native_raw[0].shape:
            raise RuntimeError("RankSEG target mask shape changed")
        reader_writer.write_seg(decoded, str(rankseg_raw), properties)
        del tensor, probabilities, decoded, native_raw
        mode = "replay"
        started = time.perf_counter()
        alternative = totalsegmentator(image_path, rankseg_path, task="lung_vessels",
                                       ml=True, device=device, quiet=True)
        replay_seconds = time.perf_counter() - started
        if seen != [COARSE_ID, TARGET_ID]:
            raise RuntimeError(f"Unexpected replay stages: {seen}")
    finally:
        ts_nnunet.nnUNetv2_predict = original_predict

    if baseline.shape != tuple(source_info["shape"]) or alternative.shape != baseline.shape:
        raise RuntimeError("Prediction differs from source array grid")
    if not np.allclose(baseline.affine, affine, atol=1e-4) or not np.allclose(alternative.affine, affine, atol=1e-4):
        raise RuntimeError("Prediction affine differs from reconstructed source grid")
    native = np.asanyarray(baseline.dataobj)
    decoded = np.asanyarray(alternative.dataobj)
    scores = {}
    for kind, label_value in LABELS.items():
        with np.load(annotation_dir / kind / f"{case_id}.npz") as archive:
            truth = archive["data"].astype(bool)
        scores[kind] = {"argmax": score(native == label_value, truth),
                        "rankseg": score(decoded == label_value, truth)}
        scores[kind]["delta_dice"] = (scores[kind]["rankseg"]["dice"]
                                       - scores[kind]["argmax"]["dice"])
    macro_argmax = float(np.mean([s["argmax"]["dice"] for s in scores.values()]))
    macro_rankseg = float(np.mean([s["rankseg"]["dice"] for s in scores.values()]))
    return {"case_id": case_id, **source_info, "probabilities_sha256": sha256(probabilities_path),
            "probability_sum_range": probability_sum_range, "scores": scores,
            "macro_dice": {"argmax": macro_argmax, "rankseg": macro_rankseg,
                           "delta": macro_rankseg - macro_argmax},
            "seconds": {"full_native_inference": infer_seconds,
                        "rankseg_decoder": decode_seconds, "pipeline_replay": replay_seconds}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", default=["001", "002", "003", "004", "005"])
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    spacing_by_case = load_spacing(args.metadata)
    from totalsegmentator.config import set_config_key, setup_totalseg
    setup_totalseg()
    set_config_key("send_usage_stats", False)
    set_config_key("statistics_disclaimer_shown", True)
    version = importlib.metadata.version("rankseg")
    if version != "0.0.7":
        raise RuntimeError(f"Expected released RankSEG 0.0.7, found {version}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        old_ids = results["requested_case_ids"]
        if args.case_ids[:len(old_ids)] != old_ids:
            raise RuntimeError("Existing result file uses an incompatible case prefix")
        results["requested_case_ids"] = args.case_ids
    else:
        results = {"cohort": "HiPaS public examples; per-case spacing, unknown orientation",
                   "task": "TotalSegmentator lung_vessels Dataset117 with Dataset297 crop",
                   "rankseg_version": version, "rankseg_settings": {"metric": "dice", "solver": "RMA",
                   "output_mode": "multiclass", "pruning_prob": 0.5,
                   "safe_screening": True},
                   "requested_case_ids": args.case_ids, "cases": []}
    results["rankseg_settings"]["pruning_prob"] = 0.5
    completed = {c["case_id"] for c in results["cases"]}
    for case_id in args.case_ids:
        if case_id in completed:
            continue
        result = evaluate_case(case_id, args.ct_dir, args.annotation_dir,
                               args.output_root, args.device, spacing_by_case[case_id])
        results["cases"].append(result)
        temporary = results_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(results_path)
        print(case_id, result["macro_dice"], flush=True)


if __name__ == "__main__":
    main()
