"""Evaluate frozen MONAI Bundles on their own labeled, case-audited cohorts.

This intentionally does not reuse the external-dataset cache specifications:
an internal validation cohort and a separately published test cohort have
different checkpoint-exposure claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
import zipfile
import zlib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch


@dataclass(frozen=True)
class Case:
    case_id: str
    image: Path
    labels: dict[str, Path]


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-root dataset file: {relative}")
    return path


def resolve_prostate_cases(root: Path) -> list[Case]:
    """Use only the publisher's separate Prostate158 test archive."""
    with (root / "test.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"t2", "t2_anatomy_reader1", "t2_anatomy_reader2"}
    if len(rows) != 19 or any(not required.issubset(row) for row in rows):
        raise ValueError("Expected the publisher's 19-case, two-rater Prostate158 test.csv")
    cases = []
    for row in rows:
        relative_image = Path(row["t2"])
        if len(relative_image.parts) != 3 or relative_image.parts[0] != "test" or relative_image.name != "t2.nii.gz":
            raise ValueError("Prostate158 test.csv must reference test/<case>/t2.nii.gz")
        image = _within(root, row["t2"])
        labels = {
            "reader1": _within(root, row["t2_anatomy_reader1"]),
            "reader2": _within(root, row["t2_anatomy_reader2"]),
        }
        if any(path.parent != image.parent for path in labels.values()):
            raise ValueError("Prostate158 image and rater labels must belong to the same test case")
        cases.append(Case(image.parent.name, image, labels))
    if len({case.case_id for case in cases}) != 19:
        raise ValueError("Prostate158 test case IDs are not unique")
    return cases


def audit_prostate_split_lists(cases: list[Case], training_list_root: Path) -> dict:
    """Audit publisher split paths, not archive-local numeric case IDs."""
    lists = {}
    for split in ("train", "valid"):
        path = training_list_root / f"{split}.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or any("t2" not in row for row in rows):
            raise ValueError(f"Malformed Prostate158 {split}.csv")
        paths = [Path(row["t2"]).parent.as_posix() for row in rows]
        if len(paths) != len(set(paths)) or any(not path.startswith("train/") for path in paths):
            raise ValueError(f"Duplicate or unexpected Prostate158 {split} case paths")
        lists[split] = {"paths": paths, "sha256": _hash_file(path)}
    training_paths = set(lists["train"]["paths"])
    validation_paths = set(lists["valid"]["paths"])
    test_paths = {f"test/{case.case_id}" for case in cases}
    if len(training_paths) + len(validation_paths) != 139 or len(test_paths) != 19:
        raise ValueError("Expected the publisher's 139 training/validation and 19 test cases")
    if training_paths & validation_paths or training_paths & test_paths or validation_paths & test_paths:
        raise ValueError("Prostate158 training, validation, and test case paths overlap")
    return {
        "training_cases": len(training_paths),
        "validation_cases": len(validation_paths),
        "test_cases": len(test_paths),
        "train_csv_sha256": lists["train"]["sha256"],
        "valid_csv_sha256": lists["valid"]["sha256"],
        "note": "Numeric IDs restart at 001 in the separate test archive; paths include the split namespace.",
    }


def audit_prostate_archives(
    training_list_root: Path,
    test_archive: Path,
    training_archive_url: str,
    expected_test_md5: str = "403b3736cb12013ed7585e07cbda6978",
) -> dict:
    """Use ZIP directories to rule out byte-identical images across archives."""
    import fsspec

    with fsspec.open(training_archive_url, "rb", block_size=1 << 20) as remote:
        with zipfile.ZipFile(remote) as train_zip, zipfile.ZipFile(test_archive) as test_zip:
            for split in ("train", "valid"):
                member = f"prostate158_train/{split}.csv"
                local_hash = _hash_file(training_list_root / f"{split}.csv")
                if hashlib.sha256(train_zip.read(member)).hexdigest() != local_hash:
                    raise ValueError(f"Extracted {split}.csv does not match the official training archive")
            counts = {}
            for modality in ("t2.nii.gz", "adc.nii.gz", "dwi.nii.gz"):
                train_files = [entry for entry in train_zip.infolist() if entry.filename.endswith("/" + modality)]
                test_files = [entry for entry in test_zip.infolist() if entry.filename.endswith("/" + modality)]
                train_signatures = {(entry.file_size, entry.CRC) for entry in train_files}
                test_signatures = {(entry.file_size, entry.CRC) for entry in test_files}
                duplicates = len(train_signatures & test_signatures)
                if len(train_files) != 139 or len(test_files) != 19 or duplicates:
                    raise ValueError(f"Unexpected Prostate158 {modality} archive counts or byte duplicates")
                counts[modality] = {"training_files": len(train_files), "test_files": len(test_files), "byte_identical_candidates": duplicates}
    test_md5 = _hash_file(test_archive, "md5")
    if test_md5 != expected_test_md5:
        raise ValueError("Prostate158 test archive does not match the publisher's MD5")
    return {"test_zip_md5": test_md5, "modality_files": counts}


def audit_prostate_extraction(cases: list[Case], root: Path, test_archive: Path) -> int:
    """Verify evaluated image/label bytes against the pinned test ZIP."""
    checked = 0
    with zipfile.ZipFile(test_archive) as archive:
        for case in cases:
            for path in (case.image, *case.labels.values()):
                relative = path.relative_to(root.resolve()).as_posix()
                member = archive.getinfo(f"prostate158_test/{relative}")
                if path.stat().st_size != member.file_size:
                    raise ValueError(f"Extracted Prostate158 file size differs from archive: {relative}")
                checksum = 0
                with path.open("rb") as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        checksum = zlib.crc32(chunk, checksum)
                if checksum != member.CRC:
                    raise ValueError(f"Extracted Prostate158 file CRC differs from archive: {relative}")
                checked += 1
    return checked


def resolve_spleen_cases(root: Path) -> tuple[list[Case], list[str]]:
    """Mirror the Bundle's sorted[:-9] training / sorted[-9:] validation."""
    labels = sorted((root / "labelsTr").glob("*.nii.gz"))
    images = sorted((root / "imagesTr").glob("*.nii.gz"))
    if len(labels) != 41 or len(images) != 41:
        raise ValueError("Expected the 41 labeled MSD Task09 Spleen training cases")
    train_ids = [path.name.removesuffix(".nii.gz") for path in labels[:-9]]
    cases = []
    for label in labels[-9:]:
        case_id = label.name.removesuffix(".nii.gz")
        image = _within(root, f"imagesTr/{label.name}")
        cases.append(Case(case_id, image, {"label": label.resolve()}))
    if set(train_ids) & {case.case_id for case in cases}:
        raise ValueError("MSD Spleen training and validation IDs overlap")
    if {path.name for path in images} != {path.name for path in labels}:
        raise ValueError("MSD Spleen images and labels do not pair one-to-one")
    return cases, train_ids


def _bundle_components(bundle_root: Path, device: torch.device):
    from monai.inferers import SlidingWindowInferer
    from monai.networks.nets import UNet

    config_path = bundle_root / "configs" / "inference.json"
    checkpoint_path = bundle_root / "models" / "model.pt"
    config = json.loads(config_path.read_text())
    definition = dict(config["network_def"])
    if definition.pop("_target_").rsplit(".", 1)[-1] != "UNet":
        raise ValueError("This evaluator expects the official UNet Bundle")
    model = UNet(**definition).to(device)
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights.get("model", weights), strict=True)
    model.eval()
    inferer_config = config["inferer"]
    if inferer_config["_target_"] != "SlidingWindowInferer":
        raise ValueError("Unexpected official inference mode")
    inferer = SlidingWindowInferer(
        roi_size=tuple(inferer_config["roi_size"]),
        sw_batch_size=1,  # bounded 24-GiB use; all other inference settings are official
        overlap=float(inferer_config["overlap"]),
    )
    return config, model, inferer, config_path, checkpoint_path


def _preprocessing(config: dict, task: str, label_keys: tuple[str, ...]):
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        ScaleIntensityd,
        ScaleIntensityRanged,
        Spacingd,
    )

    entries = config["preprocessing"]["transforms"]
    targets = [entry["_target_"] for entry in entries]
    expected = ["LoadImaged", "EnsureChannelFirstd", "Orientationd", "Spacingd"]
    expected += ["ScaleIntensityd", "NormalizeIntensityd"] if task == "prostate158_test" else ["ScaleIntensityRanged"]
    expected += ["EnsureTyped"]
    if targets != expected:
        raise ValueError(f"Unexpected Bundle preprocessing: {targets}")
    keys = ("image", *label_keys)
    orientation = entries[2]["axcodes"]
    spacing = tuple(entries[3]["pixdim"])
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes=orientation),
        Spacingd(keys=keys, pixdim=spacing, mode=("bilinear", *("nearest" for _ in label_keys))),
    ]
    if task == "prostate158_test":
        scale = entries[4]
        transforms.append(ScaleIntensityd(keys="image", minv=scale["minv"], maxv=scale["maxv"]))
        transforms.append(NormalizeIntensityd(keys="image"))
    else:
        scale = entries[4]
        transforms.append(
            ScaleIntensityRanged(
                keys="image",
                a_min=scale["a_min"],
                a_max=scale["a_max"],
                b_min=scale["b_min"],
                b_max=scale["b_max"],
                clip=scale["clip"],
            )
        )
    transforms.append(EnsureTyped(keys=keys))
    return Compose(transforms)


def score_foreground(prediction: torch.Tensor, label: torch.Tensor, foreground_ids: tuple[int, ...]) -> dict:
    """Case-level macro Dice/IoU over ground-truth-present foregrounds."""
    if prediction.shape != label.shape:
        raise ValueError("Prediction and label are not aligned")
    scores = {}
    for class_id in foreground_ids:
        truth = label == class_id
        if not bool(truth.any()):
            continue
        predicted = prediction == class_id
        tp = int((truth & predicted).sum())
        pred_count = int(predicted.sum())
        truth_count = int(truth.sum())
        scores[str(class_id)] = {
            "dice": 2 * tp / (pred_count + truth_count),
            "iou": tp / (pred_count + truth_count - tp),
            "truth_voxels": truth_count,
            "predicted_voxels": pred_count,
        }
    if not scores:
        raise ValueError("Case has no ground-truth foreground")
    return {
        "dice": statistics.mean(entry["dice"] for entry in scores.values()),
        "iou": statistics.mean(entry["iou"] for entry in scores.values()),
        "classes": scores,
    }


def _timed_call(function, probabilities: torch.Tensor) -> float:
    if probabilities.is_cuda:
        torch.cuda.synchronize(probabilities.device)
    started = time.perf_counter()
    prediction = function(probabilities)
    if probabilities.is_cuda:
        torch.cuda.synchronize(probabilities.device)
    elapsed = (time.perf_counter() - started) * 1000
    del prediction
    return elapsed


def _paired_times(probabilities: torch.Tensor, rankseg, *, warmup: int, repeats: int, case_index: int) -> dict:
    calls = {
        "argmax": lambda tensor: tensor.argmax(dim=1),
        "rma": rankseg.predict,
    }
    methods = tuple(calls)
    for iteration in range(warmup):
        for offset in range(2):
            _timed_call(calls[methods[(case_index + iteration + offset) % 2]], probabilities)
    samples = {method: [] for method in methods}
    for repetition in range(repeats):
        for offset in range(2):
            method = methods[(case_index + repetition + offset) % 2]
            samples[method].append(_timed_call(calls[method], probabilities))
    return {method: {"median_ms": statistics.median(values), "samples_ms": values} for method, values in samples.items()}


def _summary(records: list[dict], label_keys: tuple[str, ...], foreground_ids: tuple[int, ...]) -> dict:
    result = {"cases": len(records), "readers": {}}
    for label_key in label_keys:
        by_method = {}
        for method in ("argmax", "rma"):
            scores = [record["scores"][label_key][method] for record in records]
            by_method[method] = {
                "mean_case_dice": statistics.mean(score["dice"] for score in scores),
                "mean_case_iou": statistics.mean(score["iou"] for score in scores),
                "per_class_dice": {
                    str(class_id): statistics.mean(
                        score["classes"][str(class_id)]["dice"]
                        for score in scores if str(class_id) in score["classes"]
                    )
                    for class_id in foreground_ids
                },
            }
        result["readers"][label_key] = by_method
    result["mean_of_case_median_ms"] = {
        method: statistics.mean(record["timing"][method]["median_ms"] for record in records)
        for method in ("argmax", "rma")
    }
    return result


@torch.inference_mode()
def run(
    task: str,
    dataset_root: Path,
    bundle_parent: Path,
    *,
    training_list_root: Path | None,
    test_archive: Path | None,
    training_archive_url: str,
    safe_screening: bool,
    limit: int | None,
    warmup: int,
    repeats: int,
) -> dict:
    if warmup < 1 or repeats < 1 or (limit is not None and limit < 1):
        raise ValueError("warmup, repeats, and optional limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Official matched benchmark requires CUDA for comparable decoder timing")
    if task == "prostate158_test":
        cases = resolve_prostate_cases(dataset_root)
        if training_list_root is None or test_archive is None:
            raise ValueError("Prostate158 requires --training-list-root and --test-archive for split audit")
        isolation_audit = audit_prostate_split_lists(cases, training_list_root)
        isolation_audit["archives"] = audit_prostate_archives(training_list_root, test_archive, training_archive_url)
        isolation_audit["evaluated_files_verified_against_test_zip"] = audit_prostate_extraction(
            cases, dataset_root, test_archive
        )
        bundle_name, bundle_version = "prostate_mri_anatomy", "0.3.6"
        foreground_ids = (1, 2)
        exposure = "independent_publisher_test_archive"
        train_ids = None
    elif task == "msd_spleen_validation":
        cases, train_ids = resolve_spleen_cases(dataset_root)
        isolation_audit = {"training_cases": len(train_ids), "validation_cases": len(cases), "overlap": 0}
        bundle_name, bundle_version = "spleen_ct_segmentation", "0.6.1"
        foreground_ids = (1,)
        exposure = "not_gradient_trained_but_used_for_checkpoint_selection"
    else:
        raise ValueError(f"Unknown matched task: {task}")
    if limit is not None:
        cases = cases[:limit]
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")
    config, model, inferer, config_path, checkpoint_path = _bundle_components(bundle_parent / bundle_name, device)
    if int(config["network_def"]["out_channels"]) != len(foreground_ids) + 1:
        raise ValueError("Bundle output channels do not match task labels")
    from rankseg import RankSEG

    rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass", safe_screening=safe_screening)
    full_sort = RankSEG(metric="dice", solver="RMA", output_mode="multiclass", safe_screening=False)
    records = []
    for case_index, case in enumerate(cases):
        label_keys = tuple(case.labels)
        preprocessing = _preprocessing(config, task, label_keys)
        item = preprocessing({"image": str(case.image), **{key: str(path) for key, path in case.labels.items()}})
        image = item["image"].as_tensor().unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", enabled=bool(config["evaluator"]["amp"])):
            logits = inferer(image, model)
        probabilities = torch.softmax(logits.float(), dim=1).contiguous()
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError(f"Non-finite probabilities for {case.case_id}")
        base = probabilities.argmax(dim=1)[0]
        rma = rankseg.predict(probabilities)[0]
        full = full_sort.predict(probabilities)[0] if safe_screening else rma
        screening_changed_voxels = int((rma != full).sum())
        del full
        if base.shape != rma.shape:
            raise ValueError(f"Decoder shape mismatch for {case.case_id}")
        scores = {}
        for label_key in label_keys:
            label = item[label_key].as_tensor().squeeze(0).long().to(device)
            if set(int(value) for value in torch.unique(label)) - {0, *foreground_ids}:
                raise ValueError(f"Unexpected label IDs for {case.case_id}/{label_key}")
            scores[label_key] = {
                "argmax": score_foreground(base, label, foreground_ids),
                "rma": score_foreground(rma, label, foreground_ids),
            }
        timing = _paired_times(probabilities, rankseg, warmup=warmup, repeats=repeats, case_index=case_index)
        records.append({
            "case_id": case.case_id,
            "shape": list(base.shape),
            "source_sha256": {
                "image": _hash_file(case.image),
                "labels": {key: _hash_file(path) for key, path in case.labels.items()},
            },
            "scores": scores,
            "timing": timing,
            "changed_voxels": int((base != rma).sum()),
            "screening_changed_voxels_vs_full_sort": screening_changed_voxels,
        })
        print(f"[{case_index + 1}/{len(cases)}] {case.case_id}", flush=True)
        del item, image, logits, probabilities, base, rma
    return {
        "task": task,
        "bundle": bundle_name,
        "bundle_version": bundle_version,
        "bundle_config_sha256": _hash_file(config_path),
        "checkpoint_sha256": _hash_file(checkpoint_path),
        "dataset_root": str(dataset_root.resolve()),
        "exposure": exposure,
        "case_isolation_audit": isolation_audit,
        "training_case_ids": train_ids,
        "complete": limit is None,
        "rankseg_version": version("rankseg"),
        "torch_version": str(torch.__version__),
        "monai_version": version("monai"),
        "gpu": torch.cuda.get_device_name(),
        "deterministic_seed": 123,
        "rankseg": {"metric": "dice", "solver": "RMA", "output_mode": "multiclass", "smooth": 0.0,
                    "pruning_prob": 0.5, "safe_screening": safe_screening},
        "inference": {"sw_batch_size": 1, "amp": bool(config["evaluator"]["amp"]), "coordinate_space": "bundle_resampled_model_space"},
        "timing": {"warmup_per_case": warmup, "repeats_per_case": repeats, "scope": "decoder_only"},
        "summary": _summary(records, tuple(cases[0].labels), foreground_ids),
        "cases": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("prostate158_test", "msd_spleen_validation"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-list-root", type=Path, help="Directory with publisher train.csv and valid.csv (Prostate158 only).")
    parser.add_argument("--test-archive", type=Path, help="Publisher's prostate158_test.zip (Prostate158 only).")
    parser.add_argument(
        "--training-archive-url",
        default="https://zenodo.org/api/records/6481141/files/prostate158_train.zip/content",
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--safe-screening", action="store_true", help="Force safe-screened RMA Dice.")
    args = parser.parse_args(argv)
    result = run(
        args.task,
        args.dataset_root,
        args.bundle_dir,
        training_list_root=args.training_list_root,
        test_archive=args.test_archive,
        training_archive_url=args.training_archive_url,
        safe_screening=args.safe_screening,
        limit=args.limit,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
