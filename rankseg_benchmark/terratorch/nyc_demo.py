"""Paired decoder check on the one labeled TerraMind-NYC demo chip.

This is an integration smoke test. The model publisher does not identify the
demo chip's train/validation/test membership, so these scores are not an
independent test-set estimate.

Requires TerraTorch 1.2.7, the IBM ImpactMesh Python package at commit
451038f21264256590eee0c73bec958897895213, RankSEG 0.0.7, and the LULC
checkpoint, YAML, and four demo-chip files from
https://huggingface.co/msradam/TerraMind-base-Flood-NYC at revision
ddebd8390c9516ef65e6cf3e16f3a0f8a3b51d2d. This is a community model
built with TerraTorch; it is not a TerraTorch-maintained checkpoint.

Run ``python -m rankseg_benchmark.terratorch.nyc_demo --assets DIR --output FILE``.
``DIR`` must contain the checkpoint, YAML, and ``demo_chip`` subdirectory with
the original filenames from the model repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
import yaml
from safetensors.torch import load_file


CHIP_ID = "nyc_452U_625L_r1c1"
CHECKPOINT = "TerraMind_v1_base_NYC_LULC.safetensors"
CONFIG = "terramind_v1_base_nyc_phase2.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_demo(assets: Path, destination: Path) -> Path:
    files = {
        "S2L2A": ("s2.zarr.zip", "S2L2A.zarr.zip"),
        "S1RTC": ("s1.zarr.zip", "S1RTC.zarr.zip"),
        "DEM": ("dem.tif", "DEM.tif"),
        "MASK": ("msk.tif", "annotation_flood.tif"),
    }
    for modality, (source_suffix, target_suffix) in files.items():
        source = (assets / "demo_chip" / f"{CHIP_ID}_{source_suffix}").resolve(strict=True)
        target = destination / modality / f"{CHIP_ID}_{target_suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            if target.resolve() != source:
                raise FileExistsError(f"{target} already points to a different file")
        else:
            target.symlink_to(source)
    (destination / "demo_split.txt").write_text(CHIP_ID + "\n")
    return destination


def _scores(prediction: torch.Tensor, target: torch.Tensor, classes: int) -> dict:
    per_class = {}
    for label in range(classes):
        predicted = prediction == label
        actual = target == label
        intersection = int((predicted & actual).sum())
        predicted_count = int(predicted.sum())
        actual_count = int(actual.sum())
        union = predicted_count + actual_count - intersection
        per_class[str(label)] = {
            "truth_pixels": actual_count,
            "predicted_pixels": predicted_count,
            "true_positive_pixels": intersection,
            "precision": intersection / max(1, predicted_count),
            "recall": intersection / max(1, actual_count),
            "dice": 2 * intersection / max(1, predicted_count + actual_count),
            "iou": intersection / max(1, union),
        }
    return {
        "per_class": per_class,
        "macro_dice": sum(item["dice"] for item in per_class.values()) / classes,
        "macro_iou": sum(item["iou"] for item in per_class.values()) / classes,
    }


def evaluate(assets: Path, output: Path) -> dict:
    from impactmesh.impactmesh_datamodule import ImpactMeshDataModule
    from rankseg import RankSEG
    from terratorch.tasks import SemanticSegmentationTask

    if version("rankseg") != "0.0.7":
        raise RuntimeError("This experiment requires RankSEG 0.0.7")
    if version("terratorch") != "1.2.7":
        raise RuntimeError("This experiment requires TerraTorch 1.2.7")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pinned fp16 inference check")

    assets = assets.resolve(strict=True)
    checkpoint = (assets / CHECKPOINT).resolve(strict=True)
    config_path = (assets / CONFIG).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    model_args = config["model"]["init_args"]
    # The released safetensors contains every task parameter (checked by
    # strict loading below). Avoid downloading the redundant 1.5 GB base model.
    model_args["model_args"]["backbone_pretrained"] = False
    task = SemanticSegmentationTask(**model_args)
    task.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)

    staged = _stage_demo(assets, output.parent / "nyc_demo_data")
    datamodule = ImpactMeshDataModule(
        data_root=str(staged),
        test_split=str(staged / "demo_split.txt"),
        batch_size=1,
        num_workers=0,
        modalities=["S2L2A", "S1RTC", "DEM"],
        timesteps=[0, 1, 2, 3],
        no_data_replace=0,
        disaster_type="flood",
    )
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))
    batch = datamodule.aug(batch)  # ImpactMesh flood statistics from the released setup.
    target = batch["mask"].cpu()
    if target.ndim == 4:
        target = target[:, 0]
    if target.shape != (1, 256, 256) or not torch.all((0 <= target) & (target < 5)):
        raise ValueError("Unexpected demo mask shape or labels")

    task = task.eval().cuda()
    inputs = {name: tensor.cuda() for name, tensor in batch["image"].items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        logits = task.model(inputs).output
    if logits.shape != (1, 5, 256, 256):
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    # TerraTorch's segmentation head ends in a convolution, so .output is
    # logits. RankSEG requires probabilities: apply softmax exactly once.
    probs = logits.float().softmax(dim=1).contiguous()
    if not torch.isfinite(probs).all():
        raise ValueError("Non-finite model probabilities")
    probability_sums = probs.sum(dim=1)
    if not torch.allclose(probability_sums, torch.ones_like(probability_sums), atol=1e-6):
        raise ValueError("Class probabilities do not sum to one")

    prediction_argmax = probs.argmax(dim=1).cpu()
    decoder_args = {"metric": "dice", "solver": "RMA", "output_mode": "multiclass"}
    rankseg = RankSEG(**decoder_args, safe_screening=True)
    torch.cuda.synchronize()
    start = perf_counter()
    prediction_rankseg = rankseg(probs).cpu()
    torch.cuda.synchronize()
    rankseg_ms = (perf_counter() - start) * 1000
    prediction_full_sort = RankSEG(**decoder_args, safe_screening=False)(probs).cpu()
    if prediction_rankseg.shape != target.shape:
        raise ValueError(f"Unexpected RankSEG shape: {tuple(prediction_rankseg.shape)}")

    result = {
        "scope": "one published demo chip; split membership unknown",
        "model_repo": "msradam/TerraMind-base-Flood-NYC",
        "model_revision": "ddebd8390c9516ef65e6cf3e16f3a0f8a3b51d2d",
        "impactmesh_revision": "451038f21264256590eee0c73bec958897895213",
        "chip_id": CHIP_ID,
        "checkpoint_sha256": _sha256(checkpoint),
        "config_sha256": _sha256(config_path),
        "software": {"rankseg": version("rankseg"), "terratorch": version("terratorch"),
                     "torch": torch.__version__},
        "decoder": {**decoder_args, "safe_screening": True},
        "output_diagnostics": {
            "raw_output_min": float(logits.min()),
            "raw_output_max": float(logits.max()),
            "raw_channel_sum_min": float(logits.sum(dim=1).min()),
            "raw_channel_sum_max": float(logits.sum(dim=1).max()),
            "softmax_channel_sum_min": float(probability_sums.min()),
            "softmax_channel_sum_max": float(probability_sums.max()),
            "softmax_expected_pixels_by_class": [float(x) for x in probs.sum(dim=(0, 2, 3))],
        },
        "argmax": _scores(prediction_argmax, target, 5),
        "rankseg": _scores(prediction_rankseg, target, 5),
        "rankseg_full_sort_audit": _scores(prediction_full_sort, target, 5),
        "changed_pixels": int((prediction_argmax != prediction_rankseg).sum()),
        "screening_full_sort_changed_pixels": int((prediction_rankseg != prediction_full_sort).sum()),
        "rankseg_single_call_ms": rankseg_ms,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.assets, args.output), indent=2))


if __name__ == "__main__":
    main()
