"""Paired MicroNet/CoNSeP decoder comparison on the original 14 test tiles.

Use TIAToolbox 1.6.0 for inference and instance postprocessing. All methods
decode the same saved probability map. Downloaded data and outputs stay in
artifacts/tiatoolbox_micronet_consep (excluded from Git).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .provenance import runtime_source


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "tiatoolbox_micronet_consep"
TEST = BASE / "CoNSeP" / "Test"
CHECKPOINT = BASE / "micronet-consep.pth"
CHECKPOINT_SHA256 = "f085d591f74cf8de3a7a5a75f08fab3e47859c49bfc7415bc567113dba778279"
DATA_SHA256 = "7f62e3fed9435d5e91568944bca9edd112ff0a67d0940a5300fbd848e5c255dd"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def cases() -> list[str]:
    names = [f"test_{i}" for i in range(1, 15)]
    for name in names:
        assert (TEST / "Images" / f"{name}.png").exists(), name
        assert (TEST / "Labels" / f"{name}.mat").exists(), name
    assert len(list((TEST / "Images").glob("test_*.png"))) == 14
    assert len(list((TEST / "Labels").glob("test_*.mat"))) == 14
    return names


def source_meta() -> dict:
    return {
        **runtime_source(),
        "checkpoint_sha256": digest(CHECKPOINT),
        "dataset_archive_sha256": digest(BASE / "consep_kaggle.zip"),
    }


def infer() -> None:
    import torch
    from tiatoolbox import rcParam
    from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

    assert torch.cuda.is_available(), "CUDA required for this model run"
    assert digest(CHECKPOINT) == CHECKPOINT_SHA256
    assert digest(BASE / "consep_kaggle.zip") == DATA_SHA256
    names = cases()
    rcParam["torch_compile_mode"] = "disable"
    model = SemanticSegmentor(
        pretrained_model="micronet-consep",
        pretrained_weights=str(CHECKPOINT),
        batch_size=4,
        num_loader_workers=0,
        verbose=False,
    )
    out_dir = BASE / "probabilities"
    out_dir.mkdir(exist_ok=True)
    for index, name in enumerate(names, 1):
        dest = out_dir / f"{name}.npy"
        if dest.exists():
            print(f"[{index}/14] cached {name}", flush=True)
            continue
        image = TEST / "Images" / f"{name}.png"
        run_dir = BASE / "inference" / name
        assert not run_dir.exists(), f"Inspect incomplete output before retry: {run_dir}"
        start = time.monotonic()
        model.predict(
            imgs=[str(image)],
            mode="tile",
            save_dir=run_dir,
            device="cuda",
            crash_on_exception=True,
        )
        raw = run_dir / "0.raw.0.npy"
        probs = np.load(raw, mmap_mode="r")
        with Image.open(image) as rgb:
            assert rgb.size == (1000, 1000)
            assert probs.shape == (rgb.height, rgb.width, 2), probs.shape
        assert np.isfinite(probs).all()
        assert np.allclose(probs.sum(axis=-1), 1, atol=1e-5)
        del probs
        os.replace(raw, dest)
        print(f"[{index}/14] {name}: {time.monotonic() - start:.1f}s", flush=True)


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    # Same binary Dice as the HoVer-Net authors' get_dice_1(inst_map, inst_map).
    truth = gt > 0
    estimate = pred > 0
    denom = int(np.count_nonzero(truth)) + int(np.count_nonzero(estimate))
    return 1.0 if denom == 0 else 2 * int(np.count_nonzero(truth & estimate)) / denom


def postprocess_binary(mask: np.ndarray) -> np.ndarray:
    """Run TIAToolbox's unmodified MicroNet instance postprocessing."""
    from tiatoolbox.models.architecture.micronet import MicroNet

    two_channels = np.stack((~mask, mask), axis=-1).astype(np.uint8)
    return MicroNet.postproc([two_channels])[0]


def evaluate() -> None:
    import torch
    from rankseg.functional import rankseg

    assert torch.cuda.is_available(), "CUDA required for full-resolution decoding"
    assert digest(CHECKPOINT) == CHECKPOINT_SHA256
    assert digest(BASE / "consep_kaggle.zip") == DATA_SHA256
    names = cases()
    meta = source_meta()
    partial = BASE / "results.partial.json"
    state = json.loads(partial.read_text()) if partial.exists() else {"source": meta, "cases": []}
    assert state["source"] == meta, "Partial results use a different source"
    completed = {row["id"] for row in state["cases"]}
    assert len(completed) == len(state["cases"])
    for index, name in enumerate(names, 1):
        if name in completed:
            print(f"[{index}/14] already scored {name}", flush=True)
            continue
        probs = np.load(BASE / "probabilities" / f"{name}.npy", mmap_mode="r")
        assert probs.shape == (1000, 1000, 2)
        gt = loadmat(TEST / "Labels" / f"{name}.mat")["inst_map"]
        assert gt.shape == probs.shape[:2]
        argmax = np.argmax(probs, axis=-1).astype(bool)
        methods = {"argmax": argmax}
        times = {}
        for mode in ("foreground", "multiclass"):
            channels = np.asarray(probs[..., 1:2] if mode == "foreground" else probs).copy()
            tensor = torch.from_numpy(channels).permute(2, 0, 1).unsqueeze(0).to("cuda")
            torch.cuda.synchronize()
            start = time.monotonic()
            decoded = rankseg(
                tensor,
                metric="dice",
                solver="RMA",
                output_mode="multilabel" if mode == "foreground" else "multiclass",
                safe_screening=True,
            )
            torch.cuda.synchronize()
            times[mode] = time.monotonic() - start
            methods[mode] = (
                decoded[0, 0].cpu().numpy() if mode == "foreground" else decoded[0].cpu().numpy().astype(bool)
            )
            del tensor, decoded, channels
        raw_scores = {method: dice(gt, mask) for method, mask in methods.items()}
        post_scores = {}
        pred_pixels = {}
        for method, mask in methods.items():
            post = postprocess_binary(mask)
            post_scores[method] = dice(gt, post)
            pred_pixels[method] = int(np.count_nonzero(post))
            del post
        state["cases"].append(
            {
                "id": name,
                "raw_dice": raw_scores,
                "postprocessed_dice": post_scores,
                "rankseg_decode_seconds": times,
                "gt_pixels": int(np.count_nonzero(gt)),
                "postprocessed_pred_pixels": pred_pixels,
            }
        )
        partial.write_text(json.dumps(state, indent=2) + "\n")
        print(f"[{index}/14] {name}: raw {raw_scores} / post {post_scores}", flush=True)
    assert {row["id"] for row in state["cases"]} == set(names)
    state["cases"].sort(key=lambda row: int(row["id"].split("_")[1]))
    rng = np.random.default_rng(20261018)
    summaries = {}
    for metric in ("raw_dice", "postprocessed_dice"):
        scores = {
            method: np.array([row[metric][method] for row in state["cases"]])
            for method in ("argmax", "foreground", "multiclass")
        }
        summaries[metric] = {}
        for method, values in scores.items():
            delta = values - scores["argmax"]
            boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
            summaries[metric][method] = {
                "mean": float(values.mean()),
                "mean_delta_vs_argmax": float(delta.mean()),
                "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
                "wins": int(np.count_nonzero(delta > 1e-12)),
                "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
                "losses": int(np.count_nonzero(delta < -1e-12)),
            }
    state["summary"] = {
        "n": 14,
        "cohort": "CoNSeP original Test split, test_1..test_14",
        "score_rule": "For each image, binary Dice on nucleus pixels (inst_map > 0), then mean over 14 images; both-empty=1. Official MicroNet connected-component postprocessing is identical for all decoded masks.",
        "decoders": {
            "argmax": "TIAToolbox MicroNet channel argmax",
            "foreground": "RankSEG 0.0.8, Dice/RMA, single nucleus channel, multilabel, safe_screening=True",
            "multiclass": "RankSEG 0.0.8, Dice/RMA, two channels, multiclass, safe_screening=True",
        },
        "data_source": "https://www.kaggle.com/datasets/karthikperupogu/consep; original Warwick link requires login as of 2026-10-18",
        "scores": summaries,
    }
    (BASE / "results.json").write_text(json.dumps(state, indent=2) + "\n")
    with (BASE / "per_case.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "id",
                "raw_argmax",
                "raw_foreground",
                "raw_multiclass",
                "post_argmax",
                "post_foreground",
                "post_multiclass",
            )
        )
        for row in state["cases"]:
            writer.writerow(
                (
                    row["id"],
                    *(row["raw_dice"][method] for method in ("argmax", "foreground", "multiclass")),
                    *(row["postprocessed_dice"][method] for method in ("argmax", "foreground", "multiclass")),
                )
            )
    partial.unlink()
    print(json.dumps(state["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("infer", "evaluate"))
    args = parser.parse_args()
    {"infer": infer, "evaluate": evaluate}[args.step]()
