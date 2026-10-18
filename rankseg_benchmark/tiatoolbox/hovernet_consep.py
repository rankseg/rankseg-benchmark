"""Paired HoVer-Net/CoNSeP nucleus decoder comparison.

Use the official TIAToolbox 1.6.0 checkpoint and tile inference. RankSEG
replaces only the nucleus-probability threshold; HV and type maps, followed
by the official HoVer-Net instance postprocessor, are shared by both arms.
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
BASE = ROOT / "artifacts" / "tiatoolbox_hovernet_consep"
DATA = ROOT / "artifacts" / "tiatoolbox_micronet_consep"
TEST = DATA / "CoNSeP" / "Test"
CHECKPOINT = BASE / "hovernet_original-consep.pth"
CHECKPOINT_SHA256 = "77e5baf98b1a2e4c5991c82bcf08a7c1b955e146f0ba9e655b292e1686cf646e"
DATA_SHA256 = "7f62e3fed9435d5e91568944bca9edd112ff0a67d0940a5300fbd848e5c255dd"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def cases(limit: int | None = None) -> list[str]:
    names = [f"test_{i}" for i in range(1, 15)]
    assert all((TEST / "Images" / f"{name}.png").exists() for name in names)
    assert all((TEST / "Labels" / f"{name}.mat").exists() for name in names)
    assert len(list((TEST / "Images").glob("test_*.png"))) == 14
    assert len(list((TEST / "Labels").glob("test_*.mat"))) == 14
    return names[:limit]


def check_sources() -> None:
    import tiatoolbox

    assert tiatoolbox.__version__ == "1.6.0", tiatoolbox.__version__
    assert digest(CHECKPOINT) == CHECKPOINT_SHA256
    assert digest(DATA / "consep_kaggle.zip") == DATA_SHA256


def infer(names: list[str]) -> None:
    import torch
    from tiatoolbox import rcParam
    from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

    check_sources()
    assert torch.cuda.is_available(), "CUDA required"
    rcParam["torch_compile_mode"] = "disable"
    model = SemanticSegmentor(
        pretrained_model="hovernet_original-consep",
        pretrained_weights=str(CHECKPOINT),
        batch_size=4,
        num_loader_workers=0,
        verbose=False,
    )
    out_dir = BASE / "probabilities"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names, 1):
        dests = [out_dir / f"{name}.{head}.npy" for head in ("np", "hv", "tp")]
        if all(dest.exists() for dest in dests):
            print(f"[{index}/{len(names)}] cached {name}", flush=True)
            continue
        assert not any(dest.exists() for dest in dests), f"Incomplete cache for {name}"
        image = TEST / "Images" / f"{name}.png"
        run_dir = BASE / "inference" / name
        assert not run_dir.exists(), f"Inspect incomplete inference at {run_dir}"
        start = time.monotonic()
        model.predict(
            imgs=[str(image)],
            mode="tile",
            save_dir=run_dir,
            device="cuda",
            crash_on_exception=True,
        )
        raws = [run_dir / f"0.raw.{i}.npy" for i in range(3)]
        arrays = [np.load(path, mmap_mode="r") for path in raws]
        with Image.open(image) as rgb:
            assert rgb.size == (1000, 1000)
            assert [arr.shape for arr in arrays] == [
                (rgb.height, rgb.width, 1),
                (rgb.height, rgb.width, 2),
                (rgb.height, rgb.width, 1),
            ], [arr.shape for arr in arrays]
        assert all(np.isfinite(arr).all() for arr in arrays)
        assert arrays[0].min() >= 0 and arrays[0].max() <= 1
        del arrays
        for raw, dest in zip(raws, dests):
            os.replace(raw, dest)
        print(f"[{index}/{len(names)}] {name}: {time.monotonic() - start:.1f}s", flush=True)


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    truth = gt > 0
    estimate = pred > 0
    denom = int(np.count_nonzero(truth)) + int(np.count_nonzero(estimate))
    return 1.0 if denom == 0 else 2 * int(np.count_nonzero(truth & estimate)) / denom


def source_meta() -> dict:
    return {
        **runtime_source(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dataset_archive_sha256": DATA_SHA256,
    }


def evaluate(names: list[str]) -> None:
    import torch
    from rankseg.functional import rankseg
    from tiatoolbox.models.architecture.hovernet import HoVerNet

    check_sources()
    assert torch.cuda.is_available(), "CUDA required"
    meta = source_meta()
    partial = BASE / "results.partial.json"
    state = json.loads(partial.read_text()) if partial.exists() else {"source": meta, "cases": []}
    assert state["source"] == meta, "Partial results use different source code or weights"
    completed = {row["id"] for row in state["cases"]}
    assert len(completed) == len(state["cases"])
    for index, name in enumerate(names, 1):
        if name in completed:
            print(f"[{index}/{len(names)}] already scored {name}", flush=True)
            continue
        arrays = [np.load(BASE / "probabilities" / f"{name}.{head}.npy", mmap_mode="r") for head in ("np", "hv", "tp")]
        np_map, hv_map, tp_map = arrays
        assert [arr.shape for arr in arrays] == [(1000, 1000, 1), (1000, 1000, 2), (1000, 1000, 1)]
        gt = loadmat(TEST / "Labels" / f"{name}.mat")["inst_map"]
        assert gt.shape == (1000, 1000)
        argmax_mask = np_map[..., 0] >= 0.5
        baseline, _ = HoVerNet.postproc([np_map, hv_map, tp_map])
        baseline_binary, _ = HoVerNet.postproc([argmax_mask[..., None].astype(np.uint8), hv_map, tp_map])
        assert np.array_equal(baseline, baseline_binary), f"Baseline does not match official postproc for {name}"

        tensor = torch.from_numpy(np.asarray(np_map).copy()).permute(2, 0, 1).unsqueeze(0).to("cuda")
        torch.cuda.synchronize()
        start = time.monotonic()
        decoded = rankseg(
            tensor,
            metric="dice",
            solver="RMA",
            output_mode="multilabel",
            safe_screening=True,
        )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        rankseg_mask = decoded[0, 0].cpu().numpy().astype(bool)
        assert rankseg_mask.shape == (1000, 1000)
        rankseg_inst, _ = HoVerNet.postproc([rankseg_mask[..., None].astype(np.uint8), hv_map, tp_map])
        row = {
            "id": name,
            "raw_dice": {
                "argmax": dice(gt, argmax_mask),
                "rankseg": dice(gt, rankseg_mask),
            },
            "postprocessed_dice": {
                "argmax": dice(gt, baseline),
                "rankseg": dice(gt, rankseg_inst),
            },
            "gt_pixels": int(np.count_nonzero(gt)),
            "postprocessed_pred_pixels": {
                "argmax": int(np.count_nonzero(baseline)),
                "rankseg": int(np.count_nonzero(rankseg_inst)),
            },
            "rankseg_decode_seconds": elapsed,
        }
        state["cases"].append(row)
        partial.write_text(json.dumps(state, indent=2) + "\n")
        print(f"[{index}/{len(names)}] {name}: {row['postprocessed_dice']}", flush=True)
    if len(names) < 14:
        return
    assert {row["id"] for row in state["cases"]} == set(names)
    state["cases"].sort(key=lambda row: int(row["id"].split("_")[1]))
    rng = np.random.default_rng(20261018)
    summaries = {}
    for metric in ("raw_dice", "postprocessed_dice"):
        baseline = np.array([row[metric]["argmax"] for row in state["cases"]])
        candidate = np.array([row[metric]["rankseg"] for row in state["cases"]])
        delta = candidate - baseline
        boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        summaries[metric] = {
            "argmax_mean": float(baseline.mean()),
            "rankseg_mean": float(candidate.mean()),
            "mean_gain": float(delta.mean()),
            "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            "wins": int(np.count_nonzero(delta > 1e-12)),
            "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
            "losses": int(np.count_nonzero(delta < -1e-12)),
        }
    state["summary"] = {
        "n": len(names),
        "cohort": "CoNSeP original Test split, test_1..test_14",
        "score_rule": "Binary nucleus Dice on inst_map > 0 per image, then mean across images",
        "decoder_rule": "Same np probability map; argmax threshold >=0.5 or RankSEG 0.0.8 Dice/RMA/multilabel with safe_screening=True; original HV/TP and official HoVer-Net postproc shared",
        "scores": summaries,
    }
    (BASE / "results.json").write_text(json.dumps(state, indent=2) + "\n")
    with (BASE / "per_case.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "raw_argmax", "raw_rankseg", "post_argmax", "post_rankseg"))
        for row in state["cases"]:
            writer.writerow(
                (
                    row["id"],
                    row["raw_dice"]["argmax"],
                    row["raw_dice"]["rankseg"],
                    row["postprocessed_dice"]["argmax"],
                    row["postprocessed_dice"]["rankseg"],
                )
            )
    partial.unlink()
    print(json.dumps(state["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("infer", "evaluate"))
    parser.add_argument("--limit", type=int, choices=range(1, 15))
    args = parser.parse_args()
    names = cases(args.limit)
    {"infer": infer, "evaluate": evaluate}[args.step](names)
