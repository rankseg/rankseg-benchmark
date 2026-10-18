"""Reproduce a paired BCSS test ROI comparison with TIAToolbox's released model.

Downloaded images, masks, probabilities, and results live under ``artifacts`` and
are excluded from Git. Run this with the dedicated TIAToolbox environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from .provenance import decoder_source


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "tiatoolbox_bcss"
SOURCE = BASE / "BCSS-source"
CSV = SOURCE / "meta" / "roiBounds.csv"
CHECKPOINT = BASE / "fcn_resnet50_unet-bcss.pth"
CHECKPOINT_SHA256 = "92d6c78e00ca18debfa1cfbb7c976facdce05fb9c0293117fbf4229e94a1aabf"
TEST_CENTERS = {"OL", "LL", "E2", "EW", "GM", "S3"}
CLASSES = ("tumor", "stroma", "inflammatory", "necrosis", "other")
# Five-class BCSS grouping: invasive tumor + angioinvasion + DCIS;
# lymphocytic/plasma/other immune infiltrate; all other annotated regions.
GT_TO_CLASS = np.full(22, 4, dtype=np.uint8)
GT_TO_CLASS[[1, 19, 20]] = 0
GT_TO_CLASS[2] = 1
GT_TO_CLASS[[3, 10, 11]] = 2
GT_TO_CLASS[4] = 3
API = "https://demo.kitware.com/histomicstk/api/v1/"
FOLDER_ID = "5bbdeba3e629140048d017bb"


def cohort() -> list[dict[str, str]]:
    rows = list(csv.DictReader(CSV.open(newline="")))
    test = [r for r in rows if r[""].split("-")[1] in TEST_CENTERS]
    assert len(test) == 45, f"Expected 45 test-center ROIs, found {len(test)}"
    return sorted(test, key=lambda r: r[""])


def get_bytes(url: str, *, params: dict | None = None) -> bytes:
    import requests

    last = None
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=(20, 180))
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            last = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}") from last


def download_one(row: dict[str, str], items: dict[str, str]) -> dict:
    uid = row[""]
    out = BASE / "data"
    out.mkdir(parents=True, exist_ok=True)
    rgb = out / f"{uid}.jpg"
    mask = out / f"{uid}.mask.png"
    if not rgb.exists():
        item_id = items[uid[:12]]
        params = {
            "left": int(row["xmin"]),
            "right": int(row["xmax"]),
            "top": int(row["ymin"]),
            "bottom": int(row["ymax"]),
            # Explicit 0.25 MPP, retaining precision lost by the original
            # download script's four-decimal millimetre formatting.
            "mm_x": "0.00025",
            "mm_y": "0.00025",
        }
        temp = rgb.with_suffix(".part")
        temp.write_bytes(get_bytes(f"{API}item/{item_id}/tiles/region", params=params))
        with Image.open(temp) as image:
            image.verify()
        os.replace(temp, rgb)
    if not mask.exists():
        temp = mask.with_suffix(".part")
        temp.write_bytes(get_bytes(row["mask_link"]))
        with Image.open(temp) as image:
            image.verify()
        os.replace(temp, mask)
    with Image.open(rgb) as image, Image.open(mask) as gt:
        return {"id": uid, "rgb_size": image.size, "mask_size": gt.size}


def download() -> None:
    rows = cohort()
    items_raw = json.loads(get_bytes(f"{API}item", params={"folderId": FOLDER_ID, "limit": 200}))
    items = {item["name"][:12]: item["_id"] for item in items_raw}
    assert all(row[""][:12] in items for row in rows)
    manifest = {
        "selection": "All 45 ROIs from the published test centers; alphabetical output order",
        "test_centers": sorted(TEST_CENTERS),
        "mpp": 0.25,
        "source": "https://github.com/PathologyDataScience/BCSS",
        "checkpoint": "TIACentre/TIAToolbox_pretrained_weights/fcn_resnet50_unet-bcss.pth",
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "rows": rows,
    }
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(download_one, row, items): row[""] for row in rows}
        for task in as_completed(tasks):
            print(json.dumps(task.result()), flush=True)


def infer(only_case: str | None = None) -> None:
    import torch
    from tiatoolbox import rcParam
    from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

    assert torch.cuda.is_available(), "GPU is required for the full-resolution model run"
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    # Compilation affects runtime only and is unnecessary for this fixed model.
    rcParam["torch_compile_mode"] = "disable"
    model = SemanticSegmentor(
        pretrained_model="fcn_resnet50_unet-bcss",
        pretrained_weights=str(CHECKPOINT),
        batch_size=2,
        num_loader_workers=0,
        verbose=False,
    )
    prob_dir = BASE / "probabilities"
    prob_dir.mkdir(exist_ok=True)
    rows = [row for row in cohort() if only_case is None or row[""] == only_case]
    assert rows, f"Unknown case: {only_case}"
    for index, row in enumerate(rows, 1):
        uid = row[""]
        prob = prob_dir / f"{uid}.npy"
        if prob.exists():
            print(f"[{index}/{len(rows)}] cached {uid}", flush=True)
            continue
        rgb = BASE / "data" / f"{uid}.jpg"
        assert rgb.exists(), rgb
        output = BASE / "inference" / uid
        if output.exists():
            # Incomplete attempts are preserved for inspection, and the next
            # attempt gets a fresh destination required by TIAToolbox.
            output = BASE / "inference" / f"{uid}-{int(time.time())}"
        print(f"[{index}/{len(rows)}] inference {uid}", flush=True)
        started = time.time()
        model.predict(
            imgs=[str(rgb)],
            mode="tile",
            save_dir=output,
            device="cuda",
            crash_on_exception=True,
        )
        raw = output / "0.raw.0.npy"
        assert raw.exists(), raw
        arr = np.load(raw, mmap_mode="r")
        with Image.open(rgb) as image:
            assert arr.shape == (image.height, image.width, 5), (arr.shape, image.size)
        os.replace(raw, prob)
        print(f"[{index}/{len(rows)}] saved {prob} in {time.time() - started:.1f}s", flush=True)


def score_case(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> list[float]:
    scores = []
    for klass in range(5):
        truth = valid & (gt == klass)
        estimate = valid & (pred == klass)
        n_true = int(truth.sum())
        n_pred = int(estimate.sum())
        scores.append(1.0 if n_true + n_pred == 0 else 2 * int(np.count_nonzero(truth & estimate)) / (n_true + n_pred))
    return scores


def evaluate() -> None:
    import torch
    import rankseg as rankseg_package
    from rankseg.functional import rankseg

    from PIL import Image as PILImage

    assert torch.cuda.is_available()
    source_meta = decoder_source()
    partial = BASE / "results.partial.json"
    partial_meta = BASE / "results.partial.meta.json"
    if partial.exists():
        assert partial_meta.exists(), "Partial results have no local-source provenance; archive them first"
        assert json.loads(partial_meta.read_text()) == source_meta, "Partial results use another RankSEG source"
    else:
        assert not partial_meta.exists(), "Partial metadata exists without partial results"
        partial_meta.write_text(json.dumps(source_meta, indent=2) + "\n")
    results = json.loads(partial.read_text()) if partial.exists() else []
    completed = {result["id"] for result in results}
    rows = cohort()
    for index, row in enumerate(rows, 1):
        uid = row[""]
        if uid in completed:
            print(f"[{index}/{len(rows)}] already scored {uid}", flush=True)
            continue
        prob_path = BASE / "probabilities" / f"{uid}.npy"
        assert prob_path.exists(), prob_path
        probs = np.load(prob_path, mmap_mode="r")
        assert probs.ndim == 3 and probs.shape[-1] == 5
        mask = BASE / "data" / f"{uid}.mask.png"
        with PILImage.open(mask) as image:
            raw = np.asarray(image.resize((probs.shape[1], probs.shape[0]), resample=PILImage.Resampling.NEAREST))
        assert raw.min() >= 0 and raw.max() <= 21, (uid, np.unique(raw))
        # BCSS code 0 is outside_roi; code 7 is explicitly "exclude".
        # Neither contributes to the evaluation, regardless of prediction.
        valid = (raw != 0) & (raw != 7)
        gt = GT_TO_CLASS[raw]
        argmax = np.argmax(probs, axis=-1).astype(np.uint8)
        if not np.all(np.isfinite(probs)):
            raise ValueError(f"Nonfinite probabilities: {uid}")
        sums = np.asarray(probs).sum(axis=-1)
        print(
            f"[{index}/{len(rows)}] decoding {uid}: shape={probs.shape}, sum range={sums.min():.5f}..{sums.max():.5f}",
            flush=True,
        )
        if not (0.99 <= float(sums.min()) <= float(sums.max()) <= 1.01):
            raise ValueError(f"Probability channels do not sum to one: {uid}")
        del sums
        torch_probs = torch.from_numpy(np.asarray(probs).copy()).permute(2, 0, 1).unsqueeze(0).to("cuda")
        started = time.time()
        rank = (
            rankseg(
                torch_probs,
                metric="dice",
                solver="RMA",
                output_mode="multiclass",
                safe_screening=True,
            )[0]
            .to("cpu")
            .numpy()
            .astype(np.uint8)
        )
        seconds = time.time() - started
        del torch_probs
        torch.cuda.empty_cache()
        a = score_case(gt, argmax, valid)
        r = score_case(gt, rank, valid)
        present = [k for k in range(5) if np.any(valid & (gt == k))]
        results.append(
            {
                "id": uid,
                "height": int(probs.shape[0]),
                "width": int(probs.shape[1]),
                "valid_pixels": int(valid.sum()),
                "argmax": a,
                "rankseg": r,
                "argmax_macro": float(np.mean(a)),
                "rankseg_macro": float(np.mean(r)),
                "rankseg_seconds": seconds,
                "argmax_present_macro": float(np.mean([a[k] for k in present])),
                "rankseg_present_macro": float(np.mean([r[k] for k in present])),
                "gt_pixels": [int(np.count_nonzero(valid & (gt == k))) for k in range(5)],
                "argmax_pixels": [int(np.count_nonzero(valid & (argmax == k))) for k in range(5)],
                "rankseg_pixels": [int(np.count_nonzero(valid & (rank == k))) for k in range(5)],
            }
        )
        partial.write_text(json.dumps(results, indent=2) + "\n")
        print(
            f"[{index}/{len(rows)}] macro {results[-1]['argmax_macro']:.4f} -> {results[-1]['rankseg_macro']:.4f} ({seconds:.1f}s)",
            flush=True,
        )
    assert {r["id"] for r in results} == {row[""] for row in rows}, "Incomplete evaluation"
    results.sort(key=lambda r: r["id"])
    deltas = np.array([r["rankseg_macro"] - r["argmax_macro"] for r in results])
    rng = np.random.default_rng(20261017)
    boot = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
    pooled = {}
    for method in ("argmax", "rankseg"):
        pooled[method] = []
        for klass in range(5):
            n_true = sum(r["gt_pixels"][klass] for r in results)
            n_pred = sum(r[f"{method}_pixels"][klass] for r in results)
            n_intersection = sum(
                r[method][klass] * (r["gt_pixels"][klass] + r[f"{method}_pixels"][klass]) / 2 for r in results
            )
            pooled[method].append(2 * n_intersection / (n_true + n_pred))
    absent_counts = {}
    for klass, name in enumerate(CLASSES):
        absent = [r for r in results if r["gt_pixels"][klass] == 0]
        absent_counts[name] = {
            "gt_absent_cases": len(absent),
            "argmax_false_positive_cases": sum(r["argmax_pixels"][klass] > 0 for r in absent),
            "rankseg_false_positive_cases": sum(r["rankseg_pixels"][klass] > 0 for r in absent),
        }
    summary = {
        "n": len(results),
        "classes": CLASSES,
        "macro_argmax": float(np.mean([r["argmax_macro"] for r in results])),
        "macro_rankseg": float(np.mean([r["rankseg_macro"] for r in results])),
        "mean_delta": float(deltas.mean()),
        "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "wins": int(np.count_nonzero(deltas > 0)),
        "ties": int(np.count_nonzero(deltas == 0)),
        "losses": int(np.count_nonzero(deltas < 0)),
        "per_class_argmax": np.mean([r["argmax"] for r in results], axis=0).tolist(),
        "per_class_rankseg": np.mean([r["rankseg"] for r in results], axis=0).tolist(),
        "present_class_macro_argmax": float(np.mean([r["argmax_present_macro"] for r in results])),
        "present_class_macro_rankseg": float(np.mean([r["rankseg_present_macro"] for r in results])),
        "pooled_class_dice": pooled,
        "pooled_five_class_macro": {method: float(np.mean(scores)) for method, scores in pooled.items()},
        "absent_class_counts": absent_counts,
        "score_rule": "Per-ROI Dice for all five classes; both empty=1, GT empty/pred nonempty=0; ignore raw label 0 (outside_roi) and 7 (exclude); ROI mean then mean of 45 ROI means.",
        "mapping": {
            "tumor": [1, 19, 20],
            "stroma": [2],
            "inflammatory": [3, 10, 11],
            "necrosis": [4],
            "other": [5, 6, 8, 9, 12, 13, 14, 15, 16, 17, 18, 21],
        },
        "decoder": "RankSEG 0.0.8; Dice/RMA/multiclass/safe_screening=True/pruning_prob=0.5",
        "rankseg_source": source_meta,
        "versions": {"rankseg": rankseg_package.__version__, "torch": torch.__version__},
    }
    (BASE / "results.json").write_text(json.dumps({"summary": summary, "cases": results}, indent=2) + "\n")
    with (BASE / "per_case.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "argmax_macro",
                "rankseg_macro",
                "delta",
                *[f"argmax_{c}" for c in CLASSES],
                *[f"rankseg_{c}" for c in CLASSES],
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["id"],
                    result["argmax_macro"],
                    result["rankseg_macro"],
                    result["rankseg_macro"] - result["argmax_macro"],
                    *result["argmax"],
                    *result["rankseg"],
                ]
            )
    partial.unlink()
    partial_meta.unlink()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("download", "infer", "evaluate"))
    parser.add_argument("--case", help="Run inference for one ROI as a pilot")
    args = parser.parse_args()
    if args.step == "infer":
        infer(args.case)
    else:
        assert args.case is None, "--case is only supported for inference"
        {"download": download, "evaluate": evaluate}[args.step]()
