"""HoVer-Net Kumar test-set comparison: threshold decoding versus RankSEG.

The HoVer-Net authors' `kumar/test_same` and `kumar/test_diff` groups are
kept separate. Both decoders receive the same official model probability map;
the HV map and TIAToolbox instance postprocessor remain identical.
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
BASE = ROOT / "artifacts" / "tiatoolbox_hovernet_kumar"
DATA = BASE / "data"
INDEX = Path(__file__).resolve().parent / "metadata" / "kumar_test_index.json"
LEGACY_INDEX_SHA256 = "d812ccda25a4cf04e961f9fe7a4a6b70e435cb43cb7df6602bfa23395b26ea03"
MANIFEST = BASE / "data_manifest.json"
CHECKPOINT = BASE / "hovernet_original-kumar.pth"
CHECKPOINT_SHA256 = "d6f0510a99216e7295a19470c4430cb9840e2ef544b40e5f89a921e5117d674e"
AUTHOR_FOLDER = "https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def selected_index() -> list[dict]:
    entries = json.loads(INDEX.read_text())
    chosen = [
        entry
        for entry in entries
        if entry["path"].startswith(("kumar/test_same/", "kumar/test_diff/"))
        and entry["path"].split("/")[2] in ("Images", "Labels")
    ]
    paths = [entry["path"] for entry in chosen]
    assert len(paths) == len(set(paths)) == 28, len(paths)
    for split, count in (("test_same", 8), ("test_diff", 6)):
        images = {Path(path).stem for path in paths if path.startswith(f"kumar/{split}/Images/")}
        labels = {Path(path).stem for path in paths if path.startswith(f"kumar/{split}/Labels/")}
        assert images == labels and len(images) == count, (split, images, labels)
    return sorted(chosen, key=lambda entry: entry["path"])


def check_manifest_index(manifest: dict) -> None:
    # Existing local runs recorded the hash of the original full author listing.
    assert manifest["source_index_sha256"] in {digest(INDEX), LEGACY_INDEX_SHA256}
    assert manifest["source_folder"] == AUTHOR_FOLDER
    selected = {row["path"]: row["url"] for row in selected_index()}
    recorded = {row["path"]: row["url"] for row in manifest["files"]}
    assert len(recorded) == len(manifest["files"]) == len(selected)
    assert recorded == selected


def cases(limit: int | None = None) -> list[tuple[str, str]]:
    names = []
    for split in ("test_same", "test_diff"):
        stems = sorted(
            Path(entry["path"]).stem for entry in selected_index() if entry["path"].startswith(f"kumar/{split}/Images/")
        )
        names.extend((split, stem) for stem in stems)
    assert len(names) == 14
    return names[:limit]


def files(split: str, stem: str) -> tuple[Path, Path]:
    return (
        DATA / "kumar" / split / "Images" / f"{stem}.tif",
        DATA / "kumar" / split / "Labels" / f"{stem}.mat",
    )


def key(split: str, stem: str) -> str:
    return f"{split}__{stem}"


def download() -> None:
    import gdown

    chosen = selected_index()
    records = []
    for index, entry in enumerate(chosen, 1):
        path = DATA / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            output = gdown.download(entry["url"], output=str(path), quiet=True)
            assert output and path.exists(), entry["path"]
        records.append(
            {
                "path": entry["path"],
                "url": entry["url"],
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
        print(f"[{index}/{len(chosen)}] {entry['path']} ({path.stat().st_size} bytes)", flush=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "source_folder": AUTHOR_FOLDER,
                "source_index_sha256": digest(INDEX),
                "files": records,
            },
            indent=2,
        )
        + "\n"
    )


def check_sources() -> None:
    import tiatoolbox

    assert tiatoolbox.__version__ == "1.6.0", tiatoolbox.__version__
    assert digest(CHECKPOINT) == CHECKPOINT_SHA256
    manifest = json.loads(MANIFEST.read_text())
    check_manifest_index(manifest)
    for entry in manifest["files"]:
        path = DATA / entry["path"]
        assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], path


def infer(names: list[tuple[str, str]]) -> None:
    import torch
    from tiatoolbox import rcParam
    from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

    check_sources()
    assert torch.cuda.is_available(), "CUDA required"
    rcParam["torch_compile_mode"] = "disable"
    model = SemanticSegmentor(
        pretrained_model="hovernet_original-kumar",
        pretrained_weights=str(CHECKPOINT),
        batch_size=4,
        num_loader_workers=0,
        verbose=False,
    )
    out_dir = BASE / "probabilities"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, (split, stem) in enumerate(names, 1):
        name = key(split, stem)
        dests = [out_dir / f"{name}.{head}.npy" for head in ("np", "hv")]
        if all(dest.exists() for dest in dests):
            print(f"[{index}/{len(names)}] cached {name}", flush=True)
            continue
        assert not any(dest.exists() for dest in dests), f"Incomplete cache for {name}"
        image, _ = files(split, stem)
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
        raws = [run_dir / f"0.raw.{i}.npy" for i in range(2)]
        arrays = [np.load(path, mmap_mode="r") for path in raws]
        with Image.open(image) as rgb:
            assert rgb.size == (1000, 1000), (image, rgb.size)
            expected = [(rgb.height, rgb.width, 1), (rgb.height, rgb.width, 2)]
        assert [arr.shape for arr in arrays] == expected, [arr.shape for arr in arrays]
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
        "dataset_manifest_sha256": digest(MANIFEST),
    }


def summarize(rows: list[dict], rng: np.random.Generator) -> dict:
    summary = {}
    for metric in ("raw_dice", "postprocessed_dice"):
        baseline = np.array([row[metric]["argmax"] for row in rows])
        candidate = np.array([row[metric]["rankseg"] for row in rows])
        delta = candidate - baseline
        boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        summary[metric] = {
            "argmax_mean": float(baseline.mean()),
            "rankseg_mean": float(candidate.mean()),
            "mean_gain": float(delta.mean()),
            "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            "wins": int(np.count_nonzero(delta > 1e-12)),
            "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
            "losses": int(np.count_nonzero(delta < -1e-12)),
        }
    return summary


def evaluate(names: list[tuple[str, str]]) -> None:
    import torch
    from rankseg.functional import rankseg
    from tiatoolbox.models.architecture.hovernet import HoVerNet

    check_sources()
    assert torch.cuda.is_available(), "CUDA required"
    meta = source_meta()
    partial = BASE / "results.partial.json"
    state = json.loads(partial.read_text()) if partial.exists() else {"source": meta, "cases": []}
    assert state["source"] == meta, "Partial results use different source code, data, or weights"
    completed = {row["id"] for row in state["cases"]}
    assert len(completed) == len(state["cases"])
    for index, (split, stem) in enumerate(names, 1):
        name = key(split, stem)
        if name in completed:
            print(f"[{index}/{len(names)}] already scored {name}", flush=True)
            continue
        np_map, hv_map = [
            np.load(BASE / "probabilities" / f"{name}.{head}.npy", mmap_mode="r") for head in ("np", "hv")
        ]
        assert np_map.shape == (1000, 1000, 1)
        assert hv_map.shape == (1000, 1000, 2)
        _, label = files(split, stem)
        gt = loadmat(label)["inst_map"]
        assert gt.shape == (1000, 1000), (name, gt.shape)
        argmax_mask = np_map[..., 0] >= 0.5
        baseline, _ = HoVerNet.postproc([np_map, hv_map])
        baseline_binary, _ = HoVerNet.postproc([argmax_mask[..., None].astype(np.uint8), hv_map])
        assert np.array_equal(baseline, baseline_binary), f"Official baseline mismatch for {name}"

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
        rankseg_inst, _ = HoVerNet.postproc([rankseg_mask[..., None].astype(np.uint8), hv_map])
        row = {
            "id": name,
            "split": split,
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
    assert {row["id"] for row in state["cases"]} == {key(*item) for item in names}
    state["cases"].sort(key=lambda row: (row["split"], row["id"]))
    rng = np.random.default_rng(20261018)
    state["summary"] = {
        "n": len(names),
        "cohort": "HoVer-Net authors' Kumar test_same (8) and test_diff (6)",
        "score_rule": "Binary nucleus Dice on inst_map > 0 per image, then mean over images",
        "decoder_rule": "Same nucleus probabilities; argmax threshold >=0.5 versus RankSEG 0.0.8 Dice/RMA/multilabel with safe_screening=True; HV map and official HoVer-Net postproc shared",
        "overall": summarize(state["cases"], rng),
        "by_split": {
            split: summarize([row for row in state["cases"] if row["split"] == split], rng)
            for split in ("test_same", "test_diff")
        },
    }
    (BASE / "results.json").write_text(json.dumps(state, indent=2) + "\n")
    with (BASE / "per_case.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "split", "raw_argmax", "raw_rankseg", "post_argmax", "post_rankseg"))
        for row in state["cases"]:
            writer.writerow(
                (
                    row["id"],
                    row["split"],
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
    parser.add_argument("step", choices=("download", "infer", "evaluate"))
    parser.add_argument("--limit", type=int, choices=range(1, 15))
    args = parser.parse_args()
    names = cases(args.limit) if args.step != "download" else []
    {"download": download, "infer": lambda: infer(names), "evaluate": lambda: evaluate(names)}[args.step]()
