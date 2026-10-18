"""Cross-dataset MicroNet/CoNSeP evaluation on Kumar's 14 held-out images.

Inference uses the registered TIAToolbox model and saves probabilities once.
Both decoders use those probabilities and the same MicroNet postprocessor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .provenance import runtime_source

from . import hovernet_kumar as kumar
from . import micronet_consep as micronet


BASE = micronet.ROOT / "artifacts" / "tiatoolbox_micronet_kumar"
CHECKPOINT = micronet.CHECKPOINT


def check_sources() -> None:
    import tiatoolbox

    assert tiatoolbox.__version__ == "1.6.0", tiatoolbox.__version__
    assert micronet.digest(CHECKPOINT) == micronet.CHECKPOINT_SHA256
    manifest = json.loads(kumar.MANIFEST.read_text())
    kumar.check_manifest_index(manifest)
    for row in manifest["files"]:
        path = kumar.DATA / row["path"]
        assert path.stat().st_size == row["bytes"] and kumar.digest(path) == row["sha256"], path


def source_meta() -> dict:
    return {
        **runtime_source(),
        "checkpoint_sha256": micronet.digest(CHECKPOINT),
        "dataset_manifest_sha256": kumar.digest(kumar.MANIFEST),
    }


def infer(names: list[tuple[str, str]]) -> None:
    import torch
    from tiatoolbox import rcParam
    from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

    check_sources()
    assert torch.cuda.is_available(), "CUDA required"
    rcParam["torch_compile_mode"] = "disable"
    model = SemanticSegmentor(
        pretrained_model="micronet-consep",
        pretrained_weights=str(CHECKPOINT),
        batch_size=4,
        num_loader_workers=0,
        verbose=False,
    )
    out_dir = BASE / "probabilities"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, (split, stem) in enumerate(names, 1):
        name = kumar.key(split, stem)
        dest = out_dir / f"{name}.npy"
        if dest.exists():
            print(f"[{index}/{len(names)}] cached {name}", flush=True)
            continue
        image, _ = kumar.files(split, stem)
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
        raw = run_dir / "0.raw.0.npy"
        probs = np.load(raw, mmap_mode="r")
        with Image.open(image) as rgb:
            assert rgb.size == (1000, 1000)
            assert probs.shape == (rgb.height, rgb.width, 2), probs.shape
        assert np.isfinite(probs).all()
        assert np.allclose(probs.sum(axis=-1), 1, atol=1e-5)
        del probs
        os.replace(raw, dest)
        print(f"[{index}/{len(names)}] {name}: {time.monotonic() - start:.1f}s", flush=True)


def summarize(rows: list[dict], rng: np.random.Generator) -> dict:
    results = {}
    for metric in ("raw_dice", "postprocessed_dice"):
        baseline = np.array([row[metric]["argmax"] for row in rows])
        results[metric] = {}
        for method in ("foreground", "multiclass"):
            candidate = np.array([row[metric][method] for row in rows])
            delta = candidate - baseline
            boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
            results[metric][method] = {
                "argmax_mean": float(baseline.mean()),
                "rankseg_mean": float(candidate.mean()),
                "mean_gain": float(delta.mean()),
                "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
                "wins": int(np.count_nonzero(delta > 1e-12)),
                "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
                "losses": int(np.count_nonzero(delta < -1e-12)),
            }
    return results


def instance_scores(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """AJI and PQ at IoU>0.5, matching the HoVer-Net instance scorer."""
    true_ids, true_counts = np.unique(gt[gt > 0], return_counts=True)
    pred_ids, pred_counts = np.unique(pred[pred > 0], return_counts=True)
    if not len(true_ids) or not len(pred_ids):
        return {"aji": 0.0, "pq": 0.0}
    overlap = (gt > 0) & (pred > 0)
    ti = np.searchsorted(true_ids, gt[overlap])
    pi = np.searchsorted(pred_ids, pred[overlap])
    codes, intersections = np.unique(ti * len(pred_ids) + pi, return_counts=True)
    ti, pi = codes // len(pred_ids), codes % len(pred_ids)
    unions = true_counts[ti] + pred_counts[pi] - intersections
    ious = intersections / unions

    # With IoU > 0.5, valid pairs are one-to-one without an assignment solver.
    matched = ious > 0.5
    assert len(np.unique(ti[matched])) == len(np.unique(pi[matched])) == int(matched.sum())
    tp = int(matched.sum())
    fp = len(pred_ids) - tp
    fn = len(true_ids) - tp
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = float(ious[matched].sum()) / (tp + 1e-6)

    # Original AJI selects the highest-IoU prediction for each true instance;
    # a prediction may be reused. Unpaired areas enter the denominator.
    order = np.lexsort((pi, -ious, ti))
    selected = order[np.unique(ti[order], return_index=True)[1]]
    aji_inter = float(intersections[selected].sum())
    aji_union = float(unions[selected].sum())
    aji_union += float(true_counts.sum() - true_counts[ti[selected]].sum())
    aji_union += float(pred_counts.sum() - pred_counts[np.unique(pi[selected])].sum())
    return {"aji": aji_inter / aji_union, "pq": dq * sq}


def evaluate_instances(names: list[tuple[str, str]]) -> None:
    """Add instance metrics to the frozen Dice run without changing its scores."""
    import torch
    from rankseg.functional import rankseg
    from tiatoolbox.models.architecture.micronet import MicroNet

    check_sources()
    assert torch.cuda.is_available(), "CUDA required"
    results_path = BASE / "results.json"
    state = json.loads(results_path.read_text())
    assert state["source"] == source_meta()
    rows = {row["id"]: row for row in state["cases"]}
    assert set(rows) == {kumar.key(*item) for item in kumar.cases()}
    partial = BASE / "instances.partial.json"
    extra = json.loads(partial.read_text()) if partial.exists() else {}
    for index, (split, stem) in enumerate(names, 1):
        name = kumar.key(split, stem)
        if name in extra:
            print(f"[{index}/{len(names)}] already scored {name}", flush=True)
            continue
        probs = np.load(BASE / "probabilities" / f"{name}.npy", mmap_mode="r")
        gt = loadmat(kumar.files(split, stem)[1])["inst_map"]
        argmax = np.argmax(probs, axis=-1).astype(bool)
        masks = {"argmax": argmax}
        for mode in ("foreground", "multiclass"):
            channels = np.asarray(probs[..., 1:2] if mode == "foreground" else probs).copy()
            tensor = torch.from_numpy(channels).permute(2, 0, 1).unsqueeze(0).to("cuda")
            decoded = rankseg(
                tensor,
                metric="dice",
                solver="RMA",
                output_mode="multilabel" if mode == "foreground" else "multiclass",
                safe_screening=True,
            )
            masks[mode] = (
                decoded[0, 0].cpu().numpy().astype(bool)
                if mode == "foreground"
                else decoded[0].cpu().numpy().astype(bool)
            )
            del channels, tensor, decoded
        preds = {"argmax": MicroNet.postproc([probs])[0]}
        preds.update({mode: micronet.postprocess_binary(masks[mode]) for mode in ("foreground", "multiclass")})
        for method, pred in preds.items():
            assert abs(micronet.dice(gt, pred) - rows[name]["postprocessed_dice"][method]) < 1e-8, name
        extra[name] = {method: instance_scores(gt, pred) for method, pred in preds.items()}
        partial.write_text(json.dumps(extra, indent=2) + "\n")
        print(f"[{index}/{len(names)}] {name}: {extra[name]}", flush=True)
    if len(names) < 14:
        return
    assert set(extra) == set(rows)
    for name, metrics in extra.items():
        rows[name]["instance_metrics"] = metrics
    state["summary"]["instance_metrics"] = {}
    for metric in ("aji", "pq"):
        state["summary"]["instance_metrics"][metric] = {}
        for method in ("argmax", "foreground", "multiclass"):
            scores = np.array([rows[kumar.key(*item)]["instance_metrics"][method][metric] for item in kumar.cases()])
            baseline = np.array(
                [rows[kumar.key(*item)]["instance_metrics"]["argmax"][metric] for item in kumar.cases()]
            )
            delta = scores - baseline
            state["summary"]["instance_metrics"][metric][method] = {
                "mean": float(scores.mean()),
                "mean_gain": float(delta.mean()),
                "wins": int(np.count_nonzero(delta > 1e-12)),
                "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
                "losses": int(np.count_nonzero(delta < -1e-12)),
            }
    results_path.write_text(json.dumps(state, indent=2) + "\n")
    partial.unlink()
    print(json.dumps(state["summary"]["instance_metrics"], indent=2), flush=True)


def evaluate(names: list[tuple[str, str]]) -> None:
    import torch
    from rankseg.functional import rankseg
    from tiatoolbox.models.architecture.micronet import MicroNet

    check_sources()
    assert torch.cuda.is_available(), "CUDA required"
    BASE.mkdir(parents=True, exist_ok=True)
    meta = source_meta()
    partial = BASE / "results.partial.json"
    state = json.loads(partial.read_text()) if partial.exists() else {"source": meta, "cases": []}
    assert state["source"] == meta, "Partial results use different data, weights, or source"
    completed = {row["id"] for row in state["cases"]}
    assert len(completed) == len(state["cases"])
    for index, (split, stem) in enumerate(names, 1):
        name = kumar.key(split, stem)
        if name in completed:
            print(f"[{index}/{len(names)}] already scored {name}", flush=True)
            continue
        probs = np.load(BASE / "probabilities" / f"{name}.npy", mmap_mode="r")
        assert probs.shape == (1000, 1000, 2)
        _, label = kumar.files(split, stem)
        gt = loadmat(label)["inst_map"]
        assert gt.shape == (1000, 1000)
        argmax = np.argmax(probs, axis=-1).astype(bool)
        baseline = MicroNet.postproc([probs])[0]
        baseline_binary = micronet.postprocess_binary(argmax)
        assert np.array_equal(baseline, baseline_binary), f"Official baseline mismatch: {name}"

        masks = {"argmax": argmax}
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
            masks[mode] = (
                decoded[0, 0].cpu().numpy().astype(bool)
                if mode == "foreground"
                else decoded[0].cpu().numpy().astype(bool)
            )
            assert masks[mode].shape == (1000, 1000)
            del channels, tensor, decoded

        posts = {"argmax": baseline}
        posts.update({mode: micronet.postprocess_binary(masks[mode]) for mode in ("foreground", "multiclass")})
        row = {
            "id": name,
            "split": split,
            "raw_dice": {method: micronet.dice(gt, mask) for method, mask in masks.items()},
            "postprocessed_dice": {method: micronet.dice(gt, pred) for method, pred in posts.items()},
            "gt_pixels": int(np.count_nonzero(gt)),
            "postprocessed_pred_pixels": {method: int(np.count_nonzero(pred)) for method, pred in posts.items()},
            "rankseg_decode_seconds": times,
        }
        state["cases"].append(row)
        partial.write_text(json.dumps(state, indent=2) + "\n")
        print(f"[{index}/{len(names)}] {name}: {row['postprocessed_dice']}", flush=True)
    if len(names) < 14:
        return
    assert {row["id"] for row in state["cases"]} == {kumar.key(*item) for item in names}
    state["cases"].sort(key=lambda row: (row["split"], row["id"]))
    rng = np.random.default_rng(20261018)
    state["summary"] = {
        "n": 14,
        "cohort": "Kumar author test_same (8) and test_diff (6); cross-dataset MicroNet/CoNSeP checkpoint",
        "score_rule": "Binary nucleus Dice on inst_map > 0 per image, then mean over images; same official MicroNet instance postprocessing for both decoders",
        "decoder_rule": "Same 2-channel probabilities; argmax versus RankSEG 0.0.8 Dice/RMA, foreground multilabel and 2-class multiclass, safe_screening=True",
        "overall": summarize(state["cases"], rng),
        "by_split": {
            split: summarize([row for row in state["cases"] if row["split"] == split], rng)
            for split in ("test_same", "test_diff")
        },
    }
    (BASE / "results.json").write_text(json.dumps(state, indent=2) + "\n")
    with (BASE / "per_case.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "id",
                "split",
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
                    row["split"],
                    *(row["raw_dice"][method] for method in ("argmax", "foreground", "multiclass")),
                    *(row["postprocessed_dice"][method] for method in ("argmax", "foreground", "multiclass")),
                )
            )
    partial.unlink()
    print(json.dumps(state["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("infer", "evaluate", "instances"))
    parser.add_argument("--limit", type=int, choices=range(1, 15))
    args = parser.parse_args()
    names = kumar.cases(args.limit)
    {
        "infer": lambda: infer(names),
        "evaluate": lambda: evaluate(names),
        "instances": lambda: evaluate_instances(names),
    }[args.step]()
