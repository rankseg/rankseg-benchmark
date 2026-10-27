"""MicroNet/CoNSeP migration check using TIAToolbox 2.1.3's public engine.

Run ``infer``, ``evaluate``, then ``report``. Inputs are shared with the frozen
1.6 benchmark; new probabilities and predictions live in a separate artifact
directory. Neither inference nor decoding receives the ground truth.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.io import loadmat

from . import micronet_consep as old
from .micronet_kumar import instance_scores
from .provenance import decoder_source

BASE = old.ROOT / "artifacts" / "tiatoolbox_v2" / "micronet_consep"
METHODS = ("argmax", "foreground", "multiclass")
METRICS = ("raw_dice", "dice", "aji", "pq")
TIA_VERSION = "2.1.3"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inputs() -> list[dict]:
    if old.digest(old.CHECKPOINT) != old.CHECKPOINT_SHA256:
        raise ValueError("Checkpoint changed")
    if old.digest(old.BASE / "consep_kaggle.zip") != old.DATA_SHA256:
        raise ValueError("CoNSeP archive changed")
    return [
        {
            "id": name,
            "image_sha256": old.digest(old.TEST / "Images" / f"{name}.png"),
            "label_sha256": old.digest(old.TEST / "Labels" / f"{name}.mat"),
        }
        for name in old.cases()
    ]


def runtime() -> dict:
    import tiatoolbox
    import torch

    if tiatoolbox.__version__ != TIA_VERSION:
        raise ValueError(f"Expected TIAToolbox {TIA_VERSION}, found {tiatoolbox.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")
    packages = (
        "tiatoolbox",
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-image",
        "pillow",
        "dask",
        "zarr",
        "rankseg",
        "triton",
    )
    package_root = Path(tiatoolbox.__file__).parent
    files = (
        "models/architecture/micronet.py",
        "models/engine/multi_task_segmentor.py",
        "models/engine/engine_abc.py",
        "models/dataset/dataset_abc.py",
        "data/pretrained_model.yaml",
    )
    torch.set_num_threads(4)
    torch.manual_seed(0)
    return {
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "tiatoolbox_source_sha256": {p: old.digest(package_root / p) for p in files},
    }


def validate_probabilities(probs: np.ndarray) -> None:
    if probs.shape != (1000, 1000, 2) or probs.dtype != np.float32:
        raise ValueError(f"Unexpected probability shape/dtype: {probs.shape}, {probs.dtype}")
    if not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1:
        raise ValueError("Model output is not finite probabilities")
    if not np.allclose(probs.sum(axis=-1), 1, atol=1e-5, rtol=0):
        raise ValueError("Class probabilities do not sum to one")


def make_engine():
    from tiatoolbox import rcParam
    from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

    rcParam["torch_compile_mode"] = "disable"
    return MultiTaskSegmentor(
        model="micronet-consep", weights=old.CHECKPOINT, batch_size=4, num_workers=0, device="cuda", verbose=False
    )


def infer(limit: int | None) -> None:
    import torch
    import zarr

    meta = {
        "runtime": runtime(),
        "inputs": inputs(),
        "checkpoint_sha256": old.CHECKPOINT_SHA256,
        "engine": "MultiTaskSegmentor",
        "patch_mode": False,
        "auto_get_mask": False,
        "wsireader_kwargs": {"mpp": 0.25},
        "batch_size": 4,
        "num_workers": 0,
        "torch_compile_mode": "disable",
        "ioconfig": "registered micronet-consep defaults",
    }
    meta_path = BASE / "inference_manifest.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("Cached inference uses different inputs or runtime")
    write_json(meta_path, meta)
    engine = make_engine()
    for row in meta["inputs"][:limit]:
        name = row["id"]
        dest = BASE / "probabilities" / f"{name}.npy"
        pred_dest = BASE / "official_instances" / f"{name}.npy"
        case_meta = BASE / "inference" / f"{name}.json"
        if case_meta.exists():
            cached = json.loads(case_meta.read_text())
            if (
                old.digest(dest) != cached["probabilities_sha256"]
                or old.digest(pred_dest) != cached["instances_sha256"]
            ):
                raise ValueError(f"Cached output changed: {name}")
            print(f"cached {name}", flush=True)
            continue
        run_dir = BASE / "inference" / name
        image_path = old.TEST / "Images" / f"{name}.png"
        saved_output = run_dir / f"{name}.zarr"
        elapsed = None
        if run_dir.exists():
            if not saved_output.is_dir():
                raise FileExistsError(f"Inspect incomplete inference before retry: {run_dir}")
            # Recover an interrupted export without repeating model inference.
            # Probability checks below and exact postproc replay in evaluate()
            # reject incomplete or inconsistent saved arrays.
        else:
            torch.cuda.synchronize()
            start = time.monotonic()
            output = engine.run(
                images=[image_path],
                patch_mode=False,
                save_dir=run_dir,
                auto_get_mask=False,
                wsireader_kwargs={"mpp": 0.25},
                return_probabilities=True,
                return_predictions=(True,),
                output_type="zarr",
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            saved_output = output[image_path]
        group = zarr.open(saved_output, mode="r")
        probs = np.asarray(group["probabilities"]["0"])
        # MultiTaskSegmentor flattens the task group for a single-task model.
        predictions = np.asarray(group["predictions"])
        validate_probabilities(probs)
        if predictions.shape != probs.shape[:2]:
            raise ValueError(f"Instance map has wrong shape: {predictions.shape}")
        for path, array in ((dest, probs), (pred_dest, predictions)):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array)
        write_json(
            case_meta,
            {
                "id": name,
                "inference_seconds": elapsed,
                "probabilities_sha256": old.digest(dest),
                "instances_sha256": old.digest(pred_dest),
                "probability_min": float(probs.min()),
                "probability_max": float(probs.max()),
                "max_sum_error": float(np.abs(probs.sum(-1) - 1).max()),
            },
        )
        print(f"inferred {name}: {elapsed:.2f}s" if elapsed is not None else f"recovered {name}", flush=True)


def postprocess(model, probabilities: np.ndarray) -> np.ndarray:
    return model.postproc([probabilities], verbose=False)[0]["predictions"]


def one_hot(mask: np.ndarray) -> np.ndarray:
    return np.stack((~mask, mask), axis=-1).astype(np.uint8)


def fill_instances(instances: np.ndarray) -> np.ndarray:
    """Reproduce the 1.6 hole-filling step for a diagnostic, never the primary score."""
    canvas = np.zeros_like(instances)
    for label, region in enumerate(ndimage.find_objects(instances), 1):
        if region is not None:
            filled = ndimage.binary_fill_holes(instances[region] == label)
            canvas[region][filled] = label
    return canvas


def summarize(rows: list[dict]) -> dict:
    rng = np.random.default_rng(20261026)
    summaries = {}
    for metric in METRICS:
        baseline = np.array([r["scores"]["argmax"][metric] for r in rows])
        summaries[metric] = {}
        for method in METHODS:
            values = np.array([r["scores"][method][metric] for r in rows])
            delta = values - baseline
            boot = rng.choice(delta, size=(10000, len(rows)), replace=True).mean(axis=1)
            summaries[metric][method] = {
                "mean": float(values.mean()),
                "gain": float(delta.mean()),
                "paired_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "wins": int((delta > 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
            }
    return summaries


def evaluate(limit: int | None) -> None:
    import torch
    from rankseg.functional import rankseg
    from tiatoolbox.models.architecture.micronet import MicroNet
    from tiatoolbox.utils.metrics import dice

    manifest = json.loads((BASE / "inference_manifest.json").read_text())
    if manifest["runtime"] != runtime() or manifest["inputs"] != inputs():
        raise ValueError("Inference provenance no longer matches")
    source = {
        "inference_manifest_sha256": old.digest(BASE / "inference_manifest.json"),
        "rankseg": decoder_source(),
        "metric": "dice",
        "solver": "RMA",
        "safe_screening": True,
        "pruning_prob": 0.5,
        "smooth": 0.0,
    }
    path = BASE / "results.json"
    state = json.loads(path.read_text()) if path.exists() else {"source": source, "cases": []}
    if state["source"] != source:
        raise ValueError("Previous evaluation uses different provenance")
    completed = {row["id"] for row in state["cases"]}
    model = MicroNet()  # postproc uses no weights
    legacy_path = old.BASE / "results.json"
    legacy = {row["id"]: row for row in json.loads(legacy_path.read_text())["cases"]} if legacy_path.exists() else {}
    for item in manifest["inputs"][:limit]:
        name = item["id"]
        if name in completed:
            continue
        pred_path = BASE / "official_instances" / f"{name}.npy"
        probs_path = BASE / "probabilities" / f"{name}.npy"
        cached = json.loads((BASE / "inference" / f"{name}.json").read_text())
        if (
            old.digest(probs_path) != cached["probabilities_sha256"]
            or old.digest(pred_path) != cached["instances_sha256"]
        ):
            raise ValueError(f"Inference output changed: {name}")
        probs = np.load(probs_path)
        validate_probabilities(probs)
        gt = loadmat(old.TEST / "Labels" / f"{name}.mat")["inst_map"]
        if gt.shape != probs.shape[:2] or not np.any(gt):
            raise ValueError(f"Unexpected ground truth: {name}")
        masks = {"argmax": probs.argmax(-1).astype(bool)}
        direct = postprocess(model, probs)
        if not np.array_equal(direct, np.load(pred_path)):
            raise ValueError("Direct postproc differs from official engine instances")
        if not np.array_equal(direct, postprocess(model, one_hot(masks["argmax"]))):
            raise ValueError("Binary-mask adapter changed official argmax postprocessing")
        timings = {}
        for method in METHODS[1:]:
            channels = probs[..., 1:2] if method == "foreground" else probs
            tensor = torch.from_numpy(channels.copy()).permute(2, 0, 1).unsqueeze(0).cuda()
            torch.cuda.synchronize()
            start = time.monotonic()
            decoded = rankseg(
                tensor,
                metric="dice",
                solver="RMA",
                safe_screening=True,
                pruning_prob=0.5,
                smooth=0,
                output_mode="multilabel" if method == "foreground" else "multiclass",
            )
            torch.cuda.synchronize()
            timings[method] = time.monotonic() - start
            masks[method] = (decoded[0, 0] if method == "foreground" else decoded[0]).cpu().numpy().astype(bool)
        scores, filled_scores, counts = {}, {}, {}
        for method, mask in masks.items():
            pred = direct if method == "argmax" else postprocess(model, one_hot(mask))
            pred_dir = BASE / "evaluated_instances" / name
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.save(pred_dir / f"{method}.npy", pred)
            scores[method] = {
                "raw_dice": float(dice(gt > 0, mask)),
                "dice": float(dice(gt > 0, pred > 0)),
                **instance_scores(gt, pred),
            }
            filled_scores[method] = float(dice(gt > 0, fill_instances(pred) > 0))
            counts[method] = {
                "tp": int(((gt > 0) & (pred > 0)).sum()),
                "fp": int(((gt == 0) & (pred > 0)).sum()),
                "fn": int(((gt > 0) & (pred == 0)).sum()),
            }
        # A fresh 2.x run needs only the images, labels and checkpoint.
        # Replay the historical baseline when its optional artifacts exist.
        previous_path = old.BASE / "probabilities" / f"{name}.npy"
        diagnostic = {"v2_probs_filled_dice": filled_scores}
        if name in legacy and previous_path.exists():
            previous = np.load(previous_path)
            old_post = postprocess(model, previous)
            diagnostic.update(
                {
                    "v1_argmax_dice": legacy[name]["postprocessed_dice"]["argmax"],
                    "v1_probs_v2_post_dice": float(dice(gt > 0, old_post > 0)),
                    "v1_probs_filled_dice": float(dice(gt > 0, fill_instances(old_post) > 0)),
                    "probability_mean_abs_diff": float(np.abs(probs - previous).mean()),
                    "probability_max_abs_diff": float(np.abs(probs - previous).max()),
                    "raw_argmax_disagreement_pixels": int((masks["argmax"] != previous.argmax(-1)).sum()),
                }
            )
            if abs(diagnostic["v1_probs_filled_dice"] - diagnostic["v1_argmax_dice"]) > 1e-12:
                raise ValueError("Could not replay frozen 1.6 argmax postprocessing")
        state["cases"].append(
            {
                "id": name,
                "scores": scores,
                "counts": counts,
                "migration_diagnostics": diagnostic,
                "decode_seconds": timings,
                "official_engine_and_adapter_exact_match": True,
            }
        )
        write_json(path, state)
        print(f"scored {name}: " + str({m: round(scores[m]["dice"], 6) for m in METHODS}), flush=True)
    state["cases"].sort(key=lambda row: int(row["id"].split("_")[1]))
    state["complete"] = {r["id"] for r in state["cases"]} == {r["id"] for r in manifest["inputs"]}
    state["summary"] = summarize(state["cases"])
    write_json(path, state)
    fields = ["id"] + [f"{method}_{metric}" for method in METHODS for metric in METRICS]
    with (BASE / "per_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in state["cases"]:
            writer.writerow({"id": row["id"], **{f"{m}_{k}": row["scores"][m][k] for m in METHODS for k in METRICS}})


def report(*, from_evidence: bool = False) -> None:
    if from_evidence:
        evidence = Path(__file__).parent / "evidence" / "micronet_consep_v2.csv"
        with evidence.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [row["id"] for row in rows] != [f"test_{i}" for i in range(1, 15)]:
            raise ValueError("Expected all 14 test images in numeric order")
        cases = [
            {
                "id": row["id"],
                "scores": {
                    method: {metric: float(row[f"{method}_{metric}"]) for metric in METRICS} for method in METHODS
                },
            }
            for row in rows
        ]
        state = {"cases": cases, "complete": True, "summary": summarize(cases)}
    else:
        state = json.loads((BASE / "results.json").read_text())
    print(f"TIAToolbox {TIA_VERSION}, n={len(state['cases'])}, complete={state['complete']}")
    print("| Endpoint | Argmax | RankSEG foreground | Gain (pp) | 95% CI (pp) | Improved |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for metric in METRICS:
        base = state["summary"][metric]["argmax"]
        rank = state["summary"][metric]["foreground"]
        lo, hi = np.array(rank["paired_ci95"]) * 100
        print(
            f"| {metric} | {base['mean']:.6f} | {rank['mean']:.6f} | {rank['gain'] * 100:+.3f} "
            f"| {lo:+.3f} to {hi:+.3f} | {rank['wins']}/{len(state['cases'])} |"
        )


def audit_metrics(scorer_path: Path, limit: int | None) -> None:
    """Compare recorded scores with the unmodified authors' HoVer-Net scorer."""
    spec = importlib.util.spec_from_file_location("hovernet_reference_stats", scorer_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import reference scorer: {scorer_path}")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    state = json.loads((BASE / "results.json").read_text())
    checks = []
    for row in state["cases"][:limit]:
        gt = loadmat(old.TEST / "Labels" / f"{row['id']}.mat")["inst_map"]
        truth = reference.remap_label(gt)
        for method in METHODS:
            pred = np.load(BASE / "evaluated_instances" / row["id"] / f"{method}.npy")
            pred = reference.remap_label(pred)
            official = {
                "dice": float(reference.get_dice_1(truth, pred)),
                "aji": float(reference.get_fast_aji(truth, pred)),
                "pq": float(reference.get_fast_pq(truth, pred)[0][2]),
            }
            error = {k: abs(v - row["scores"][method][k]) for k, v in official.items()}
            if max(error.values()) > 1e-10:
                raise ValueError(f"Metric mismatch: {row['id']}, {method}, {error}")
            checks.append({"id": row["id"], "method": method, "reference_scores": official, "absolute_errors": error})
    write_json(BASE / "metric_audit.json", {"scorer_sha256": old.digest(scorer_path), "checks": checks})
    print(f"Verified {len(checks)} predictions against the authors' Dice/AJI/PQ scorer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("infer", "evaluate", "report", "audit"))
    parser.add_argument("--limit", type=int, default=None, help="Pilot on the first N numeric test IDs")
    parser.add_argument("--from-evidence", action="store_true", help="Rebuild the report from tracked per-case CSVs")
    parser.add_argument("--reference-scorer", type=Path, default=BASE.parent / "hovernet_stats_utils.py")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 14:
        parser.error("--limit must be between 1 and 14")
    if args.from_evidence and args.step != "report":
        parser.error("--from-evidence applies only to report")
    if args.step == "report":
        report(from_evidence=args.from_evidence)
    elif args.step == "audit":
        audit_metrics(args.reference_scorer, args.limit)
    else:
        {"infer": infer, "evaluate": evaluate}[args.step](args.limit)
