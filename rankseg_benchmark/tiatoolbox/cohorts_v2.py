"""Repeat the remaining frozen TIAToolbox cohorts using the 2.1.3 engines.

Run ``infer``, ``evaluate``, and ``report`` for each cohort. Model inputs and
weights are shared with the historical runs; outputs are isolated in artifacts.
``padding`` is an optional controlled replay requiring both probability caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from . import bcss
from . import hovernet_consep as hc
from . import hovernet_kumar as hk
from . import micronet_consep as mc
from . import micronet_consep_v2 as shared
from . import micronet_kumar as mk
from .provenance import decoder_source

COHORTS = {
    "micronet_kumar": ("micronet-consep", mc.CHECKPOINT, mc.CHECKPOINT_SHA256, mk.BASE, 14),
    "hovernet_consep": ("hovernet_original-consep", hc.CHECKPOINT, hc.CHECKPOINT_SHA256, hc.BASE, 14),
    "hovernet_kumar": ("hovernet_original-kumar", hk.CHECKPOINT, hk.CHECKPOINT_SHA256, hk.BASE, 14),
    "bcss": ("fcn_resnet50_unet-bcss", bcss.CHECKPOINT, bcss.CHECKPOINT_SHA256, bcss.BASE, 45),
}
BASE = shared.BASE.parent


def methods(cohort: str) -> tuple[str, ...]:
    extra = ("multiclass",) if cohort.startswith("micronet") else (("multilabel",) if cohort == "bcss" else ())
    return ("argmax", "rankseg", *extra)


def inputs(cohort: str) -> list[dict]:
    _, checkpoint, checksum, _, expected = COHORTS[cohort]
    if mc.digest(checkpoint) != checksum:
        raise ValueError("Checkpoint checksum mismatch")
    rows = []
    if cohort.endswith("kumar"):
        manifest = json.loads(hk.MANIFEST.read_text())
        hk.check_manifest_index(manifest)
        for item in manifest["files"]:
            if mc.digest(hk.DATA / item["path"]) != item["sha256"]:
                raise ValueError("Kumar input differs from frozen manifest")
        rows = [
            {"id": hk.key(s, n), "image": str(hk.files(s, n)[0]), "label": str(hk.files(s, n)[1])}
            for s, n in hk.cases()
        ]
    elif cohort == "hovernet_consep":
        if mc.digest(mc.BASE / "consep_kaggle.zip") != mc.DATA_SHA256:
            raise ValueError("CoNSeP archive checksum mismatch")
        rows = [
            {"id": n, "image": str(mc.TEST / "Images" / f"{n}.png"), "label": str(mc.TEST / "Labels" / f"{n}.mat")}
            for n in mc.cases()
        ]
    else:
        if mc.digest(bcss.CSV) != "564bc31fdd5808e5d57de40cc65541aad2c1dbc9415e2e0448f25c00ed144d28":
            raise ValueError("BCSS cohort index changed")
        rows = [
            {
                "id": r[""],
                "image": str(bcss.BASE / "data" / f"{r['']}.jpg"),
                "label": str(bcss.BASE / "data" / f"{r['']}.mask.png"),
            }
            for r in bcss.cohort()
        ]
    if len(rows) != expected:
        raise ValueError("Unexpected cohort size")
    for row in rows:
        for key in ("image", "label"):
            path = Path(row[key])
            row[f"{key}_sha256"] = mc.digest(path)
            row[key] = str(path.relative_to(mc.ROOT))
        with Image.open(mc.ROOT / row["image"]) as image:
            row["shape"] = [image.height, image.width]
    return rows


def runtime() -> dict:
    import tiatoolbox

    value = shared.runtime()
    root = Path(tiatoolbox.__file__).parent
    for file in (
        "models/architecture/hovernet.py",
        "models/architecture/unet.py",
        "models/engine/semantic_segmentor.py",
        "models/engine/patch_predictor.py",
    ):
        value["tiatoolbox_source_sha256"][file] = mc.digest(root / file)
    return value


def make_engine(cohort: str):
    from tiatoolbox import rcParam
    from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor
    from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

    rcParam["torch_compile_mode"] = "disable"
    name, checkpoint, _, _, _ = COHORTS[cohort]
    engine = SemanticSegmentor if cohort == "bcss" else MultiTaskSegmentor
    return engine(
        model=name,
        weights=checkpoint,
        batch_size=2 if cohort == "bcss" else 4,
        num_workers=0,
        device="cuda",
        verbose=False,
    )


def validate(cohort: str, arrays: list[np.ndarray], shape: list[int]) -> None:
    channels = (
        [5]
        if cohort == "bcss"
        else ([2] if cohort.startswith("micronet") else ([1, 2, 1] if cohort == "hovernet_consep" else [1, 2]))
    )
    if len(arrays) != len(channels):
        raise ValueError("Wrong number of model outputs")
    for arr, count in zip(arrays, channels, strict=True):
        if arr.shape != (*shape, count) or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid output shape/dtype: {arr.shape}, {arr.dtype}")
    probs = arrays[0]
    if probs.min() < 0 or probs.max() > 1:
        raise ValueError("Output is not probabilities")
    if channels[0] > 1 and not np.allclose(probs.sum(-1), 1, atol=1e-5, rtol=0):
        raise ValueError("Probability channels do not sum to one")


def infer(cohort: str, limit: int | None) -> None:
    import torch
    import zarr

    dest = BASE / cohort
    meta = {
        "runtime": runtime(),
        "inputs": inputs(cohort),
        "model": COHORTS[cohort][0],
        "checkpoint_sha256": COHORTS[cohort][2],
        "batch_size": 2 if cohort == "bcss" else 4,
        "patch_mode": False,
        "auto_get_mask": False,
        "mpp": 0.25,
        "torch_compile_mode": "disable",
        "num_workers": 0,
        "ioconfig": "registered defaults",
    }
    manifest = dest / "inference_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != meta:
        raise ValueError("Inference provenance changed")
    shared.write_json(manifest, meta)
    engine = make_engine(cohort)
    for row in meta["inputs"][:limit]:
        name = row["id"]
        record = dest / "inference" / f"{name}.json"
        if record.exists():
            cached = json.loads(record.read_text())
            for file, checksum in cached["outputs_sha256"].items():
                if mc.digest(dest / file) != checksum:
                    raise ValueError(f"Cached output changed: {name}")
            print(f"cached {cohort}/{name}", flush=True)
            continue
        run_dir = dest / "inference" / name
        image_path = mc.ROOT / row["image"]
        saved = run_dir / f"{image_path.stem}.zarr"
        elapsed = None
        if run_dir.exists():
            if not saved.is_dir():
                raise FileExistsError(f"Inspect incomplete run: {run_dir}")
        else:
            torch.cuda.synchronize()
            start = time.monotonic()
            result = engine.run(
                images=[image_path],
                patch_mode=False,
                save_dir=run_dir,
                auto_get_mask=False,
                wsireader_kwargs={"mpp": 0.25},
                return_probabilities=True,
                return_predictions=True if cohort == "bcss" else (True,),
                output_type="zarr",
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            saved = result[image_path]
        group = zarr.open(saved, mode="r")
        if cohort == "bcss":
            arrays = [np.asarray(group["probabilities"])]
        else:
            heads = sorted(group["probabilities"].keys(), key=int)
            arrays = [np.asarray(group["probabilities"][h]) for h in heads]
        validate(cohort, arrays, row["shape"])
        pred = np.asarray(group["predictions"])
        if pred.shape != tuple(row["shape"]):
            raise ValueError(f"Unexpected official prediction shape: {pred.shape}")
        outputs = {}
        for h, arr in enumerate([*arrays, pred]):
            file = f"official_instances/{name}.npy" if h == len(arrays) else f"probabilities/{name}.{h}.npy"
            path = dest / file
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, arr)
            outputs[file] = mc.digest(path)
        shared.write_json(record, {"id": name, "inference_seconds": elapsed, "outputs_sha256": outputs})
        print(f"inferred {cohort}/{name}: {elapsed}s", flush=True)


def get_model(cohort: str):
    from tiatoolbox.models.architecture.hovernet import HoVerNet
    from tiatoolbox.models.architecture.micronet import MicroNet

    if cohort.startswith("micronet"):
        return MicroNet()
    if cohort.startswith("hovernet"):
        return HoVerNet(num_types=5 if cohort == "hovernet_consep" else None, mode="original")
    return None


def postprocess(cohort: str, model, arrays: list[np.ndarray]) -> np.ndarray:
    if cohort == "bcss":
        return arrays[0].argmax(-1)
    if cohort.startswith("micronet"):
        return model.postproc(arrays, verbose=False)[0]["predictions"]
    return model.postproc(arrays)[0]["predictions"]


def adapted(cohort: str, arrays: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    if cohort.startswith("micronet"):
        return [shared.one_hot(mask)]
    return [mask[..., None].astype(np.uint8), *arrays[1:]]


def legacy_arrays(cohort: str, name: str) -> list[np.ndarray]:
    path = COHORTS[cohort][3] / "probabilities"
    suffix = ("np", "hv", "tp") if cohort == "hovernet_consep" else ("np", "hv")
    files = [path / f"{name}.{s}.npy" for s in suffix] if cohort.startswith("hovernet") else [path / f"{name}.npy"]
    return [np.load(p, mmap_mode="r") for p in files] if all(p.exists() for p in files) else []


def truth(cohort: str, row: dict):
    path = mc.ROOT / row["label"]
    if cohort != "bcss":
        return loadmat(path)["inst_map"], None
    with Image.open(path) as image:
        raw = np.asarray(image.resize(tuple(row["shape"][::-1]), resample=Image.Resampling.NEAREST))
    if raw.min() < 0 or raw.max() > 21:
        raise ValueError("Unexpected BCSS label codes")
    return bcss.GT_TO_CLASS[raw], (raw != 0) & (raw != 7)


def scores(cohort: str, gt: np.ndarray, pred: np.ndarray, valid=None, raw=None, multilabel=False) -> dict:
    if cohort != "bcss":
        from tiatoolbox.utils.metrics import dice

        return {
            "dice": float(dice(gt > 0, pred > 0)),
            "raw_dice": float(dice(gt > 0, raw)),
            **mk.instance_scores(gt, pred),
        }
    values = []
    for k in range(5):
        a = valid & (gt == k)
        b = valid & (pred[k] if multilabel else pred == k)
        denom = int(a.sum()) + int(b.sum())
        values.append(1.0 if denom == 0 else 2 * int((a & b).sum()) / denom)
    return {"dice": float(np.mean(values)), **{f"dice_{k}": v for k, v in zip(bcss.CLASSES, values, strict=True)}}


def summarize(rows: list[dict]) -> dict:
    rng = np.random.default_rng(20261026)
    summary = {}
    for metric in rows[0]["scores"]["argmax"]:
        summary[metric] = {}
        base = np.array([r["scores"]["argmax"][metric] for r in rows])
        for method in rows[0]["scores"]:
            values = np.array([r["scores"][method][metric] for r in rows])
            delta = values - base
            boots = rng.choice(delta, size=(10000, len(rows)), replace=True).mean(1)
            summary[metric][method] = {
                "mean": float(values.mean()),
                "gain": float(delta.mean()),
                "paired_ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
                "wins": int((delta > 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
            }
    return summary


def evaluate(cohort: str, limit: int | None) -> None:
    import torch
    from rankseg.functional import rankseg

    dest = BASE / cohort
    manifest = json.loads((dest / "inference_manifest.json").read_text())
    if manifest["runtime"] != runtime() or manifest["inputs"] != inputs(cohort):
        raise ValueError("Inference provenance changed")
    source = {
        "manifest_sha256": mc.digest(dest / "inference_manifest.json"),
        "rankseg": decoder_source(),
        "metric": "dice",
        "solver": "RMA",
        "safe_screening": True,
        "pruning_prob": 0.5,
        "smooth": 0.0,
    }
    path = dest / "results.json"
    state = json.loads(path.read_text()) if path.exists() else {"source": source, "cases": []}
    if state["source"] != source:
        raise ValueError("Decoder provenance changed")
    complete = {r["id"] for r in state["cases"]}
    model = get_model(cohort)
    for row in manifest["inputs"][:limit]:
        name = row["id"]
        if name in complete:
            continue
        record = json.loads((dest / "inference" / f"{name}.json").read_text())
        for file, checksum in record["outputs_sha256"].items():
            if mc.digest(dest / file) != checksum:
                raise ValueError(f"Inference output changed: {file}")
        files = sorted((dest / "probabilities").glob(f"{name}.*.npy"))
        arrays = [np.load(p, mmap_mode="r") for p in files]
        validate(cohort, arrays, row["shape"])
        gt, valid = truth(cohort, row)
        direct = postprocess(cohort, model, arrays)
        if not np.array_equal(direct, np.load(dest / "official_instances" / f"{name}.npy")):
            raise ValueError(f"Direct postproc does not match engine: {name}")
        raw = arrays[0].argmax(-1) if cohort == "bcss" or cohort.startswith("micronet") else arrays[0][..., 0] >= 0.5
        if cohort != "bcss":
            raw = raw.astype(bool)
            if not np.array_equal(direct, postprocess(cohort, model, adapted(cohort, arrays, raw))):
                raise ValueError("Binary adapter changed the baseline")
        case_scores = {"argmax": scores(cohort, gt, direct, valid, raw)}
        times = {}
        for method in methods(cohort)[1:]:
            foreground = cohort.startswith("micronet") and method == "rankseg"
            channels = arrays[0][..., 1:2] if foreground else arrays[0]
            tensor = torch.from_numpy(np.array(channels)).permute(2, 0, 1).unsqueeze(0).cuda()
            output_mode = (
                "multiclass" if method == "multiclass" or (cohort == "bcss" and method == "rankseg") else "multilabel"
            )
            torch.cuda.synchronize()
            start = time.monotonic()
            decoded = rankseg(
                tensor,
                metric="dice",
                solver="RMA",
                safe_screening=True,
                pruning_prob=0.5,
                smooth=0,
                output_mode=output_mode,
            )
            torch.cuda.synchronize()
            times[method] = time.monotonic() - start
            mask = decoded[0].cpu().numpy()
            del tensor, decoded
            torch.cuda.empty_cache()
            if cohort != "bcss":
                mask = (mask[0] if output_mode == "multilabel" else mask).astype(bool)
                pred = postprocess(cohort, model, adapted(cohort, arrays, mask))
            else:
                pred = mask.astype(bool) if method == "multilabel" else mask
            case_scores[method] = scores(cohort, gt, pred, valid, mask, multilabel=method == "multilabel")
        previous = legacy_arrays(cohort, name)
        diagnostic = {}
        if previous:
            old_pred = postprocess(cohort, model, previous)
            old_raw = (
                previous[0].argmax(-1)
                if cohort == "bcss" or cohort.startswith("micronet")
                else previous[0][..., 0] >= 0.5
            )
            diagnostic = {
                "v1_probabilities_v2_post_scores": scores(cohort, gt, old_pred, valid, old_raw),
                "probability_mean_absolute_difference": float(np.abs(arrays[0] - previous[0]).mean()),
                "raw_disagreement_pixels": int((raw != old_raw).sum()),
            }
        state["cases"].append(
            {
                "id": name,
                "scores": case_scores,
                "decode_seconds": times,
                "official_engine_and_adapter_exact_match": True,
                "migration_diagnostics": diagnostic,
            }
        )
        shared.write_json(path, state)
        print(f"scored {cohort}/{name}: " + str({m: round(s["dice"], 6) for m, s in case_scores.items()}), flush=True)
    order = {r["id"]: i for i, r in enumerate(manifest["inputs"])}
    state["cases"].sort(key=lambda r: order[r["id"]])
    state["complete"] = {r["id"] for r in state["cases"]} == set(order)
    state["summary"] = summarize(state["cases"])
    shared.write_json(path, state)
    fields = [f"{m}_{k}" for m, s in state["cases"][0]["scores"].items() for k in s]
    with (dest / "per_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", *fields])
        writer.writeheader()
        for row in state["cases"]:
            writer.writerow(
                {"id": row["id"], **{f"{m}_{k}": v for m, s in row["scores"].items() for k, v in s.items()}}
            )


def padding_audit(cohort: str, limit: int | None, *, legacy_bcss_stride: bool = False) -> None:
    """Change only edge padding, retaining the same model and patch layout."""
    import torch
    from tiatoolbox.models.architecture import get_pretrained_model
    from tiatoolbox.tools.patchextraction import PatchExtractor
    from tiatoolbox.wsicore.wsireader import WSIReader

    env = runtime()
    data = inputs(cohort)
    name, checkpoint, checksum, _, _ = COHORTS[cohort]
    model, config = get_pretrained_model(name, checkpoint)
    model = model.to("cuda").eval()
    stride_shape = (256, 256) if legacy_bcss_stride else config.stride_shape
    batch_size = 2 if cohort == "bcss" else 4
    results = []
    for row in data[:limit]:
        height, width = row["shape"]
        gt, valid = truth(cohort, row)
        bounds, outputs = PatchExtractor.get_coordinates(
            image_shape=(width, height),
            patch_input_shape=config.patch_input_shape,
            patch_output_shape=config.patch_output_shape,
            stride_shape=stride_shape,
        )
        reader = WSIReader.open(mc.ROOT / row["image"], mpp=0.25)
        old = legacy_arrays(cohort, row["id"])
        new = [np.load(p, mmap_mode="r") for p in sorted((BASE / cohort / "probabilities").glob(f"{row['id']}.*.npy"))]
        if not old or not new:
            raise FileNotFoundError("Padding audit requires both inference caches")
        checks = {}
        for constant, reference in ((0, old), (255, new)):
            totals = [np.zeros_like(a) for a in reference]
            count = np.zeros((height, width, 1), dtype=np.float32)
            for start in range(0, len(bounds), batch_size):
                patches = [
                    model.preproc_func(
                        reader.read_bounds(
                            b,
                            resolution=0.25,
                            units="mpp",
                            coord_space="resolution",
                            pad_constant_values=constant,
                        )
                    )
                    for b in bounds[start : start + batch_size]
                ]
                predicted = model.infer_batch(model, torch.from_numpy(np.stack(patches)), device="cuda")
                if isinstance(predicted, np.ndarray):
                    predicted = (predicted,)
                for index, (x1, y1, x2, y2) in enumerate(outputs[start : start + batch_size]):
                    x2, y2 = min(x2, width), min(y2, height)
                    for total, batch in zip(totals, predicted, strict=True):
                        total[y1:y2, x1:x2] += batch[index, : y2 - y1, : x2 - x1]
                    count[y1:y2, x1:x2] += 1
            if not np.all(count > 0):
                raise ValueError("Patch replay did not cover the image")
            for arr in totals:
                arr /= count
            validate(cohort, totals, row["shape"])
            replay_pred = postprocess(cohort, model, totals)
            reference_pred = postprocess(cohort, model, reference)
            replay_raw = (
                totals[0].argmax(-1) if cohort == "bcss" or cohort.startswith("micronet") else totals[0][..., 0] >= 0.5
            )
            reference_raw = (
                reference[0].argmax(-1)
                if cohort == "bcss" or cohort.startswith("micronet")
                else reference[0][..., 0] >= 0.5
            )
            checks[str(constant)] = {
                "head_mean_absolute_errors": [
                    float(np.abs(a - b).mean()) for a, b in zip(totals, reference, strict=True)
                ],
                "head_max_absolute_errors": [
                    float(np.abs(a - b).max()) for a, b in zip(totals, reference, strict=True)
                ],
                "raw_disagreement_pixels": int((replay_raw != reference_raw).sum()),
                "replay_scores": scores(cohort, gt, replay_pred, valid, replay_raw),
                "reference_scores": scores(cohort, gt, reference_pred, valid, reference_raw),
            }
        results.append({"id": row["id"], "patches": len(bounds), "padding": checks})
        print(
            f"padding audit {cohort}/{row['id']}: " + str({k: v["replay_scores"]["dice"] for k, v in checks.items()}),
            flush=True,
        )
    shared.write_json(
        BASE / cohort / ("padding_stride256_audit.json" if legacy_bcss_stride else "padding_audit.json"),
        {
            "runtime": env,
            "checkpoint_sha256": checksum,
            "cases": results,
            "complete_cohort": len(results) == len(data),
            "selection": "first N cases in manifest order",
            "stride_shape": list(stride_shape),
            "legacy_bcss_stride": legacy_bcss_stride,
        },
    )


def report(cohort: str, *, evidence: bool = False) -> None:
    if evidence:
        with (Path(__file__).parent / "evidence" / f"{cohort}_v2.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        metrics = [k[len("argmax_") :] for k in rows[0] if k.startswith("argmax_")]
        cases = [
            {"id": r["id"], "scores": {m: {k: float(r[f"{m}_{k}"]) for k in metrics} for m in methods(cohort)}}
            for r in rows
        ]
        if len(cases) != COHORTS[cohort][4] or len({r["id"] for r in cases}) != len(cases):
            raise ValueError("Incomplete evidence")
        state = {"cases": cases, "summary": summarize(cases), "complete": True}
    else:
        state = json.loads((BASE / cohort / "results.json").read_text())
    print(f"{cohort}, n={len(state['cases'])}, complete={state['complete']}")
    print("| Metric | Method | Argmax | RankSEG | Change (pp) | 95% CI (pp) | Improved |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for metric, summary in state["summary"].items():
        for method, s in summary.items():
            if method == "argmax":
                continue
            lo, hi = np.array(s["paired_ci95"]) * 100
            print(
                f"| {metric} | {method} | {summary['argmax']['mean']:.6f} | {s['mean']:.6f} "
                f"| {s['gain'] * 100:+.3f} | {lo:+.3f} to {hi:+.3f} | {s['wins']}/{len(state['cases'])} |"
            )


def presence_audit() -> None:
    """Explain BCSS class scores on annotated-positive versus absent ROIs."""
    cohort = "bcss"
    dest = BASE / cohort
    manifest = json.loads((dest / "inference_manifest.json").read_text())
    state = json.loads((dest / "results.json").read_text())
    if not state["complete"]:
        raise ValueError("Presence audit requires all 45 results")
    records = {r["id"]: r for r in state["cases"]}
    rows = []
    for row in manifest["inputs"]:
        if mc.digest(mc.ROOT / row["label"]) != row["label_sha256"]:
            raise ValueError("Label changed")
        gt, valid = truth(cohort, row)
        rows.append(
            {
                "id": row["id"],
                "valid_pixels": int(valid.sum()),
                "class_pixels": {k: int((valid & (gt == index)).sum()) for index, k in enumerate(bcss.CLASSES)},
            }
        )
    summary = {}
    for k in bcss.CLASSES:
        summary[k] = {}
        for group in ("present", "absent"):
            selected = [r for r in rows if (r["class_pixels"][k] > 0) == (group == "present")]
            means = (
                {
                    m: float(np.mean([records[r["id"]]["scores"][m][f"dice_{k}"] for r in selected]))
                    for m in methods(cohort)
                }
                if selected
                else {}
            )
            switches = []
            if group == "absent":
                for r in selected:
                    values = {m: records[r["id"]]["scores"][m][f"dice_{k}"] for m in methods(cohort)}
                    if not all(v in (0, 1) for v in values.values()):
                        raise ValueError("Absent class Dice must be 0 or 1")
                    if len(set(values.values())) > 1:
                        switches.append({"id": r["id"], "scores": values})
            summary[k][group] = {"n": len(selected), "mean_dice": means, "discordant_absence_cases": switches}
    shared.write_json(
        dest / "presence_audit.json",
        {
            "cases": rows,
            "summary": summary,
            "note": "Diagnostic only; primary evaluation still averages every class over all 45 ROIs.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=COHORTS)
    parser.add_argument("step", choices=("infer", "evaluate", "report", "padding", "presence"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--from-evidence", action="store_true")
    parser.add_argument(
        "--legacy-bcss-stride", action="store_true", help="Diagnostic replay of the old BCSS stride 256"
    )
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= COHORTS[args.cohort][4]:
        parser.error("--limit exceeds cohort size")
    if args.from_evidence and args.step != "report":
        parser.error("--from-evidence requires report")
    if args.step == "presence" and (args.cohort != "bcss" or args.limit is not None):
        parser.error("presence requires the complete BCSS cohort")
    if args.legacy_bcss_stride and (args.step != "padding" or args.cohort != "bcss"):
        parser.error("--legacy-bcss-stride requires bcss padding")
    if args.step == "report":
        report(args.cohort, evidence=args.from_evidence)
    elif args.step == "presence":
        presence_audit()
    elif args.step == "padding":
        padding_audit(args.cohort, args.limit, legacy_bcss_stride=args.legacy_bcss_stride)
    else:
        {"infer": infer, "evaluate": evaluate}[args.step](args.cohort, args.limit)
