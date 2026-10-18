"""Evaluate local RankSEG multilabel masks on the cached BCSS test probabilities."""

from __future__ import annotations

import csv
import json
import time

import numpy as np
from PIL import Image

from .bcss import BASE, CLASSES, GT_TO_CLASS, cohort, score_case
from .provenance import decoder_source


def score_binary(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> tuple[list[float], list[int], list[int]]:
    scores, sizes, intersections = [], [], []
    for klass in range(len(CLASSES)):
        truth = valid & (gt == klass)
        estimate = valid & pred[klass]
        n_true = int(np.count_nonzero(truth))
        n_pred = int(np.count_nonzero(estimate))
        intersection = int(np.count_nonzero(truth & estimate))
        scores.append(1.0 if n_true + n_pred == 0 else 2 * intersection / (n_true + n_pred))
        sizes.append(n_pred)
        intersections.append(intersection)
    return scores, sizes, intersections


def source_meta() -> dict:
    return decoder_source()


def main() -> None:
    import torch
    from rankseg.functional import rankseg

    assert torch.cuda.is_available(), "CUDA is required for this cached full-resolution run"
    previous = json.loads((BASE / "results.json").read_text())
    meta = source_meta()
    assert previous["summary"]["rankseg_source"] == meta, "Multiclass reference uses a different local source"
    reference = {case["id"]: case for case in previous["cases"]}
    assert len(reference) == 45

    partial = BASE / "multilabel.partial.json"
    output = BASE / "multilabel_results.json"
    if partial.exists():
        state = json.loads(partial.read_text())
        assert state["source"] == meta, "Partial run uses a different RankSEG source"
        results = state["cases"]
    else:
        results = []
    completed = {case["id"] for case in results}
    assert len(completed) == len(results), "Duplicate case in partial run"

    rows = cohort()
    for index, row in enumerate(rows, 1):
        uid = row[""]
        if uid in completed:
            print(f"[{index}/45] cached {uid}", flush=True)
            continue
        probs = np.load(BASE / "probabilities" / f"{uid}.npy", mmap_mode="r")
        assert probs.ndim == 3 and probs.shape[-1] == len(CLASSES)
        with Image.open(BASE / "data" / f"{uid}.mask.png") as image:
            raw = np.asarray(image.resize((probs.shape[1], probs.shape[0]), resample=Image.Resampling.NEAREST))
        assert raw.min() >= 0 and raw.max() <= 21
        valid = (raw != 0) & (raw != 7)
        gt = GT_TO_CLASS[raw]
        argmax = np.argmax(probs, axis=-1).astype(np.uint8)
        a = score_case(gt, argmax, valid)
        assert np.allclose(a, reference[uid]["argmax"], rtol=0, atol=1e-12), uid
        assert int(np.count_nonzero(valid)) == reference[uid]["valid_pixels"], uid

        torch_probs = torch.from_numpy(np.asarray(probs).copy()).permute(2, 0, 1).unsqueeze(0).to("cuda")
        started = time.time()
        pred = (
            rankseg(
                torch_probs,
                metric="dice",
                solver="RMA",
                output_mode="multilabel",
                safe_screening=True,
            )[0]
            .cpu()
            .numpy()
        )
        seconds = time.time() - started
        assert pred.dtype == np.bool_ and pred.shape == (len(CLASSES), *gt.shape), uid
        scores, pred_pixels, intersections = score_binary(gt, pred, valid)
        selected_count = pred.sum(axis=0)
        present = [k for k in range(len(CLASSES)) if reference[uid]["gt_pixels"][k] > 0]
        case = {
            "id": uid,
            "argmax": a,
            "multilabel": scores,
            "argmax_macro": float(np.mean(a)),
            "multilabel_macro": float(np.mean(scores)),
            "argmax_present_macro": float(np.mean([a[k] for k in present])),
            "multilabel_present_macro": float(np.mean([scores[k] for k in present])),
            "gt_pixels": reference[uid]["gt_pixels"],
            "argmax_pixels": reference[uid]["argmax_pixels"],
            "multilabel_pixels": pred_pixels,
            "multilabel_intersections": intersections,
            "valid_pixels": reference[uid]["valid_pixels"],
            "valid_overlap_pixels": int(np.count_nonzero(valid & (selected_count > 1))),
            "valid_unassigned_pixels": int(np.count_nonzero(valid & (selected_count == 0))),
            "decode_seconds": seconds,
        }
        results.append(case)
        partial.write_text(json.dumps({"source": meta, "cases": results}, indent=2) + "\n")
        print(
            f"[{index}/45] {uid} macro {case['argmax_macro']:.4f} -> {case['multilabel_macro']:.4f} ({seconds:.1f}s)",
            flush=True,
        )
        del torch_probs, pred, selected_count, probs
        torch.cuda.empty_cache()

    assert {case["id"] for case in results} == set(reference), "Incomplete run"
    results.sort(key=lambda case: case["id"])
    deltas = np.array([case["multilabel_macro"] - case["argmax_macro"] for case in results])
    rng = np.random.default_rng(20261017)
    boot = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
    summary = {
        "n": len(results),
        "classes": CLASSES,
        "argmax_macro": float(np.mean([case["argmax_macro"] for case in results])),
        "multilabel_macro": float(np.mean([case["multilabel_macro"] for case in results])),
        "multiclass_macro_from_reference": previous["summary"]["macro_rankseg"],
        "mean_delta_vs_argmax": float(deltas.mean()),
        "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "wins": int(np.count_nonzero(deltas > 0)),
        "ties": int(np.count_nonzero(deltas == 0)),
        "losses": int(np.count_nonzero(deltas < 0)),
        "per_class_argmax": np.mean([case["argmax"] for case in results], axis=0).tolist(),
        "per_class_multilabel": np.mean([case["multilabel"] for case in results], axis=0).tolist(),
        "present_class_macro_argmax": float(np.mean([case["argmax_present_macro"] for case in results])),
        "present_class_macro_multilabel": float(np.mean([case["multilabel_present_macro"] for case in results])),
        "valid_pixels": sum(case["valid_pixels"] for case in results),
        "valid_overlap_pixels": sum(case["valid_overlap_pixels"] for case in results),
        "valid_unassigned_pixels": sum(case["valid_unassigned_pixels"] for case in results),
        "total_gt_pixels": np.sum([case["gt_pixels"] for case in results], axis=0).tolist(),
        "total_argmax_pixels": np.sum([case["argmax_pixels"] for case in results], axis=0).tolist(),
        "total_multilabel_pixels": np.sum([case["multilabel_pixels"] for case in results], axis=0).tolist(),
        "total_multilabel_intersections": np.sum(
            [case["multilabel_intersections"] for case in results], axis=0
        ).tolist(),
        "score_rule": "Per-ROI binary Dice for all five classes against one-hot GT; both empty=1, GT empty/pred nonempty=0; ignore raw labels 0 and 7; average classes then ROIs. Overlap is allowed and each class is scored independently.",
        "decoder": "RankSEG 0.0.8; Dice/RMA/multilabel/safe_screening=True/pruning_prob=0.5",
        "rankseg_source": meta,
        "torch_version": torch.__version__,
    }
    output.write_text(json.dumps({"summary": summary, "cases": results}, indent=2) + "\n")
    with (BASE / "multilabel_per_case.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "argmax_macro",
                "multilabel_macro",
                "delta",
                "overlap_pixels",
                "unassigned_pixels",
                *[f"argmax_{name}" for name in CLASSES],
                *[f"multilabel_{name}" for name in CLASSES],
            ]
        )
        for case in results:
            writer.writerow(
                [
                    case["id"],
                    case["argmax_macro"],
                    case["multilabel_macro"],
                    case["multilabel_macro"] - case["argmax_macro"],
                    case["valid_overlap_pixels"],
                    case["valid_unassigned_pixels"],
                    *case["argmax"],
                    *case["multilabel"],
                ]
            )
    partial.unlink()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
