"""Explain Other-class multilabel errors on the cached BCSS probabilities."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from .bcss_multilabel import source_meta
from .bcss import BASE, GT_TO_CLASS


def main() -> None:
    import torch
    from rankseg.functional import rankseg

    assert torch.cuda.is_available()
    recorded = json.loads((BASE / "multilabel_results.json").read_text())
    assert recorded["summary"]["rankseg_source"] == source_meta()
    edges = np.linspace(0, 1, 11)
    names = ("tumor", "stroma", "inflammatory", "necrosis", "other")
    totals = {key: np.zeros(5, dtype=np.int64) for key in ("ground_truth_by_class", "selected_other_by_gt")}
    prob_mass = np.zeros(5, dtype=np.float64)
    hist = {
        key: np.zeros(10, dtype=np.int64)
        for key in ("selected_other_true_positive", "selected_other_false_positive", "missed_true_other")
    }
    gt_raw_other = np.zeros(22, dtype=np.int64)
    cases = []
    for i, case in enumerate(recorded["cases"], 1):
        uid = case["id"]
        probs = np.load(BASE / "probabilities" / f"{uid}.npy", mmap_mode="r")
        with Image.open(BASE / "data" / f"{uid}.mask.png") as image:
            raw = np.asarray(image.resize((probs.shape[1], probs.shape[0]), Image.Resampling.NEAREST))
        valid = (raw != 0) & (raw != 7)
        gt = GT_TO_CLASS[raw]
        prob_gpu = torch.from_numpy(np.asarray(probs).copy()).permute(2, 0, 1).unsqueeze(0).to("cuda")
        selected = (
            rankseg(prob_gpu, metric="dice", solver="RMA", output_mode="multilabel", safe_screening=True)[0, 4]
            .cpu()
            .numpy()
        )
        del prob_gpu
        case_selected_by_gt = np.zeros(5, dtype=np.int64)
        for start in range(0, gt.shape[0], 128):
            stop = min(start + 128, gt.shape[0])
            v = valid[start:stop]
            g = gt[start:stop]
            s = selected[start:stop]
            p = np.asarray(probs[start:stop, :, 4])
            true_other = v & (g == 4)
            selected_valid = v & s
            selected_tp = selected_valid & (g == 4)
            selected_fp = selected_valid & (g != 4)
            missed_other = true_other & ~s
            totals["ground_truth_by_class"] += np.bincount(g[v], minlength=5)
            by_gt = np.bincount(g[selected_valid], minlength=5)
            case_selected_by_gt += by_gt
            totals["selected_other_by_gt"] += by_gt
            gt_raw_other += np.bincount(raw[start:stop][true_other], minlength=22)
            for klass in range(5):
                prob_mass[klass] += p[v & (g == klass)].sum(dtype=np.float64)
            hist["selected_other_true_positive"] += np.histogram(p[selected_tp], bins=edges)[0]
            hist["selected_other_false_positive"] += np.histogram(p[selected_fp], bins=edges)[0]
            hist["missed_true_other"] += np.histogram(p[missed_other], bins=edges)[0]
        assert (
            sum(case_selected_by_gt) == case["multilabel_pixels"][4]
            or abs(sum(case_selected_by_gt) - case["multilabel_pixels"][4]) < 5000
        ), uid
        cases.append({"id": uid, "selected_other_by_gt": case_selected_by_gt.tolist()})
        print(f"[{i}/45] {uid} selected Other={sum(case_selected_by_gt):,}", flush=True)
        del selected, probs
        torch.cuda.empty_cache()

    result = {
        "classes": names,
        "rankseg_source": source_meta(),
        "probability_bin_edges": edges.tolist(),
        "ground_truth_by_class": totals["ground_truth_by_class"].tolist(),
        "selected_other_by_gt": totals["selected_other_by_gt"].tolist(),
        "probability_mass_other_by_gt": prob_mass.tolist(),
        "histograms": {key: value.tolist() for key, value in hist.items()},
        "raw_other_gt_codes": gt_raw_other.tolist(),
        "cases": cases,
    }
    (BASE / "other_multilabel_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
