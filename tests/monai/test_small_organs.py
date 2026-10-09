"""AMOS22 small-organ protocol tests; no network or checkpoint download."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from rankseg_benchmark.monai.cache import _atomic_torch_save, _hash_file
from rankseg_benchmark.monai.small_organs import (
    CASE_LIST_SHA256,
    EXPECTED_CASES,
    ORGANS,
    _scores,
    evaluate,
    native_organ_volumes,
    resolve_cases,
    shared_positive_summary,
    size_baseline_strata,
    summarize,
    validate_native_pair,
    validate_payload,
)


def test_amos_study_has_locked_cohort_and_unique_class_mapping():
    assert EXPECTED_CASES == 100
    assert len(CASE_LIST_SHA256) == 64
    assert len(ORGANS) == 6
    assert len({organ.name for organ in ORGANS}) == 6
    assert len({organ.amos_label for organ in ORGANS}) == 6
    assert len({organ.model_channel for organ in ORGANS}) == 6
    assert {organ.name for organ in ORGANS if organ.size_group == "small"} == {
        "left_adrenal", "right_adrenal", "gallbladder"
    }


def test_amos_resolver_rejects_missing_labels_or_changed_case_list(tmp_path):
    images = tmp_path / "valid/imagesVa"
    labels = tmp_path / "valid/labelsVa"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name in ("amos_0008", "amos_0013"):
        (images / f"{name}.nii.gz").touch()
        (labels / f"{name}.nii.gz").touch()
    digest = hashlib.sha256("amos_0008\namos_0013".encode()).hexdigest()
    assert [case[0] for case in resolve_cases(tmp_path, expected_count=2, expected_digest=digest)] == [
        "amos_0008", "amos_0013"
    ]
    with pytest.raises(ValueError, match="exactly 100"):
        resolve_cases(tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        resolve_cases(tmp_path, expected_count=2, expected_digest="0" * 64)
    (labels / "amos_0013.nii.gz").unlink()
    with pytest.raises(ValueError, match="case mismatch"):
        resolve_cases(tmp_path, expected_count=2, expected_digest=digest)


def test_amos_native_volume_uses_affine_determinant(tmp_path):
    nib = pytest.importorskip("nibabel")
    data = np.zeros((2, 3, 4), dtype=np.uint8)
    data[0, 0, :2] = 12
    data[1, 1, 0] = 6
    affine = np.diag([1.5, 2.0, 3.0, 1.0])
    path = tmp_path / "label.nii.gz"
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    volumes = native_organ_volumes(path)
    assert volumes["left_adrenal"]["voxels"] == 2
    assert volumes["left_adrenal"]["volume_mm3"] == pytest.approx(18.0)
    assert volumes["liver"]["volume_mm3"] == pytest.approx(9.0)


def test_amos_native_pair_requires_the_same_physical_grid(tmp_path):
    nib = pytest.importorskip("nibabel")
    data = np.zeros((2, 3, 4), dtype=np.uint8)
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_xyzt_units("mm")
    image_path = tmp_path / "image.nii.gz"
    nib.save(image, image_path)
    label = nib.Nifti1Image(data, np.eye(4))
    label.header.set_xyzt_units("mm")
    label_path = tmp_path / "label.nii.gz"
    nib.save(label, label_path)
    validate_native_pair(image_path, label_path)
    shifted = nib.Nifti1Image(data, np.diag([1.1, 1.0, 1.0, 1.0]))
    shifted.header.set_xyzt_units("mm")
    nib.save(shifted, label_path)
    with pytest.raises(ValueError, match="grids differ"):
        validate_native_pair(image_path, label_path)


def test_amos_payload_accepts_softmax_subset_but_rejects_invalid_data():
    payload = {"probabilities": torch.full((6, 2, 2, 2), 0.02),
               "label": torch.zeros((2, 2, 2), dtype=torch.uint8)}
    validate_payload(payload)  # six channels are not expected to sum to one
    bad = dict(payload, label=torch.full((2, 2, 2), 16, dtype=torch.uint8))
    with pytest.raises(ValueError, match="invalid label"):
        validate_payload(bad)
    bad = dict(payload, probabilities=torch.full((6, 2, 2, 2), float("nan")))
    with pytest.raises(ValueError, match="finite"):
        validate_payload(bad)
    bad = dict(payload, label=payload["label"].float())
    with pytest.raises(TypeError, match="uint8"):
        validate_payload(bad)


def test_amos_scores_keep_absent_ground_truth_separate():
    pred = torch.tensor([True, False, True])
    truth = torch.tensor([True, True, False])
    assert _scores(pred, truth) == (0.5, 1 / 3, 1, 2, 2)
    assert _scores(pred, torch.zeros_like(truth)) == (None, None, 0, 2, 0)


def test_amos_summary_is_per_case_and_reports_absent_classes():
    rows = []
    for organ in ORGANS:
        rows.append({"organ": organ.name, "gt_model_voxels": 1,
                     "gt_native_voxels": 1, "gt_native_mm3": 9.0,
                     "argmax_voxels": 1, "rankseg_voxels": 1,
                     "argmax_dice": 0.5, "rankseg_dice": 0.75,
                     "argmax_iou": 1 / 3, "rankseg_iou": 0.6,
                     "argmax_ms": 0.1, "rankseg_ms": 0.2})
        rows.append({"organ": organ.name, "gt_model_voxels": 0,
                     "gt_native_voxels": 0, "gt_native_mm3": 0.0,
                     "argmax_voxels": 2, "rankseg_voxels": 3,
                     "argmax_dice": None, "rankseg_dice": None,
                     "argmax_iou": None, "rankseg_iou": None,
                     "argmax_ms": 0.1, "rankseg_ms": 0.2})
    result = summarize(rows)
    for organ in ORGANS:
        entry = result[organ.name]
        assert entry["cases"] == 2 and entry["positive_cases"] == 1 and entry["absent_cases"] == 1
        assert entry["delta_dice_pp"] == 25.0
        assert entry["wins"] == 1 and entry["losses"] == 0
        assert entry["absent_argmax_predicted_voxels"] == 2
        assert entry["absent_rankseg_predicted_voxels"] == 3
        assert entry["absent_argmax_false_positive_cases"] == 1
        assert entry["absent_rankseg_false_positive_cases"] == 1


def test_amos_strata_cross_size_with_baseline_inside_each_organ():
    rows = []
    for organ in ORGANS:
        for volume, baseline, delta in ((1.0, 0.2, 0.1), (2.0, 0.8, -0.1),
                                        (3.0, 0.3, 0.2), (4.0, 0.9, -0.2)):
            rows.append({"organ": organ.name, "gt_model_voxels": 1,
                         "gt_native_mm3": volume, "argmax_dice": baseline,
                         "rankseg_dice": baseline + delta})
    result = size_baseline_strata(rows)
    for organ in ORGANS:
        entry = result[organ.name]
        assert entry["native_volume_median_mm3"] == 2.5
        assert entry["argmax_dice_median"] == 0.55
        assert entry["cells"]["smaller__lower_baseline"]["cases"] == 1
        assert entry["cells"]["smaller__lower_baseline"]["mean_delta_dice_pp"] == pytest.approx(10.0)
        assert entry["cells"]["larger__higher_baseline"]["cases"] == 1
        assert entry["cells"]["larger__higher_baseline"]["mean_delta_dice_pp"] == pytest.approx(-20.0)


def test_amos_evaluate_rejects_partial_cache_then_runs_cpu_fixture(tmp_path):
    label = torch.zeros((2, 2, 2), dtype=torch.uint8)
    probs = torch.full((len(ORGANS), 2, 2, 2), 0.01)
    for index, organ in enumerate(ORGANS):
        location = (index // 4, (index // 2) % 2, index % 2)
        label[location] = organ.amos_label
        probs[(index, *location)] = 0.9
    payload = {"schema_version": 1, "case_id": "amos_0008", "probabilities": probs,
               "label": label, "metadata": {"fingerprint": "fixture",
               "native_organ_volumes": {organ.name: {"voxels": 1, "volume_mm3": 1.0} for organ in ORGANS}}}
    path = tmp_path / "cases/amos_0008.pt"
    _atomic_torch_save(payload, path)
    manifest = {"study": "amos22_ct_official_validation", "expected_cases": EXPECTED_CASES,
                "case_list_sha256": CASE_LIST_SHA256, "complete": False,
                "targets": [vars(organ) for organ in ORGANS], "generation_fingerprint": "fixture",
                "cases": [{"case_id": "amos_0008", "file": "cases/amos_0008.pt", "sha256": _hash_file(path)}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Partial"):
        evaluate(tmp_path, torch.device("cpu"), warmup=1, repeats=1)
    report = evaluate(tmp_path, torch.device("cpu"), warmup=1, repeats=1, allow_partial=True)
    assert not report["complete_cohort"] and len(report["case_organ_results"]) == 6
    assert all(report["summary"][organ.name]["argmax_dice"] == 1.0 for organ in ORGANS)
    assert all(report["size_baseline_strata"][organ.name]["cells"]["smaller__lower_baseline"]["cases"] == 1
               for organ in ORGANS)
    assert report["shared_positive"]["cases"] == 1
    manifest["complete"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared complete"):
        evaluate(tmp_path, torch.device("cpu"), warmup=1, repeats=1)
    manifest["cases"] *= EXPECTED_CASES
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="case list"):
        evaluate(tmp_path, torch.device("cpu"), warmup=1, repeats=1)


def test_amos_shared_positive_selection_uses_ground_truth_only():
    rows = []
    for case_id in ("one", "two"):
        for organ in ORGANS:
            rows.append({"case_id": case_id, "organ": organ.name,
                         "gt_model_voxels": 0 if case_id == "two" and organ.name == "gallbladder" else 1,
                         "argmax_dice": 0.2 if case_id == "one" else 0.8,
                         "rankseg_dice": 0.3 if case_id == "one" else 0.9})
    result = shared_positive_summary(rows)
    assert result["cases"] == 1
    assert all(result["organs"][organ.name]["argmax_dice"] == 0.2 for organ in ORGANS)
