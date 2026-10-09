"""FLARE22 external-checkpoint protocol tests; no dataset or network required."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from rankseg_benchmark.monai.cache import _atomic_torch_save, _hash_file
from rankseg_benchmark.monai.flare22 import (
    CASE_LIST_SHA256,
    DATASET_ID,
    EXPECTED_CASES,
    ORGANS,
    PRIOR_MSD_OVERLAP,
    SOURCE_LABELS,
    evaluate,
    native_organ_volumes,
    resolve_cases,
    target_mapping,
)


def test_flare22_locked_case_list_and_label_mapping(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for index in range(1, 51):
        (images / f"FLARE22_Tr_{index:04d}_0000.nii.gz").touch()
        (labels / f"FLARE22_Tr_{index:04d}.nii.gz").touch()
    cases = resolve_cases(tmp_path)
    assert len(cases) == EXPECTED_CASES
    assert hashlib.sha256("\n".join(case_id for case_id, _, _ in cases).encode()).hexdigest() == CASE_LIST_SHA256
    assert SOURCE_LABELS == {"left_adrenal": 8, "right_adrenal": 7, "gallbladder": 9,
                             "pancreas": 4, "spleen": 3, "liver": 1}
    assert {row["name"]: row["model_channel"] for row in target_mapping()} == {
        "left_adrenal": 13, "right_adrenal": 12, "gallbladder": 4,
        "pancreas": 11, "spleen": 1, "liver": 6,
    }
    assert len(PRIOR_MSD_OVERLAP) == 4
    (labels / "FLARE22_Tr_0050.nii.gz").unlink()
    with pytest.raises(ValueError, match="image/label mismatch"):
        resolve_cases(tmp_path)


def test_flare22_native_volumes_use_official_label_ids(tmp_path):
    nib = pytest.importorskip("nibabel")
    data = np.zeros((2, 3, 4), dtype=np.uint8)
    data[0, 0, :2] = 8
    data[1, 1, 0] = 7
    data[1, 1, 1] = 9
    image = nib.Nifti1Image(data, np.diag([1.5, 2.0, 3.0, 1.0]))
    image.header.set_xyzt_units("mm")
    path = tmp_path / "label.nii.gz"
    nib.save(image, path)
    volumes = native_organ_volumes(path)
    assert volumes["left_adrenal"] == {"voxels": 2, "volume_mm3": pytest.approx(18.0)}
    assert volumes["right_adrenal"]["voxels"] == 1
    assert volumes["gallbladder"]["volume_mm3"] == pytest.approx(9.0)


def test_flare22_primary_excludes_only_predeclared_image_overlaps(tmp_path):
    label = torch.zeros((2, 2, 2), dtype=torch.uint8)
    probs = torch.full((len(ORGANS), 2, 2, 2), 0.01)
    for index, organ in enumerate(ORGANS):
        location = (index // 4, (index // 2) % 2, index % 2)
        label[location] = SOURCE_LABELS[organ.name]
        probs[(index, *location)] = 0.9
    case_id = "FLARE22_Tr_0001"
    payload = {"schema_version": 1, "case_id": case_id, "probabilities": probs,
               "label": label, "metadata": {"fingerprint": "fixture",
               "native_organ_volumes": {organ.name: {"voxels": 1, "volume_mm3": 1.0} for organ in ORGANS}}}
    path = tmp_path / "cases" / f"{case_id}.pt"
    _atomic_torch_save(payload, path)
    manifest = {"study": DATASET_ID, "expected_cases": EXPECTED_CASES,
                "case_list_sha256": CASE_LIST_SHA256, "complete": False,
                "targets": target_mapping(), "prior_msd_overlap": PRIOR_MSD_OVERLAP,
                "generation_fingerprint": "fixture",
                "cases": [{"case_id": case_id, "file": f"cases/{case_id}.pt",
                           "sha256": _hash_file(path), "prior_msd_overlap": False}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Partial"):
        evaluate(tmp_path, torch.device("cpu"), warmup=1, repeats=1)
    report = evaluate(tmp_path, torch.device("cpu"), warmup=1, repeats=1, allow_partial=True)
    assert report["cases"] == 1 and report["primary_cases"] == 1
    assert len(report["case_organ_results"]) == 6
    assert all(report["primary_summary"][organ.name]["argmax_dice"] == 1 for organ in ORGANS)
    assert report["primary_shared_positive"]["cases"] == 1
    manifest["cases"][0]["prior_msd_overlap"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap mismatch"):
        evaluate(tmp_path, torch.device("cpu"), warmup=1, repeats=1, allow_partial=True)
