"""Case-isolation and scoring tests for matched official MONAI Bundles."""

import csv
import hashlib
import zipfile

import pytest
import torch

from rankseg_benchmark.monai.matched_official import (
    _summary,
    audit_prostate_archives,
    audit_prostate_extraction,
    audit_prostate_split_lists,
    resolve_prostate_cases,
    resolve_spleen_cases,
    score_foreground,
)


def test_prostate_uses_only_nineteen_distinct_test_cases(tmp_path):
    rows = []
    for index in range(19):
        case_id = f"case{index:02d}"
        case_dir = tmp_path / "test" / case_id
        case_dir.mkdir(parents=True)
        for name in ("t2.nii.gz", "reader1.nii.gz", "reader2.nii.gz"):
            (case_dir / name).touch()
        rows.append(
            {
                "t2": f"test/{case_id}/t2.nii.gz",
                "t2_anatomy_reader1": f"test/{case_id}/reader1.nii.gz",
                "t2_anatomy_reader2": f"test/{case_id}/reader2.nii.gz",
            }
        )
    with (tmp_path / "test.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cases = resolve_prostate_cases(tmp_path)
    assert len(cases) == 19
    assert len({case.case_id for case in cases}) == 19
    assert set(cases[0].labels) == {"reader1", "reader2"}

    rows[1] = dict(rows[0])
    with (tmp_path / "test.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="not unique"):
        resolve_prostate_cases(tmp_path)


def test_prostate_rejects_training_or_validation_case_overlap(tmp_path):
    from rankseg_benchmark.monai.matched_official import Case

    cases = [
        Case(f"{index:03d}", tmp_path / f"{index:03d}.nii.gz", {"reader1": tmp_path / "label.nii.gz"})
        for index in range(140, 159)
    ]
    for split, ids in (("train", range(1, 120)), ("valid", range(120, 140))):
        with (tmp_path / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("ID", "t2"))
            writer.writeheader()
            writer.writerows({"ID": str(case_id), "t2": f"train/{case_id:03d}/t2.nii.gz"} for case_id in ids)
    result = audit_prostate_split_lists(cases, tmp_path)
    assert (result["training_cases"], result["validation_cases"], result["test_cases"]) == (119, 20, 19)
    with (tmp_path / "valid.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ID", "t2"))
        writer.writeheader()
        writer.writerows(
            {"ID": str(case_id), "t2": f"train/{case_id:03d}/t2.nii.gz"}
            for case_id in (*range(120, 139), 1)
        )
    with pytest.raises(ValueError, match="overlap"):
        audit_prostate_split_lists(cases, tmp_path)


def test_archive_directory_audit_rejects_byte_duplicate_images(tmp_path):
    list_root = tmp_path / "lists"
    list_root.mkdir()
    training_archive = tmp_path / "train.zip"
    test_archive = tmp_path / "test.zip"
    with zipfile.ZipFile(training_archive, "w") as archive:
        for split in ("train", "valid"):
            data = f"ID,t2\n{split},train/{split}/t2.nii.gz\n".encode()
            (list_root / f"{split}.csv").write_bytes(data)
            archive.writestr(f"prostate158_train/{split}.csv", data)
        for index in range(139):
            for modality in ("t2", "adc", "dwi"):
                archive.writestr(
                    f"prostate158_train/train/{index:03d}/{modality}.nii.gz",
                    f"train-{index}-{modality}".encode(),
                )
    with zipfile.ZipFile(test_archive, "w") as archive:
        for index in range(19):
            for modality in ("t2", "adc", "dwi"):
                archive.writestr(
                    f"prostate158_test/test/{index:03d}/{modality}.nii.gz",
                    f"test-{index}-{modality}".encode(),
                )
    expected_md5 = hashlib.md5(test_archive.read_bytes()).hexdigest()
    result = audit_prostate_archives(list_root, test_archive, str(training_archive), expected_md5)
    assert result["modality_files"]["t2.nii.gz"]["byte_identical_candidates"] == 0

    with zipfile.ZipFile(test_archive, "a") as archive:
        archive.writestr("prostate158_test/test/extra/t2.nii.gz", b"train-0-t2")
    with pytest.raises(ValueError, match="byte duplicates"):
        audit_prostate_archives(list_root, test_archive, str(training_archive), expected_md5)


def test_extracted_prostate_files_must_match_archive(tmp_path):
    from rankseg_benchmark.monai.matched_official import Case

    case_dir = tmp_path / "test" / "001"
    case_dir.mkdir(parents=True)
    paths = [case_dir / name for name in ("t2.nii.gz", "reader1.nii.gz", "reader2.nii.gz")]
    archive_path = tmp_path / "test.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in paths:
            path.write_bytes(path.name.encode())
            archive.write(path, f"prostate158_test/test/001/{path.name}")
    case = Case("001", paths[0].resolve(), {"reader1": paths[1].resolve(), "reader2": paths[2].resolve()})
    assert audit_prostate_extraction([case], tmp_path, archive_path) == 3
    paths[1].write_bytes(b"different")
    with pytest.raises(ValueError, match="differs from archive"):
        audit_prostate_extraction([case], tmp_path, archive_path)


def test_spleen_is_exact_sorted_last_n_and_disjoint(tmp_path):
    images = tmp_path / "imagesTr"
    labels = tmp_path / "labelsTr"
    images.mkdir()
    labels.mkdir()
    for index in range(41):
        name = f"spleen_{index:02d}.nii.gz"
        (images / name).touch()
        (labels / name).touch()
    cases, train_ids = resolve_spleen_cases(tmp_path)
    assert len(train_ids) == 32
    assert [case.case_id for case in cases] == [f"spleen_{index:02d}" for index in range(32, 41)]
    assert set(train_ids).isdisjoint(case.case_id for case in cases)


def test_foreground_macro_scores_and_absent_class():
    prediction = torch.tensor([0, 1, 1, 2, 2, 0])
    truth = torch.tensor([0, 1, 2, 2, 0, 0])
    result = score_foreground(prediction, truth, (1, 2))
    assert result["classes"]["1"]["dice"] == pytest.approx(2 / 3)
    assert result["classes"]["2"]["dice"] == pytest.approx(1 / 2)
    assert result["dice"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert result["iou"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    single = score_foreground(prediction, torch.tensor([0, 1, 0, 0, 0, 0]), (1, 2))
    assert set(single["classes"]) == {"1"}


def test_foreground_score_matches_existing_benchmark_metric():
    from rankseg_benchmark.quick.metrics import ConfusionAccumulator

    prediction = torch.tensor([0, 1, 1, 2, 2, 0]).view(1, 2, 3)
    truth = torch.tensor([0, 1, 2, 2, 0, 0]).view(1, 2, 3)
    independent = ConfusionAccumulator(3, evaluation_class_ids=(1, 2))
    independent.update(prediction, truth)
    observed = score_foreground(prediction[0], truth[0], (1, 2))
    assert observed["dice"] == pytest.approx(independent.summary()["mDice"])
    assert observed["iou"] == pytest.approx(independent.summary()["mIoU"])


def test_case_macro_summary_is_not_voxel_weighted():
    records = [
        {"scores": {"label": {"argmax": {"dice": 0.2, "iou": 0.1, "classes": {"1": {"dice": 0.2}}},
                              "rma": {"dice": 0.4, "iou": 0.25, "classes": {"1": {"dice": 0.4}}}}},
         "timing": {"argmax": {"median_ms": 1.0}, "rma": {"median_ms": 5.0}}},
        {"scores": {"label": {"argmax": {"dice": 0.8, "iou": 0.7, "classes": {"1": {"dice": 0.8}}},
                              "rma": {"dice": 0.6, "iou": 0.5, "classes": {"1": {"dice": 0.6}}}}},
         "timing": {"argmax": {"median_ms": 3.0}, "rma": {"median_ms": 9.0}}},
    ]
    result = _summary(records, ("label",), (1,))
    assert result["readers"]["label"]["argmax"]["mean_case_dice"] == pytest.approx(0.5)
    assert result["readers"]["label"]["rma"]["mean_case_dice"] == pytest.approx(0.5)
    assert result["mean_of_case_median_ms"] == {"argmax": 2.0, "rma": 7.0}
