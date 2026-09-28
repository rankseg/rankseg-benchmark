"""Smoke tests using synthetic probs (no network, no HF downloads)."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import types
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image

from rankseg_benchmark.datasets import REGISTRY, DatasetSpec, _decode_label, _decode_probs, iter_batches
from rankseg_benchmark.demo import (
    DEMO_THEMES,
    DemoSelection,
    _base_image,
    _contour,
    _mask_image,
    _overlay,
    _validate_demo_manifest,
    render_figure,
)
from rankseg_benchmark.metrics import ConfusionAccumulator, FoldMeanAccumulator, MedicalCaseAccumulator
from rankseg_benchmark.monai_cache import (
    EXTERNAL_EXPOSURE_DECLARATION,
    _adapt_label,
    _adapt_probabilities,
    _atomic_torch_save,
    _software_info,
    _validate_external_test_provenance,
    load_monai_specs,
    resolve_cases,
    validate_probability_payload,
)
from rankseg_benchmark.runner import (
    _rankseg_output_mode,
    _rankseg_predict,
    configure_rankseg_path,
    format_report,
    run_benchmark,
)
from rankseg_benchmark.timing import Timer


def test_builtin_monai_specs_match_registered_volume_targets():
    specs = load_monai_specs()

    assert set(specs) == {
        "monai_btcv_swin_v058_msd_pancreas",
        "monai_btcv_swin_v058_msd_spleen",
    }
    for benchmark_id in specs:
        registry_spec = REGISTRY[benchmark_id]
        assert registry_spec.source_type == "local_artifacts"
        assert registry_spec.spatial_dims == 3
        assert registry_spec.eval_unit == "volume"
        assert tuple(specs[benchmark_id]["dataset"]["class_names"]) == registry_spec.class_names
        assert specs[benchmark_id]["dataset"]["evaluation_design"] == "external_test"
        assert specs[benchmark_id]["dataset"]["checkpoint_exposure"] == EXTERNAL_EXPOSURE_DECLARATION
        assert not any(specs[benchmark_id]["dataset"]["test_case_usage"].values())
        source_id = specs[benchmark_id]["dataset"]["source_dataset_id"]
        bundle = specs[benchmark_id]["bundle"]
        assert source_id != bundle["supervised_training_dataset_id"]
        assert source_id not in bundle["self_supervised_pretraining_dataset_ids"]
        assert specs[benchmark_id]["bundle"]["supervised_training_dataset"] == "Beyond the Cranial Vault (BTCV)"


def test_monai_external_test_provenance_rejects_checkpoint_source_overlap():
    spec = deepcopy(load_monai_specs()["monai_btcv_swin_v058_msd_pancreas"])
    spec["dataset"]["source_dataset_id"] = spec["bundle"]["supervised_training_dataset_id"]

    with pytest.raises(ValueError, match="separately"):
        _validate_external_test_provenance("overlap-fixture", spec)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_monai_demo_renderer_is_deterministic(tmp_path, theme):
    label = torch.zeros((64, 64), dtype=torch.bool)
    label[20:44, 18:48] = True
    baseline = label.clone()
    baseline[20:32, 18:25] = False
    rankseg = label.clone()
    rankseg[40:44, 44:48] = False
    selected = DemoSelection(
        case_id="fixture",
        case_path=tmp_path / "fixture.pt",
        slice_index=1,
        label=label,
        baseline=baseline,
        rankseg=rankseg,
        baseline_dice=0.9,
        rankseg_dice=0.98,
        corrected=84,
        introduced=16,
    )
    image_volume = torch.linspace(0, 1, 64 * 64 * 3).reshape(64, 64, 3)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    render_figure(selected, image_volume, first, theme=theme)
    render_figure(selected, image_volume, second, theme=theme)

    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.size == (1440, 624)
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0
        # With the padding used here, the entire fixture is the shared ROI.
        # Check the CT, orientation, and each overlay pixel-for-pixel, not just
        # the canvas dimensions. The two themes must not alter medical pixels.
        base = _base_image(np.rot90(image_volume[:, :, 1].numpy()), (440, 440))
        masks = [_mask_image(np.rot90(mask.numpy()), (440, 440)) for mask in (label, baseline, rankseg)]
        expected_panels = [
            _overlay(base, masks[0], (48, 211, 190), 145),
            _contour(_overlay(base, masks[1], (251, 146, 60), 125), masks[0], (242, 246, 255)),
            _contour(_overlay(base, masks[2], (45, 212, 145), 135), masks[0], (242, 246, 255)),
        ]
        for x, expected in zip((36, 500, 964), expected_panels, strict=True):
            # Exclude the rounded frame at the very edge.
            actual = image.crop((x + 4, 96, x + 436, 528))
            assert np.array_equal(np.asarray(actual), np.asarray(expected.convert("RGBA").crop((4, 4, 436, 436))))
        title_pixels = np.asarray(image.crop((0, 0, 1440, 80)))
        assert np.any(np.all(title_pixels == DEMO_THEMES[theme]["primary"], axis=-1))


def test_monai_demo_requires_complete_predeclared_cohort():
    spec = load_monai_specs()["monai_btcv_swin_v058_msd_pancreas"]
    manifest = {
        "complete": True,
        "cases": [{"case_id": case_id} for case_id in spec["dataset"]["split"]["case_ids"]],
        "bundle": {
            "name": spec["bundle"]["name"],
            "version": spec["bundle"]["version"],
            "checkpoint": spec["bundle"]["checkpoint"],
        },
    }

    _validate_demo_manifest(manifest, spec)
    manifest["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        _validate_demo_manifest(manifest, spec)


def test_monai_fixed_external_spleen_split(tmp_path):
    spec = load_monai_specs()["monai_btcv_swin_v058_msd_spleen"]
    expected_case_ids = spec["dataset"]["split"]["case_ids"]
    for case_id in expected_case_ids:
        for directory in ("imagesTr", "labelsTr"):
            path = tmp_path / directory / f"{case_id}.nii.gz"
            path.parent.mkdir(exist_ok=True)
            path.touch()

    cases = resolve_cases(spec, tmp_path)

    assert len(cases) == 9
    assert [case_id for case_id, _, _ in cases] == expected_case_ids


def test_monai_binary_output_and_label_adapters():
    spec = load_monai_specs()["monai_btcv_swin_v058_msd_pancreas"]
    probabilities = torch.softmax(torch.randn((14, 2, 3, 4)), dim=0)
    source_label = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 0]]])

    adapted_probabilities = _adapt_probabilities(probabilities, spec)
    adapted_label = _adapt_label(source_label, spec)

    assert adapted_probabilities.shape == (2, 2, 3, 4)
    assert torch.equal(adapted_probabilities[1], probabilities[11])
    assert torch.allclose(adapted_probabilities.sum(dim=0), torch.ones((2, 3, 4)))
    assert torch.equal(adapted_label, (source_label > 0).long())

    with pytest.raises(ValueError, match="outside"):
        _adapt_label(torch.tensor([[[3]]]), spec)


def test_monai_probability_payload_validation():
    label = torch.tensor(
        [
            [[0, 1], [1, 0]],
            [[0, 0], [1, 1]],
        ]
    )
    foreground = torch.where(label == 1, 0.9, 0.1).float()
    payload = {
        "probabilities": torch.stack((1 - foreground, foreground)),
        "label": label,
    }

    validate_probability_payload(payload, num_classes=2)


def test_monai_software_metadata_is_weights_only_safe(tmp_path):
    payload = {
        "schema_version": 1,
        "case_id": "safe-metadata",
        "probabilities": torch.ones((1, 1, 1, 1)),
        "label": torch.zeros((1, 1, 1), dtype=torch.uint8),
        "metadata": {"software": _software_info(types.SimpleNamespace(__version__="test"))},
    }
    path = tmp_path / "case.pt"

    _atomic_torch_save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    assert loaded["metadata"]["software"]["torch"] == str(torch.__version__)


@pytest.mark.parametrize("metric", ["dice", "iou"])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_local_monai_3d_artifact_runs_end_to_end(tmp_path, metric, device):
    spec = REGISTRY["monai_btcv_swin_v058_msd_spleen"]
    label = torch.tensor(
        [
            [[0, 1], [1, 0]],
            [[0, 0], [1, 1]],
        ]
    )
    foreground = torch.where(label == 1, 0.9, 0.1).float()
    case_path = tmp_path / "cases" / "spleen_fixture.pt"
    case_path.parent.mkdir()
    torch.save(
        {
            "schema_version": 1,
            "case_id": "spleen_fixture",
            "probabilities": torch.stack((1 - foreground, foreground)),
            "label": label,
            "metadata": {},
        },
        case_path,
    )
    case_sha256 = hashlib.sha256(case_path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": spec.name,
                "cases": [
                    {
                        "case_id": "spleen_fixture",
                        "file": "cases/spleen_fixture.pt",
                        "sha256": case_sha256,
                    }
                ],
            }
        )
    )

    baseline, rankseg = run_benchmark(
        spec,
        artifact_dir=tmp_path,
        device=device,
        metric=metric,
        warmup=1,
        batch_size=1,
        progress=False,
    )

    assert baseline.confusion.n_images == 1
    assert rankseg.confusion.n_images == 1
    assert baseline.summary()["mDice"] == 1.0
    assert rankseg.summary()["mDice"] == 1.0
    assert baseline.summary()["mIoU"] == 1.0
    assert rankseg.summary()["mIoU"] == 1.0
    assert baseline.summary()["unit"] == "volume"
    report = format_report(
        baseline,
        rankseg,
        class_names=list(spec.class_names),
        per_class=True,
    )
    assert "spleen" in report
    assert "background" not in report
    assert "active evaluation units" in report


def test_local_monai_artifact_loader_yields_3d_volume(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    cases_dir = artifact_dir / "cases"
    cases_dir.mkdir(parents=True)
    probabilities = torch.full((2, 2, 3, 4), 0.25)
    probabilities[0] = 0.75
    label = torch.zeros((2, 3, 4), dtype=torch.uint8)
    torch.save(
        {
            "schema_version": 1,
            "case_id": "case-001",
            "probabilities": probabilities,
            "label": label,
            "metadata": {"coordinate_space": "resampled_model_space"},
        },
        cases_dir / "case-001.pt",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": "local-volume",
                "cases": [{"case_id": "case-001", "file": "cases/case-001.pt"}],
            }
        )
    )
    spec = DatasetSpec(
        name="local-volume",
        hf_repo="",
        hf_split="validation",
        num_classes=2,
        output_mode="multiclass",
        spatial_dims=3,
        eval_unit="volume",
        source_type="local_artifacts",
    )

    batches = list(iter_batches(spec, artifact_dir=artifact_dir, batch_size=1))

    assert len(batches) == 1
    probs, labels = batches[0]
    assert probs.shape == (1, 2, 2, 3, 4)
    assert labels.shape == (1, 2, 3, 4)


def test_local_monai_artifact_loader_rejects_checksum_mismatch(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case_path = cases_dir / "case-001.pt"
    torch.save(
        {
            "schema_version": 1,
            "case_id": "case-001",
            "probabilities": torch.stack((torch.ones((1, 1, 1)), torch.zeros((1, 1, 1)))),
            "label": torch.zeros((1, 1, 1), dtype=torch.uint8),
        },
        case_path,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": "local-volume",
                "cases": [{"case_id": "case-001", "file": "cases/case-001.pt", "sha256": "0" * 64}],
            }
        )
    )
    spec = DatasetSpec(
        name="local-volume",
        hf_repo="",
        hf_split="validation",
        num_classes=2,
        output_mode="multiclass",
        spatial_dims=3,
        eval_unit="volume",
        source_type="local_artifacts",
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        list(iter_batches(spec, artifact_dir=tmp_path, batch_size=1))


