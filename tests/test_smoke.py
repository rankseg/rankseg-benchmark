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


def test_multiclass_perfect_prediction_scores_one():
    acc = ConfusionAccumulator(num_classes=3, output_mode="multiclass")
    label = torch.tensor([[[0, 1, 2], [2, 1, 0]]])
    pred = label.clone()
    acc.update(pred, label)
    s = acc.summary()
    for key in ("mDice", "mIoU"):
        assert abs(s[key] - 1.0) < 1e-6, f"{key} should be 1.0, got {s[key]}"


def test_multiclass_respects_ignore_index():
    acc = ConfusionAccumulator(num_classes=2, output_mode="multiclass", ignore_index=255)
    label = torch.tensor([[[0, 1, 255]]])
    pred = torch.tensor([[[0, 1, 0]]])  # the 255 pixel would otherwise be FP for class 0
    acc.update(pred, label)
    assert abs(acc.summary()["mIoU"] - 1.0) < 1e-6


def test_mDice_averages_per_image_then_across_images():
    """mDice: per-image average over active classes, then average over images.

    Image 0: 4 pixels, 2 classes active (0 and 1), prediction perfect -> Dice = (1 + 1) / 2 = 1.0
    Image 1: 4 pixels, 1 class active (class 0), prediction half wrong:
        label = [0, 0, 0, 0], pred = [0, 0, 1, 1] -> for class 0: TP=2, FP=0, FN=2 -> Dice=2*2/(2*2+0+2)=0.6667
        only class 0 is active for image 1, so per-image dice = 0.6667
    mDice = (1.0 + 0.6667) / 2 = 0.8333
    """
    acc = ConfusionAccumulator(num_classes=2, output_mode="multiclass")
    label = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]])
    pred = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    acc.update(pred, label)
    s = acc.summary()
    assert abs(s["mDice"] - (1.0 + 2 / 3) / 2) < 1e-4


def test_multilabel_perfect_prediction_scores_one():
    acc = ConfusionAccumulator(num_classes=2, output_mode="multilabel")
    label = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]], dtype=torch.bool)  # (1, 2, 2, 2)
    pred = label.clone()
    acc.update(pred, label)
    s = acc.summary()
    assert abs(s["mDice"] - 1.0) < 1e-6
    assert abs(s["mIoU"] - 1.0) < 1e-6


def test_per_class_breakdown_shape():
    acc = ConfusionAccumulator(num_classes=3, output_mode="multiclass")
    label = torch.tensor([[0, 1, 2], [0, 1, 2]])
    pred = label.clone()
    acc.update(pred, label)
    s = acc.summary()
    assert s["per_class_dice"].shape == (3,)
    assert s["per_class_iou"].shape == (3,)
    assert s["active_per_class"].shape == (3,)
    assert torch.allclose(s["active_per_class"], torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))


def test_medical_case_binary_matches_reference_empty_foreground_rule():
    acc = MedicalCaseAccumulator(num_classes=2, binary=True)
    acc.update_slice("case-a", torch.tensor([[0, 0], [0, 0]]), torch.tensor([[0, 0], [0, 0]]))
    acc.update_slice("case-a", torch.tensor([[1, 0], [0, 0]]), torch.tensor([[1, 1], [0, 0]]))

    s = acc.summary()
    # Slice 0 has no foreground in pred or label, so RankSEG-RMA reduceI treats it as 1.
    # Slice 1 foreground IoU is 1 / (1 + 0 + 1) = 0.5. Case mIoUI = (1 + 0.5) / 2.
    assert abs(s["mIoU"] - 0.75) < 1e-6
    assert abs(s["per_class_iou"][1].item() - 0.75) < 1e-6


def test_fold_mean_accumulator_averages_fold_summaries():
    fold0 = MedicalCaseAccumulator(num_classes=2, binary=True)
    fold0.update_slice("case-a", torch.tensor([[1, 0]]), torch.tensor([[1, 0]]))

    fold1 = MedicalCaseAccumulator(num_classes=2, binary=True)
    fold1.update_slice("case-b", torch.tensor([[1, 0]]), torch.tensor([[1, 1]]))

    folds = FoldMeanAccumulator(num_classes=2)
    folds.add_fold(0, fold0)
    folds.add_fold(1, fold1)

    s = folds.summary()
    assert abs(s["mIoU"] - 0.75) < 1e-6
    assert s["folds"][0]["n_cases"] == 1
    assert s["folds"][1]["n_cases"] == 1


def test_timer_records_calls():
    t = Timer(use_cuda=False)
    with t.measure():
        sum(range(1000))
    with t.measure():
        sum(range(1000))
    assert t.n_calls == 2
    assert t.mean_ms >= 0.0


def test_binary_multiclass_rankseg_adapter_uses_foreground_channel():
    class FakeRankSEG:
        def predict(self, probs):
            assert probs.shape == (1, 1, 2, 2)
            return probs > 0.5

    probs = torch.tensor([[[[0.9, 0.2], [0.1, 0.4]], [[0.1, 0.8], [0.9, 0.6]]]])
    spec = DatasetSpec(
        name="binary-foreground",
        hf_repo="example/repo",
        hf_split="test",
        num_classes=2,
        output_mode="multiclass",
        rankseg_channels=(1,),
        evaluation_class_ids=(1,),
    )
    pred = _rankseg_predict(
        FakeRankSEG(),
        probs,
        spec=spec,
        rankseg_output_mode="multilabel",
    )

    assert pred.shape == (1, 2, 2)
    assert pred.dtype == torch.int64
    assert torch.equal(pred, torch.tensor([[[0, 1], [1, 1]]]))


def test_kits_uses_foreground_channel_rankseg_for_every_solver():
    spec = REGISTRY["kits"]

    assert spec.rankseg_channels == (1,)
    assert spec.evaluation_class_ids == (1,)
    assert spec.display_mode == "foreground"
    for solver in ("RMA", "rma", "BA", "TRNA", "BA+TRNA"):
        assert _rankseg_output_mode(spec, solver) == "multilabel"


def test_regular_binary_multiclass_rma_keeps_multiclass_semantics():
    spec = DatasetSpec(
        name="binary-macro",
        hf_repo="example/repo",
        hf_split="test",
        num_classes=2,
        output_mode="multiclass",
    )

    assert spec.rankseg_channels is None
    assert _rankseg_output_mode(spec, "RMA") == "multiclass"
    assert _rankseg_output_mode(spec, "ba") == "multilabel"


@pytest.mark.parametrize("rankseg_channels", [(0,), (1, 0)])
def test_multiclass_rankseg_channel_selection_rejects_ambiguous_class_mapping(rankseg_channels):
    with pytest.raises(ValueError, match="rankseg_channels|non-background"):
        DatasetSpec(
            name="invalid-routing",
            hf_repo="example/repo",
            hf_split="test",
            num_classes=2,
            output_mode="multiclass",
            rankseg_channels=rankseg_channels,
        )


def test_evaluation_class_ids_exclude_background_from_macro_metrics():
    acc = ConfusionAccumulator(
        num_classes=2,
        output_mode="multiclass",
        evaluation_class_ids=(1,),
    )
    label = torch.tensor([[[0, 0, 1, 1]]])
    pred = torch.tensor([[[0, 0, 0, 1]]])

    acc.update(pred, label)
    summary = acc.summary()

    assert abs(summary["mDice"] - 2 / 3) < 1e-6
    assert abs(summary["mIoU"] - 1 / 2) < 1e-6
    assert summary["active_per_class"].tolist() == [0.0, 1.0]


def test_decodes_npy_bytes_from_hf_parquet_rows():
    probs_arr = np.ones((2, 3, 4), dtype=np.float16)
    label_arr = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]], dtype=np.int64)

    probs_buf = io.BytesIO()
    label_buf = io.BytesIO()
    np.save(probs_buf, probs_arr)
    np.save(label_buf, label_arr)

    probs = _decode_probs(probs_buf.getvalue(), num_classes=2)
    label = _decode_label(label_buf.getvalue(), output_mode="multiclass")

    assert probs.shape == (2, 3, 4)
    assert probs.dtype == torch.float32
    assert label.shape == (3, 4)
    assert label.dtype == torch.int64


def test_decodes_3d_volume_from_npy_bytes():
    probs_arr = np.full((3, 2, 3, 4), 1 / 3, dtype=np.float32)
    label_arr = np.zeros((2, 3, 4), dtype=np.uint8)

    probs_buf = io.BytesIO()
    label_buf = io.BytesIO()
    np.save(probs_buf, probs_arr)
    np.save(label_buf, label_arr)

    probs = _decode_probs(probs_buf.getvalue(), num_classes=3, spatial_dims=3)
    label = _decode_label(
        label_buf.getvalue(),
        output_mode="multiclass",
        num_classes=3,
        spatial_dims=3,
    )

    assert probs.shape == (3, 2, 3, 4)
    assert label.shape == (2, 3, 4)


def test_tensor_decode_avoids_copy_when_dtype_and_layout_already_match():
    raw_probs = torch.rand((2, 2, 3, 4), dtype=torch.float32)
    raw_label = torch.zeros((2, 3, 4), dtype=torch.int64)

    probs = _decode_probs(raw_probs, num_classes=2, spatial_dims=3)
    label = _decode_label(raw_label, output_mode="multiclass", num_classes=2, spatial_dims=3)

    assert probs.data_ptr() == raw_probs.data_ptr()
    assert label.data_ptr() == raw_label.data_ptr()


def test_warmup_does_not_drop_samples_from_metrics_or_timing(monkeypatch):
    spec = DatasetSpec(
        name="warmup-fixture",
        hf_repo="example/repo",
        hf_split="test",
        num_classes=2,
        output_mode="multiclass",
    )
    label = torch.tensor([[[0, 1]], [[1, 0]]])
    probs = torch.nn.functional.one_hot(label, num_classes=2).movedim(-1, 1).float()

    class FakeRankSEG:
        def __init__(self, **_kwargs):
            pass

        def predict(self, values):
            return values.argmax(dim=1)

    def fake_iter_batches(*_args, **_kwargs):
        yield probs, label

    monkeypatch.setitem(sys.modules, "rankseg", types.SimpleNamespace(RankSEG=FakeRankSEG))
    monkeypatch.setattr("rankseg_benchmark.runner.iter_batches", fake_iter_batches)

    baseline, rankseg = run_benchmark(spec, warmup=2, batch_size=2, progress=False)

    assert baseline.confusion.n_images == 2
    assert rankseg.confusion.n_images == 2
    assert baseline.timer.n_calls == 2
    assert rankseg.timer.n_calls == 2
    assert baseline.summary()["mDice"] == 1.0
    assert rankseg.summary()["mDice"] == 1.0


def test_case_level_warmup_does_not_drop_slices(monkeypatch):
    spec = DatasetSpec(
        name="case-warmup-fixture",
        hf_repo="example/repo",
        hf_split="test",
        hf_data_dir="fixture",
        num_classes=2,
        output_mode="multiclass",
        eval_unit="case",
        num_folds=1,
        case_id_key="case_id",
        rankseg_channels=(1,),
        evaluation_class_ids=(1,),
    )
    label = torch.tensor([[[0, 1], [1, 0]]])
    probs = torch.nn.functional.one_hot(label, num_classes=2).movedim(-1, 1).float()

    class FakeRankSEG:
        def __init__(self, **_kwargs):
            pass

        def predict(self, values):
            return values > 0.5

    def fake_iter_batches_with_metadata(*_args, **_kwargs):
        yield probs, label, [{"case_id": "case-a"}]

    monkeypatch.setitem(sys.modules, "rankseg", types.SimpleNamespace(RankSEG=FakeRankSEG))
    monkeypatch.setattr(
        "rankseg_benchmark.runner.iter_batches_with_metadata",
        fake_iter_batches_with_metadata,
    )

    baseline, rankseg = run_benchmark(spec, warmup=1, batch_size=1, progress=False)

    assert baseline.confusion.n_images == 1
    assert rankseg.confusion.n_images == 1
    assert baseline.timer.n_calls == 1
    assert rankseg.timer.n_calls == 1
    assert baseline.summary()["mDice"] == 1.0
    assert rankseg.summary()["mDice"] == 1.0


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


def test_configure_rankseg_path_prepends_local_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "rankseg"
    checkout.mkdir()
    monkeypatch.delenv("RANKSEG_PATH", raising=False)
    monkeypatch.setattr(sys, "path", ["existing"])

    resolved = configure_rankseg_path(checkout)

    assert resolved == checkout.resolve()
    assert sys.path[0] == str(checkout.resolve())
    assert sys.path[1:] == ["existing"]


def test_configure_rankseg_path_uses_env_when_argument_missing(tmp_path, monkeypatch):
    checkout = tmp_path / "rankseg"
    checkout.mkdir()
    monkeypatch.setenv("RANKSEG_PATH", str(checkout))
    monkeypatch.setattr(sys, "path", [])

    configure_rankseg_path()

    assert sys.path[0] == str(checkout.resolve())
