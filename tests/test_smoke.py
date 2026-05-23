"""Smoke tests using synthetic probs (no network, no HF downloads)."""

from __future__ import annotations

import io
import sys

import numpy as np
import torch

from rankseg_benchmark.datasets import _decode_label, _decode_probs
from rankseg_benchmark.metrics import ConfusionAccumulator, FoldMeanAccumulator, MedicalCaseAccumulator
from rankseg_benchmark.runner import _rankseg_predict, configure_rankseg_path
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
    pred = _rankseg_predict(
        FakeRankSEG(),
        probs,
        dataset_output_mode="multiclass",
        rankseg_output_mode="multilabel",
    )

    assert pred.shape == (1, 2, 2)
    assert pred.dtype == torch.int64
    assert torch.equal(pred, torch.tensor([[[0, 1], [1, 1]]]))


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
