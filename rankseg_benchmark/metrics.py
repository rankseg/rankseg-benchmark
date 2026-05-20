"""Per-image (instance-level) Dice / IoU, following the RankSEG-RMA paper.

Reference: https://github.com/ZixunWang/RankSEG-RMA/blob/master/exp/metrics/accuracy_metric.py

For every image we compute per-class TP/FP/FN and a boolean ``active`` vector
identifying which classes "exist" in that image. The metrics are:

- ``mDice`` / ``mIoU`` (per-image, as reported in the paper):
    1. For each (image, class) compute Dice/IoU = 2TP / (2TP+FP+FN).
    2. For each image, average the score over its *active* classes only.
    3. Average those per-image numbers across the dataset.

- ``per_class_dice`` / ``per_class_iou`` (class-level, ``C`` style):
    For each class, average its Dice/IoU across the images where it is active.
    Used by the ``--per-class`` breakdown.

Active-class rule (matches the reference implementation):
- ``multiclass``: a class is active for an image iff it appears in that image's
  ground-truth label.
- ``multilabel``: a class is active for an image iff it has any positive pixel
  in pred OR label (mirrors the reference's ``binary=True`` branch).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ConfusionAccumulator:
    """Streaming accumulator for per-image Dice/IoU without materializing one-hot masks."""

    num_classes: int
    output_mode: str = "multiclass"  # "multiclass" or "multilabel"
    ignore_index: int | None = None

    # Accumulated on CPU as float64 to keep long runs numerically stable.
    _n_images: int = 0
    _sum_image_dice: float = 0.0
    _sum_image_iou: float = 0.0
    _sum_class_dice: torch.Tensor = field(init=False, repr=False)
    _sum_class_iou: torch.Tensor = field(init=False, repr=False)
    _active_per_class: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sum_class_dice = torch.zeros(self.num_classes, dtype=torch.float64)
        self._sum_class_iou = torch.zeros(self.num_classes, dtype=torch.float64)
        self._active_per_class = torch.zeros(self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Append per-image stats for every image in the batch.

        Multiclass:
            preds  : (B, *spatial) int
            labels : (B, *spatial) int
        Multilabel:
            preds  : (B, C, *spatial) bool/int
            labels : (B, C, *spatial) bool/int
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        if self.output_mode == "multiclass":
            self._update_multiclass(preds, labels)
        elif self.output_mode == "multilabel":
            self._update_multilabel(preds, labels)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode!r}")

    def _update_multiclass(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        B = preds.size(0)
        preds = preds.view(B, -1).long()
        labels = labels.view(B, -1).long()

        for i in range(B):
            pred_i = preds[i]
            label_i = labels[i]
            if self.ignore_index is not None:
                keep = label_i != self.ignore_index
                if not keep.any():
                    continue
                pred_i = pred_i[keep]
                label_i = label_i[keep]
            else:
                if label_i.numel() < 1:
                    continue

            label_valid = (label_i >= 0) & (label_i < self.num_classes)
            if not label_valid.any():
                continue
            pred_i = pred_i[label_valid]
            label_i = label_i[label_valid]
            pred_valid = (pred_i >= 0) & (pred_i < self.num_classes)

            tp = torch.bincount(label_i[pred_valid & (pred_i == label_i)], minlength=self.num_classes).double()
            pred_count = torch.bincount(pred_i[pred_valid], minlength=self.num_classes).double()
            label_count = torch.bincount(label_i, minlength=self.num_classes).double()
            fp = pred_count - tp
            fn = label_count - tp
            active = label_count > 0
            self._append_stats(tp, fp, fn, active)

    def _update_multilabel(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        # preds/labels: (B, C, *spatial); flatten spatial
        B, C = preds.shape[0], preds.shape[1]
        p = preds.reshape(B, C, -1).bool()
        l = labels.reshape(B, C, -1).bool()
        for i in range(B):
            tp = (p[i] & l[i]).sum(dim=1).double()
            pred_count = p[i].sum(dim=1).double()
            label_count = l[i].sum(dim=1).double()
            fp = pred_count - tp
            fn = label_count - tp
            active = (pred_count > 0) | (label_count > 0)
            self._append_stats(tp, fp, fn, active)

    def _append_stats(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, active: torch.Tensor) -> None:
        active = active.bool()
        active_count = int(active.sum().item())
        if active_count < 1:
            return

        smooth = 1e-7
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        active_f = active.double()

        self._sum_image_dice += float((dice * active_f).sum().item() / active_count)
        self._sum_image_iou += float((iou * active_f).sum().item() / active_count)
        self._sum_class_dice += dice * active_f
        self._sum_class_iou += iou * active_f
        self._active_per_class += active_f
        self._n_images += 1

    # ------------------------------------------------------------------ stats

    @property
    def n_images(self) -> int:
        return self._n_images

    def summary(self) -> dict[str, float | torch.Tensor]:
        if self._n_images == 0:
            per_class_dice = torch.zeros(self.num_classes, dtype=torch.float64)
            per_class_iou = torch.zeros(self.num_classes, dtype=torch.float64)
            mDice = 0.0
            mIoU = 0.0
        else:
            active_count = self._active_per_class.clamp(min=1.0)
            per_class_dice = self._sum_class_dice / active_count
            per_class_iou = self._sum_class_iou / active_count
            mDice = self._sum_image_dice / self._n_images
            mIoU = self._sum_image_iou / self._n_images

        return {
            "mDice": mDice,
            "mIoU": mIoU,
            "per_class_dice": per_class_dice,
            "per_class_iou": per_class_iou,
            "active_per_class": self._active_per_class.clone(),
        }
