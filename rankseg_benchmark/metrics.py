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


@dataclass
class MedicalCaseAccumulator:
    """Case-level medical metric aligned with RankSEG-RMA ``test_medical``.

    The reference creates one AccuracyMetric per case, adds slice predictions to
    that case metric, calls ``value()`` per case, then averages case-level
    results. The main ``mDice``/``mIoU`` values here correspond to the
    reference's ``mDiceI``/``mIoUI``: each slice gets a foreground metric first,
    empty-foreground slices are treated as 1 for binary datasets, then slices
    are averaged within the case.
    """

    num_classes: int
    ignore_index: int | None = None
    binary: bool = True

    _cases: dict[object, "_MedicalCaseMetric"] = field(default_factory=dict, init=False, repr=False)

    @torch.no_grad()
    def update_slice(self, case_id: object, pred: torch.Tensor, label: torch.Tensor) -> None:
        metric = self._cases.setdefault(
            case_id,
            _MedicalCaseMetric(self.num_classes, self.ignore_index, self.binary),
        )
        metric.add(pred.detach().cpu(), label.detach().cpu())

    @property
    def n_cases(self) -> int:
        return len(self._cases)

    def summary(self) -> dict[str, float | torch.Tensor]:
        if not self._cases:
            return {
                "mDice": 0.0,
                "mIoU": 0.0,
                "per_class_dice": torch.zeros(self.num_classes, dtype=torch.float64),
                "per_class_iou": torch.zeros(self.num_classes, dtype=torch.float64),
                "active_per_class": torch.zeros(self.num_classes, dtype=torch.float64),
                "n_cases": 0,
            }

        case_summaries = [case.summary() for case in self._cases.values()]
        mDice = float(torch.tensor([s["mDice"] for s in case_summaries], dtype=torch.float64).mean().item())
        mIoU = float(torch.tensor([s["mIoU"] for s in case_summaries], dtype=torch.float64).mean().item())

        per_class_dice = torch.stack([s["per_class_dice"] for s in case_summaries]).mean(dim=0)
        per_class_iou = torch.stack([s["per_class_iou"] for s in case_summaries]).mean(dim=0)
        active_per_class = torch.stack([s["active_per_class"] for s in case_summaries]).sum(dim=0)

        return {
            "mDice": mDice,
            "mIoU": mIoU,
            "per_class_dice": per_class_dice,
            "per_class_iou": per_class_iou,
            "active_per_class": active_per_class,
            "n_cases": self.n_cases,
        }


@dataclass
class FoldMeanAccumulator:
    """Average metric summaries across folds."""

    num_classes: int

    _folds: dict[int, MedicalCaseAccumulator] = field(default_factory=dict, init=False, repr=False)

    def add_fold(self, fold: int, accumulator: MedicalCaseAccumulator) -> None:
        self._folds[fold] = accumulator

    @property
    def n_images(self) -> int:
        return int(sum(acc.n_cases for acc in self._folds.values()))

    def summary(self) -> dict[str, float | torch.Tensor | dict[int, dict[str, float | int]]]:
        if not self._folds:
            return {
                "mDice": 0.0,
                "mIoU": 0.0,
                "per_class_dice": torch.zeros(self.num_classes, dtype=torch.float64),
                "per_class_iou": torch.zeros(self.num_classes, dtype=torch.float64),
                "active_per_class": torch.zeros(self.num_classes, dtype=torch.float64),
                "folds": {},
            }

        fold_summaries = {fold: acc.summary() for fold, acc in sorted(self._folds.items())}
        mDice = float(torch.tensor([s["mDice"] for s in fold_summaries.values()], dtype=torch.float64).mean().item())
        mIoU = float(torch.tensor([s["mIoU"] for s in fold_summaries.values()], dtype=torch.float64).mean().item())
        per_class_dice = torch.stack([s["per_class_dice"] for s in fold_summaries.values()]).mean(dim=0)
        per_class_iou = torch.stack([s["per_class_iou"] for s in fold_summaries.values()]).mean(dim=0)
        active_per_class = torch.stack([s["active_per_class"] for s in fold_summaries.values()]).sum(dim=0)
        folds = {
            fold: {
                "mDice": float(summary["mDice"]),
                "mIoU": float(summary["mIoU"]),
                "n_cases": int(summary["n_cases"]),
            }
            for fold, summary in fold_summaries.items()
        }

        return {
            "mDice": mDice,
            "mIoU": mIoU,
            "per_class_dice": per_class_dice,
            "per_class_iou": per_class_iou,
            "active_per_class": active_per_class,
            "folds": folds,
        }


@dataclass
class _MedicalCaseMetric:
    num_classes: int
    ignore_index: int | None
    binary: bool

    tp: list[torch.Tensor] = field(default_factory=list)
    fp: list[torch.Tensor] = field(default_factory=list)
    fn: list[torch.Tensor] = field(default_factory=list)
    active_classes: list[torch.Tensor] = field(default_factory=list)

    def add(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        pred = pred.reshape(-1).long()
        label = label.reshape(-1).long()
        if self.ignore_index is not None:
            keep = label != self.ignore_index
            pred = pred[keep]
            label = label[keep]
        if label.numel() < 1:
            return

        label = torch.clamp(label, 0, self.num_classes - 1)
        pred_valid = (pred >= 0) & (pred < self.num_classes)
        pred_count = torch.bincount(pred[pred_valid], minlength=self.num_classes).double()
        tp = torch.bincount(label[pred_valid & (pred == label)], minlength=self.num_classes).double()
        label_count = torch.bincount(label, minlength=self.num_classes).double()
        fp = pred_count - tp
        fn = label_count - tp

        if self.binary:
            active = (pred_count + label_count) > 0
        else:
            active = label_count > 0

        self.tp.append(tp)
        self.fp.append(fp)
        self.fn.append(fn)
        self.active_classes.append(active.bool())

    def summary(self) -> dict[str, float | torch.Tensor]:
        if not self.tp:
            return {
                "mDice": 0.0,
                "mIoU": 0.0,
                "per_class_dice": torch.zeros(self.num_classes, dtype=torch.float64),
                "per_class_iou": torch.zeros(self.num_classes, dtype=torch.float64),
                "active_per_class": torch.zeros(self.num_classes, dtype=torch.float64),
            }

        tp = torch.stack(self.tp)
        fp = torch.stack(self.fp)
        fn = torch.stack(self.fn)
        active = torch.stack(self.active_classes)

        iou_by_slice = tp / (tp + fp + fn)
        dice_by_slice = 2 * tp / (2 * tp + fp + fn)
        iou_by_slice = torch.nan_to_num(iou_by_slice, nan=0.0)
        dice_by_slice = torch.nan_to_num(dice_by_slice, nan=0.0)
        iou_by_slice[~active] = 0
        dice_by_slice[~active] = 0

        if self.binary:
            active_sum = active.sum(dim=1)
            image_iou = iou_by_slice[:, 1].clone()
            image_dice = dice_by_slice[:, 1].clone()
            image_iou[active_sum < 2] = 1
            image_dice[active_sum < 2] = 1
            per_class_iou = torch.zeros(self.num_classes, dtype=torch.float64)
            per_class_dice = torch.zeros(self.num_classes, dtype=torch.float64)
            per_class_iou[1] = image_iou.mean()
            per_class_dice[1] = image_dice.mean()
        else:
            active_sum = active.sum(dim=1).clamp(min=1)
            image_iou = iou_by_slice.sum(dim=1) / active_sum
            image_dice = dice_by_slice.sum(dim=1) / active_sum
            per_class_iou = _reduce_class(iou_by_slice, active)
            per_class_dice = _reduce_class(dice_by_slice, active)

        return {
            "mDice": float(image_dice.mean().item()),
            "mIoU": float(image_iou.mean().item()),
            "per_class_dice": per_class_dice,
            "per_class_iou": per_class_iou,
            "active_per_class": active.sum(dim=0).double(),
        }


def _reduce_class(value_matrix: torch.Tensor, active_classes: torch.Tensor) -> torch.Tensor:
    value_matrix = value_matrix.clone()
    value_matrix[~active_classes] = 1e6
    num_images, num_classes = value_matrix.shape
    active_sum = active_classes.sum(dim=0).double().clamp(min=1)
    indices = torch.arange(num_images).view(-1, 1).expand_as(value_matrix)
    mask = indices < active_sum.view(1, -1)
    return (mask * torch.sort(value_matrix, dim=0)[0]).sum(dim=0) / active_sum
