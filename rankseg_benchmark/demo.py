"""Render a reproducible README figure from real MONAI external-test artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from rankseg_benchmark.datasets import get_spec
from rankseg_benchmark.monai_cache import _adapt_label, _build_preprocessing, _require_monai, load_monai_specs
from rankseg_benchmark.runner import configure_rankseg_path

DEMO_THEMES = {
    "light": {
        "primary": (17, 24, 39, 255),
        "muted": (51, 65, 85, 255),
        "border": (100, 116, 139, 220),
    },
    "dark": {
        "primary": (244, 248, 255, 255),
        "muted": (183, 202, 224, 255),
        "border": (69, 91, 119, 220),
    },
}


@dataclass
class DemoSelection:
    case_id: str
    case_path: Path
    slice_index: int
    label: torch.Tensor
    baseline: torch.Tensor
    rankseg: torch.Tensor
    baseline_dice: float
    rankseg_dice: float
    corrected: int
    introduced: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dice(prediction: torch.Tensor, label: torch.Tensor) -> float:
    denominator = int(prediction.sum()) + int(label.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int((prediction & label).sum()) / denominator


def _iou(prediction: torch.Tensor, label: torch.Tensor) -> float:
    union = int((prediction | label).sum())
    if union == 0:
        return 1.0
    return int((prediction & label).sum()) / union


def _validate_demo_manifest(manifest: dict, monai_spec: dict) -> None:
    if manifest.get("complete") is not True:
        raise ValueError("README visualization requires a complete external-test artifact manifest")
    records = manifest.get("cases")
    if not isinstance(records, list):
        raise ValueError("Artifact manifest cases must be a list")
    actual_case_ids = [record.get("case_id") for record in records]
    expected_case_ids = monai_spec["dataset"]["split"]["case_ids"]
    if actual_case_ids != expected_case_ids:
        raise ValueError("README visualization requires the exact predeclared external-test case list")

    actual_bundle = manifest.get("bundle", {})
    expected_bundle = monai_spec["bundle"]
    pinned_fields = ("name", "version", "checkpoint")
    if any(actual_bundle.get(field) != expected_bundle[field] for field in pinned_fields):
        raise ValueError("Artifact manifest does not match the pinned MONAI Bundle checkpoint")


def select_demo(
    artifact_dir: Path,
    *,
    device: torch.device,
    min_foreground: int,
) -> tuple[DemoSelection, dict[str, float], str]:
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    benchmark_id = manifest["benchmark_id"]
    spec = get_spec(benchmark_id)
    monai_spec = load_monai_specs()[benchmark_id]
    if spec.class_names != ("background", "pancreas"):
        raise ValueError("The README renderer currently requires the binary external Pancreas benchmark")
    _validate_demo_manifest(manifest, monai_spec)

    from rankseg import RankSEG

    decoder = RankSEG(metric="dice", solver="RMA", output_mode="multilabel")
    candidates: list[tuple[tuple[int, float, int, str], DemoSelection]] = []
    volume_metrics: list[tuple[float, float, float, float]] = []
    for record in manifest["cases"]:
        case_path = (artifact_dir / record["file"]).resolve()
        if _sha256(case_path) != record["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {case_path}")
        payload = torch.load(case_path, map_location="cpu", weights_only=True)
        probabilities = payload["probabilities"].to(device)
        label = payload["label"].bool()
        baseline = (probabilities[1] > 0.5).cpu()
        rankseg = decoder.predict(probabilities[1:2].unsqueeze(0))[0, 0].cpu()
        volume_metrics.append(
            (
                _dice(baseline, label),
                _dice(rankseg, label),
                _iou(baseline, label),
                _iou(rankseg, label),
            )
        )

        for slice_index in range(label.shape[2]):
            label_slice = label[:, :, slice_index]
            if int(label_slice.sum()) < min_foreground:
                continue
            baseline_slice = baseline[:, :, slice_index]
            rankseg_slice = rankseg[:, :, slice_index]
            baseline_dice = _dice(baseline_slice, label_slice)
            rankseg_dice = _dice(rankseg_slice, label_slice)
            corrected = int(((baseline_slice != label_slice) & (rankseg_slice == label_slice)).sum())
            introduced = int(((baseline_slice == label_slice) & (rankseg_slice != label_slice)).sum())
            selection = DemoSelection(
                case_id=payload["case_id"],
                case_path=case_path,
                slice_index=slice_index,
                label=label_slice,
                baseline=baseline_slice,
                rankseg=rankseg_slice,
                baseline_dice=baseline_dice,
                rankseg_dice=rankseg_dice,
                corrected=corrected,
                introduced=introduced,
            )
            # Predeclared selection rule: maximize net corrected pixels among
            # axial slices with enough foreground; deterministic tie-breakers.
            key = (corrected - introduced, rankseg_dice - baseline_dice, int(label_slice.sum()), payload["case_id"])
            candidates.append((key, selection))
        del probabilities

    if not candidates:
        raise ValueError(f"No slice has at least {min_foreground} foreground pixels")
    selected = max(candidates, key=lambda item: item[0])[1]
    values = torch.tensor(volume_metrics, dtype=torch.float64).mean(dim=0)
    summary = {
        "baseline_dice": float(values[0]),
        "rankseg_dice": float(values[1]),
        "baseline_iou": float(values[2]),
        "rankseg_iou": float(values[3]),
        "num_volumes": len(volume_metrics),
    }
    return selected, summary, benchmark_id


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:  # pragma: no cover - depends on fonts installed by the OS
        return ImageFont.load_default()


def _draw_theme_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    draw.text(position, text, font=font, fill=fill)


def _mask_image(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)


def _base_image(image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    display = np.clip((image - 0.06) / 0.78, 0.0, 1.0)
    grayscale = Image.fromarray(np.round(display * 255).astype(np.uint8)).resize(size, Image.Resampling.BICUBIC)
    return grayscale.convert("RGB")


def _overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: int) -> Image.Image:
    layer = Image.new("RGBA", image.size, (*color, 0))
    layer.putalpha(mask.point(lambda value: alpha if value else 0))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def _contour(image: Image.Image, mask: Image.Image, color: tuple[int, int, int], width: int = 3) -> Image.Image:
    expanded = mask.filter(ImageFilter.MaxFilter(width * 2 + 1))
    contracted = mask.filter(ImageFilter.MinFilter(width * 2 + 1))
    outline = np.asarray(expanded, dtype=np.int16) - np.asarray(contracted, dtype=np.int16)
    return _overlay(image, Image.fromarray(np.clip(outline, 0, 255).astype(np.uint8)), color, 255)


def _square_crop_bounds(mask: np.ndarray, shape: tuple[int, int], padding: int = 38) -> tuple[slice, slice]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return slice(0, shape[0]), slice(0, shape[1])
    lower = coordinates.min(axis=0) - padding
    upper = coordinates.max(axis=0) + padding + 1
    center = (lower + upper) / 2
    side = int(max(upper - lower))
    side = min(side, min(shape))
    start = np.floor(center - side / 2).astype(int)
    start = np.maximum(start, 0)
    start = np.minimum(start, np.asarray(shape) - side)
    return slice(int(start[0]), int(start[0] + side)), slice(int(start[1]), int(start[1] + side))


def render_figure(
    selected: DemoSelection,
    image_volume: torch.Tensor,
    output_path: Path,
    *,
    theme: str = "light",
) -> None:
    """Render three large comparison panels; cohort results belong in the caption."""
    colors = DEMO_THEMES[theme]
    image_slice = image_volume[:, :, selected.slice_index].numpy()
    label = selected.label.numpy()
    baseline = selected.baseline.numpy()
    rankseg = selected.rankseg.numpy()
    crop = _square_crop_bounds(label | baseline | rankseg, label.shape)

    # Radiological-style display orientation, applied identically to every panel.
    image_slice = np.rot90(image_slice[crop])
    label = np.rot90(label[crop])
    baseline = np.rot90(baseline[crop])
    rankseg = np.rot90(rankseg[crop])

    panel_size = (440, 440)
    gt_mask = _mask_image(label, panel_size)
    baseline_mask = _mask_image(baseline, panel_size)
    rankseg_mask = _mask_image(rankseg, panel_size)
    base = _base_image(image_slice, panel_size)

    ground_truth = _overlay(base, gt_mask, (48, 211, 190), 145)
    argmax = _contour(_overlay(base, baseline_mask, (251, 146, 60), 125), gt_mask, (242, 246, 255))
    optimized = _contour(_overlay(base, rankseg_mask, (45, 212, 145), 135), gt_mask, (242, 246, 255))
    canvas = Image.new("RGBA", (1440, 624), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    panels = [ground_truth, argmax, optimized]
    titles = ["Ground truth", "Argmax", "RankSEG"]
    subtitles = [
        "Pancreas + tumor",
        f"Dice {selected.baseline_dice * 100:.1f}",
        f"Dice {selected.rankseg_dice * 100:.1f}",
    ]
    panel_width = panel_size[0]
    gap = 24
    first_x = 36
    panel_y = 92
    for index, (panel, title, subtitle) in enumerate(zip(panels, titles, subtitles, strict=True)):
        x = first_x + index * (panel_width + gap)
        canvas.paste(panel.convert("RGBA"), (x, panel_y))
        draw.rounded_rectangle(
            (x - 2, panel_y - 2, x + panel_width + 2, panel_y + panel_width + 2),
            radius=5,
            outline=colors["border"],
            width=2,
        )
        title_box = draw.textbbox((0, 0), title, font=_font(40, bold=True))
        title_width = title_box[2] - title_box[0]
        _draw_theme_text(
            draw,
            (x + (panel_width - title_width) / 2, 22),
            title,
            font=_font(40, bold=True),
            fill=colors["primary"],
        )
        subtitle_box = draw.textbbox((0, 0), subtitle, font=_font(34))
        subtitle_width = subtitle_box[2] - subtitle_box[0]
        _draw_theme_text(
            draw,
            (x + (panel_width - subtitle_width) / 2, 554),
            subtitle,
            font=_font(34),
            fill=colors["muted"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rankseg-path", type=Path)
    parser.add_argument("--min-foreground", type=int, default=400)
    parser.add_argument("--theme", choices=sorted(DEMO_THEMES), default="light")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_foreground < 1:
        raise ValueError("--min-foreground must be >= 1")
    configure_rankseg_path(args.rankseg_path)
    device = torch.device(args.device)
    selected, summary, benchmark_id = select_demo(
        args.artifact_dir.expanduser().resolve(),
        device=device,
        min_foreground=args.min_foreground,
    )
    monai_spec = load_monai_specs()[benchmark_id]
    image_path = args.dataset_root.expanduser().resolve() / "imagesTr" / f"{selected.case_id}.nii.gz"
    label_path = args.dataset_root.expanduser().resolve() / "labelsTr" / f"{selected.case_id}.nii.gz"
    preprocessing = _build_preprocessing(monai_spec, _require_monai())
    transformed = preprocessing({"image": str(image_path), "label": str(label_path)})
    image_volume = transformed["image"].as_tensor().squeeze(0).float().cpu()
    transformed_label = _adapt_label(transformed["label"].as_tensor().squeeze(0), monai_spec).cpu().bool()
    payload = torch.load(selected.case_path, map_location="cpu", weights_only=True)
    if not torch.equal(transformed_label, payload["label"].bool()):
        raise ValueError("Reprocessed source label does not exactly match the artifact label")
    render_figure(
        selected,
        image_volume,
        args.output.expanduser().resolve(),
        theme=args.theme,
    )
    print(
        f"Wrote {args.output} from {selected.case_id} slice {selected.slice_index}; "
        f"slice Dice {selected.baseline_dice:.6f} -> {selected.rankseg_dice:.6f}"
    )
    print(
        f"{int(summary['num_volumes'])}-volume mean Dice "
        f"{summary['baseline_dice']:.6f} -> {summary['rankseg_dice']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
