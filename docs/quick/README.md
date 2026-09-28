# Quick benchmark

Run commands from the repository root. Quick uses hosted probabilities for
VOC, Cityscapes, ADE20K, and KiTS; no model inference is required.

```bash
python -m pip install -e . "rankseg==0.0.7"
rankseg-bench quick --dataset pascal_voc --limit 5 --device cpu
rankseg-bench quick --list-datasets
rankseg-bench quick --help
```

Remove `--limit` for the full hosted dataset. `--metric iou` selects the IoU
objective; both Dice and IoU are reported. For KiTS, limits apply per fold.

Natural-image metrics average over GT-present classes per image, then images.
KiTS retains foreground slice → case → fold aggregation, including its existing
empty-slice convention. Neither is interchangeable with whole-volume nnU-Net
evaluation. Timing excludes inference, loading, and transfers.

The [archived 0.0.6 reference](REFERENCE_0.0.6.md) preserves the original result
tables, detailed protocol, custom-model API examples, and MONAI comparison.
Those numbers have not been relabeled as 0.0.7 measurements.

Canonical modules: `rankseg_benchmark.quick.datasets`, `.runner`, and `.metrics`.
The former `rankseg_benchmark.datasets`, `.runner`, and `.metrics` imports remain
aliases of those same modules.
