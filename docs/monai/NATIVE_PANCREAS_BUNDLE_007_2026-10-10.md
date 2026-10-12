# MONAI DiNTS Pancreas Bundle native-grid test — RankSEG 0.0.7

The released [MONAI Pancreas CT DiNTS Bundle](https://github.com/Project-MONAI/model-zoo/blob/dev/models/pancreas_ct_dints_segmentation/docs/README.md)
was evaluated on the 49 independently labeled MSD Task07 `imagesTs` scans in
the [post-hoc expert annotation release](https://zenodo.org/records/5510650).
These masks are not the original MSD challenge's hidden test labels. The 49
image IDs are absent from all 281 `imagesTr` scans used for model development.
All 49 have binary pancreas and tumor masks aligned to their native CT grid.
Tumor overrides the whole-pancreas mask where they overlap, giving background,
pancreas, and tumor as three exclusive classes.

| Metric | Argmax | RankSEG-RMA | Paired improvement | 95% paired bootstrap interval |
| :--- | ---: | ---: | ---: | ---: |
| Foreground macro Dice | 55.474% | 58.018% | **+2.545 percentage points** | +1.649 to +3.534 pp |
| Foreground macro IoU | 43.209% | 45.193% | +1.984 pp | +1.248 to +2.763 pp |
| Pancreas Dice | 74.402% | 74.769% | +0.367 pp | −0.058 to +0.809 pp |
| Tumor Dice | 36.545% | 41.268% | **+4.722 pp** | +3.165 to +6.489 pp |

The case-level foreground macro Dice improved in **38/49** cases and fell in
11/49. Median paired improvement was +1.983 pp. Scores are averaged across
cases; each case's macro score averages its two foreground classes. The
intervals are descriptive percentiles from 20,000 paired case bootstrap
resamples with NumPy seed 123. The pancreas-only interval includes zero, so
the aggregate gain is principally a tumor gain.

## Method and audit

MONAI 1.4.0 loaded the frozen DiNTS architecture and checkpoint. Both arms
used the Bundle's preprocessing, `96³` sliding-window inference with window
batch size `4` and overlap `0.625`, AMP, softmax, and
`Invertd(nearest_interp=False)`. Only the final native-grid decoder changed:
the Bundle's Argmax versus released RankSEG **0.0.7** with
`metric="dice"`, `solver="RMA"`, `output_mode="multiclass"`, and
**`safe_screening=True`**. Both outputs then passed through the Bundle's
`SaveImaged` configuration. The 98 saved NIfTI predictions match the decoded
label arrays exactly and have the original images' shapes and affines.

All 49 expert image/mask triplets passed a shape-and-affine audit. Both
annotation ZIP MD5 checks matched the publisher's values:
`a0c3c838496502105cf02a2217e1b586` for pancreas and
`f045275daee60be12b68d3e63a651d6d` for tumor. The DiNTS checkpoint
SHA-256 was
`a18ae8b837f6affe778d7e9f130e6045c04a6f7d5b5dd8470155b9a18b6bcb65`.
The raw result records every image and mask SHA-256 and every case-level score.

Decoder timing here is **one synchronized call per case**, averaged over 49
cases: 0.659 ms Argmax and 200.971 ms RankSEG. These are exploratory timing
figures, not a repeated timing benchmark. Mean model inference time was
20.914 seconds per volume; preprocessing, inverse transform, and saving are
excluded from that inference figure. RankSEG adds a material decoder cost but
less than 1% of the observed model inference time in this setup.

## Reproduce

```bash
PYTHONPATH=artifacts/monai_140_pkg:artifacts/rankseg_007_pkg \
python -m rankseg_benchmark.monai.native_pancreas \
  --dataset-root artifacts/datasets/Task07_Pancreas \
  --expert-root artifacts/monai_pancreas_external_2026_10_10/source \
  --bundle-root artifacts/monai_bundles/pancreas_ct_dints_segmentation \
  --output artifacts/monai_native_pancreas_expert49_007_screened_v140.json
```

Implementation: [`native_pancreas.py`](../../rankseg_benchmark/monai/native_pancreas.py).
Raw result: `artifacts/monai_native_pancreas_expert49_007_screened_v140.json`.
Saved predictions: `artifacts/monai_native_pancreas_predictions_007/`.

The second 42-case CPTAC-PDA external-image subset from the same expert
annotation release has now [completed](NATIVE_CPTAC_BUNDLE_007_2026-10-11.md):
its macro Dice improved by +3.672 pp. The model's full checkpoint exposure
outside the declared MSD training split cannot be proven from these files
alone, so the independence claim for this 49-case analysis is limited to the
MSD `imagesTr` / `imagesTs` split and the frozen released checkpoint.
