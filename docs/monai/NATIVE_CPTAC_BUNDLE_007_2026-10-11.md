# MONAI DiNTS Pancreas Bundle: 42-case CPTAC external test

The frozen [MONAI Pancreas CT DiNTS 0.5.2 Bundle](https://github.com/Project-MONAI/model-zoo/blob/dev/models/pancreas_ct_dints_segmentation/docs/README.md)
was evaluated on all 42 CPTAC-PDA scans with pancreas and tumor annotations
from the [expert annotation release](https://zenodo.org/records/5510650).
These external case IDs are distinct from the MSD Task07 `imagesTr` and
`imagesTs` case IDs used in the companion 49-case test. The inference code,
checkpoint, decoder settings, and scoring convention match that test.

| Metric | Bundle Argmax | RankSEG-RMA | Paired improvement | 95% paired bootstrap interval |
| :--- | ---: | ---: | ---: | ---: |
| Foreground macro Dice | 55.521% | 59.193% | **+3.672 percentage points** | +2.379 to +5.223 pp |
| Foreground macro IoU | 41.406% | 44.556% | +3.150 pp | +2.082 to +4.398 pp |
| Pancreas Dice | 68.467% | 69.396% | +0.929 pp | +0.174 to +1.779 pp |
| Tumor Dice | 42.576% | 48.991% | **+6.415 pp** | +4.062 to +9.339 pp |

The case-level foreground macro Dice improved in **37/42** cases and fell in
5/42; median paired improvement was +2.165 pp. Tumor Dice improved in 37/42
cases. Each case's macro score averages the two foreground classes; cases
are then averaged equally. The intervals are descriptive percentiles from
20,000 paired case bootstrap resamples with NumPy seed 123. This analysis did
not tune the model or RankSEG settings on CPTAC cases.

## Pipeline and data checks

The official Bundle preprocessing, `96³` sliding window with batch size `4`
and overlap `0.625`, AMP, softmax, `Invertd(nearest_interp=False)`, and
`SaveImaged` were retained. The sole comparison is the final decoder on the
native-grid three-channel probability volume: Argmax versus released RankSEG
**0.0.7** with `metric="dice"`, `solver="RMA"`,
`output_mode="multiclass"`, and **`safe_screening=True`**. Pancreas and tumor
masks were combined into three exclusive labels, with tumor overriding the
whole-pancreas mask where they overlap.

The 4.28-GB source image ZIP matched the publisher's MD5
`6068ad9fc389c56d6c448f4370af6051`. All 42 extracted CTs matched their
ZIP CRCs. Every CT, pancreas mask, and tumor mask had matching shapes and
affines. All **84 saved prediction NIfTIs** were checked against the decoded
label values and native CT grids. The released checkpoint SHA-256 was
`a18ae8b837f6affe778d7e9f130e6045c04a6f7d5b5dd8470155b9a18b6bcb65`.
The raw JSON records per-case image and mask SHA-256 hashes.

Decoder timing is one synchronized call per case, averaged across the 42
cases: 1.892 ms for Argmax and 447.233 ms for RankSEG. This is an exploratory
timing measurement, not a repeated timing benchmark. The mean measured model
inference time was 32.587 seconds per case; preprocessing, inverse transform,
and saving are outside that inference figure. The measured additional decoder
time is approximately 1.4% of that inference time, while the decoder-only
relative cost is substantial.

## Companion evidence and limits

The [49-case MSD `imagesTs` test](NATIVE_PANCREAS_BUNDLE_007_2026-10-10.md)
showed +2.545 pp macro Dice, and the separate
[19-case Prostate158 test](NATIVE_PROSTATE_BUNDLE_007_2026-10-10.md) showed
smaller gains with two readers. Across the **91 DiNTS cases** from MSD and
CPTAC, the case-weighted macro Dice was 55.496% for Argmax and 58.561% for
RankSEG (+3.065 pp; 75/91 cases improved). The two DiNTS cohorts remain
reported separately because their source populations differ.

This external result strengthens the case for MONAI to offer RankSEG as an
optional segmentation decoder. It does not show that the same gain holds for
other architectures, label definitions, hospitals, or modalities. The
released checkpoint's complete training and selection history cannot be
independently reconstructed from these files; the exposure claim rests on
the Bundle's declared MSD source and the distinct CPTAC cohort.

Implementation: [`native_pancreas.py`](../../rankseg_benchmark/monai/native_pancreas.py).
Raw result: `artifacts/monai_native_cptac_expert42_007_screened_v140.json`.
Saved predictions: `artifacts/monai_native_cptac_predictions_007/`.

Reproduce from the benchmark repository:

```bash
PYTHONPATH=artifacts/monai_140_pkg:artifacts/rankseg_007_pkg \
python -m rankseg_benchmark.monai.native_pancreas \
  --cohort cptac_pda \
  --dataset-root artifacts/monai_pancreas_external_2026_10_10/source/CPTAC_volumes \
  --expert-root artifacts/monai_pancreas_external_2026_10_10/source \
  --bundle-root artifacts/monai_bundles/pancreas_ct_dints_segmentation \
  --output artifacts/monai_native_cptac_expert42_007_screened_v140.json
```
