# MONAI Prostate Bundle native-grid test — RankSEG 0.0.7

This is the primary independent, native three-class test for the MONAI
integration proposal. The released [Prostate MRI Anatomy 0.3.6 Bundle](https://github.com/Project-MONAI/model-zoo/blob/dev/models/prostate_mri_anatomy/docs/README.md)
was run on all 19 cases in the publisher's separate
[Prostate158 test archive](https://zenodo.org/records/6592345). Central gland
and peripheral zone are evaluated as separate foreground classes, with both
available readers scored on the same 19 cases.

| Reader | Argmax Dice | RankSEG Dice | Paired improvement | 95% paired bootstrap interval | Improved cases |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 81.687% | 81.882% | +0.195 percentage points | +0.083 to +0.334 pp | 16/19 |
| 2 | 79.630% | 79.915% | +0.284 percentage points | +0.183 to +0.413 pp | 19/19 |

Dice is the per-case mean across the two foreground classes, averaged over
the 19 cases. The intervals are descriptive percentiles from 10,000 paired
case resamples with NumPy seed 123. The two readers are repeated annotations
of the *same* images, not 38 independent test cases.

| Reader | Argmax IoU | RankSEG IoU | Central gland Dice gain | Peripheral zone Dice gain |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 69.604% | 69.846% | +0.014 pp | +0.375 pp |
| 2 | 67.098% | 67.429% | +0.029 pp | +0.540 pp |

## Exact comparison

Both arms receive the same image, official Bundle preprocessing, frozen
checkpoint, three-channel logits, and sliding-window inference. The Bundle's
`96³` window, overlap `0.5`, window batch size `4`, AMP, and the Bundle's
`torch.backends.cudnn.benchmark=True` setting were retained. Argmax and
RankSEG-RMA decode the same model-grid probability
volume. RankSEG is the released `0.0.7` wheel with `metric="dice"`,
`solver="RMA"`, `output_mode="multiclass"`, and **`safe_screening=True`**.

Both decoded maps then pass through the Bundle's
`KeepLargestConnectedComponentd(applied_labels=[1, 2])`,
`Invertd(nearest_interp=False)`, and `SaveImaged(resample=False)`. The 38 saved
NIfTI files were checked against the tensors used for scoring. Scores use the
publisher's *native image grid* and original reader labels; the code checks
shape and affine equality. The Bundle's non-nearest inverse can produce
fractional label IDs after discretization. For evaluation only, both arms are
rounded with `torch.round` after the official inverse; the saved float output
is retained. There were 196,752 fractional native voxels in Argmax outputs
and 198,645 in RankSEG outputs across all cases.

The comparison isolates the decoder, but this is not a claim that safe
screening is bitwise identical to the unscreened RankSEG path. It changed
10,116 model-grid voxel assignments across 19 cases. Screening was explicitly
enabled as requested; the per-case counts are in the raw results.

The test archive MD5 matched `403b3736cb12013ed7585e07cbda6978`, and all
57 evaluated image/label files matched its ZIP CRC. Publisher split lists
contain 119 training, 20 validation, and 19 independent test cases with no
split-path overlap. The Bundle checkpoint SHA-256 was
`3c18e4c658bb088f551b1d63219ef8340fe6256d016c75fc140d3da49dda696d`.
The exact config and per-file hashes are recorded in the result JSON.

Synchronized decoder-only timing, averaged from the per-case median of seven
alternating calls after two warmups, was 0.184 ms for Argmax and 0.952 ms for
RankSEG per volume on the RTX 3090. Inference and postprocessing are excluded
from those decoder times. The mean one-pass model inference time was 177 ms.

## Reproduce

Run from the benchmark repository with MONAI 1.4.0 and RankSEG 0.0.7 on the
module path:

```bash
PYTHONPATH=artifacts/monai_140_pkg:artifacts/rankseg_007_pkg \
python -m rankseg_benchmark.monai.native_prostate \
  --dataset-root artifacts/prostate158_test_dataset/prostate158_test \
  --training-list-root artifacts/prostate158_train_lists/prostate158_train \
  --test-archive artifacts/prostate158_test.zip \
  --bundle-root artifacts/monai_bundles/prostate_mri_anatomy \
  --output artifacts/monai_native_prostate_test_007_screened_v140.json
```

Implementation: [`native_prostate.py`](../../rankseg_benchmark/monai/native_prostate.py).
Raw result: `artifacts/monai_native_prostate_test_007_screened_v140.json`.
Native predictions: `artifacts/monai_native_prostate_predictions_007/`.

The modest mean gain and small 19-case cohort support an integration trial,
not a general claim across MONAI tasks. The second three-class native-grid
[DiNTS Pancreas test](NATIVE_PANCREAS_BUNDLE_007_2026-10-10.md) is complete
on 49 post-hoc expert-annotated MSD `imagesTs` scans.
