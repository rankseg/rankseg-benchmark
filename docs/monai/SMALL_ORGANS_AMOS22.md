# AMOS22 CT small-organ study

This study tests whether RankSEG gains are associated with foreground size,
rather than only with low baseline Dice. It applies the **frozen MONAI BTCV
Swin UNETR 0.5.8 checkpoint** to the **100 labeled CT cases in the AMOS22
official validation split**. These cases are separate from the checkpoint's
declared BTCV training data, but patient identity has not been independently
audited across upstream collections.

The six targets share **the same cases, model inference, preprocessing, and
softmax output**. They are evaluated separately as binary one-vs-rest tasks:

| Size group | AMOS label | Target | BTCV softmax channel |
| :--- | ---: | :--- | ---: |
| Small | 12 | Left adrenal gland | 13 |
| Small | 11 | Right adrenal gland | 12 |
| Small | 4 | Gallbladder | 4 |
| Medium | 10 | Pancreas | 11 |
| Large | 1 | Spleen | 1 |
| Large | 6 | Liver | 6 |

For each target, Argmax is computed over `[1-p(target), p(target)]`; RMA Dice
receives `p(target)` with `safe_screening=True`, `smooth=0`, and default
`pruning_prob=0.5`. This matches the one-vs-rest design of the earlier MSD
MONAI comparisons. It is **not** a 14-class exclusive segmentation comparison;
the six binary masks may overlap. It also does not benchmark a native
AMOS-trained model.

The primary endpoints are foreground-positive **per-case, per-organ Dice** and
the paired Dice change in percentage points. IoU and decoder-only CUDA time
are secondary. Empty-ground-truth cases are counted separately, not assigned
an artificial Dice of 1 or 0; their false-positive predictions are reported
separately. Each case records the native physical GT volume in mm³ and its
resampled model-grid voxel count. The generator rejects a case if resampling
erases an originally present target.

The study uses **every CT validation case**, with an exact 100-case count and
SHA-256 fingerprint of the sorted case IDs:
`6f4c322228a29b8eee612ccea0056759bbac6e0e704f21ce0957c1b16dcf3891`.
No cases or decoder parameters are chosen using the AMOS predictions. The
public [original AMOS archive](https://zenodo.org/records/7155725) is about
24 GB. For this run, the unchanged-NIfTI-layout [MedOtter CT mirror](https://huggingface.co/datasets/MedOtter/amos22-ct-dataset)
is pinned at commit `c67f7c01e66277038d87975b03b73ece489a3035`;
image/label SHA-256 hashes are recorded in each local case artifact, while the
cache artifact and MONAI Bundle checkpoint hashes are recorded in the
manifest. The mirror is a distribution convenience, not a separate source of
training data. Image/label files and probability caches remain ignored local
artifacts and are **not** committed.
For this run, all **200 image/label files** matched the official ZIP's member
sizes and CRC32 values; the first case's image and label also matched its
official bytes by SHA-256. This verifies transport integrity, not patient
identity across datasets.

## Reproduce

Download `valid/imagesVa/*.nii.gz` and `valid/labelsVa/*.nii.gz` from the
pinned mirror revision, or extract the original AMOS archive. For the mirror:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MedOtter/amos22-ct-dataset",
    repo_type="dataset",
    revision="c67f7c01e66277038d87975b03b73ece489a3035",
    allow_patterns=["valid/imagesVa/*.nii.gz", "valid/labelsVa/*.nii.gz"],
    local_dir="artifacts/amos22_ct_validation_mirror",
)
```

The dataset root may contain `valid/` or directly contain `imagesVa/` and
`labelsVa/`. With the MONAI BTCV 0.5.8 Bundle already under
`artifacts/monai_bundles/`:

```bash
python -m rankseg_benchmark.monai.small_organs cache \
  --dataset-root artifacts/amos22_ct_validation_mirror \
  --bundle-dir artifacts/monai_bundles \
  --output-dir artifacts/monai_amos22_ct_btcv \
  --device cuda

python -m rankseg_benchmark.monai.small_organs evaluate \
  --cache-dir artifacts/monai_amos22_ct_btcv \
  --output artifacts/monai_amos22_ct_btcv/results_wheel007.json \
  --device cuda --warmup 2 --repeats 9
```

`--limit 1` on `cache` and `--allow-partial` on `evaluate` are smoke-test
options only. A partial cache is rejected by default and must never be
reported as the cohort result. Per-case timings warm each organ/shape
separately and alternate nine synchronized Argmax/RMA calls with probabilities
already resident on GPU. Model inference and I/O are excluded.

## Full-cohort results

RankSEG **0.0.7**, 100 CT volumes, 600 case-organ evaluations, NVIDIA RTX 3090.
All rows are from the same frozen checkpoint and probability cache. Dice and
IoU are percentages averaged **per foreground-positive case**, not pooled
voxel metrics; decoder times are the mean of per-case median CUDA times in ms.
Mean native GT volume is shown only for positive cases. The cache was generated
with Python 3.10.12, PyTorch 2.11.0+cu130, and MONAI commit
`fcee761a3280cd740753331a913a65fa0d90ba89`; the checkpoint SHA-256 is
`52e7c3114444e41bb14f644e0dd2b7d42d70ad4b4dec0c1bfa4a552a4b92a096`.

| Target | Positive cases | Mean GT cm³ | Dice: Argmax → RMA | Δ Dice pp | IoU: Argmax → RMA | Decoder ms: Argmax → RMA | Wins / losses / ties |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Left adrenal | 100 | 3.75 | 38.21 → 41.19 | +2.97 | 27.16 → 28.79 | 0.330 → 0.495 | 63 / 28 / 9 |
| Right adrenal | 100 | 3.29 | 51.93 → 56.54 | +4.61 | 38.73 → 42.24 | 0.329 → 0.458 | 64 / 30 / 6 |
| Gallbladder | 93 | 29.97 | 49.70 → 54.32 | +4.61 | 38.88 → 42.97 | 0.330 → 0.460 | 83 / 4 / 6 |
| Pancreas | 100 | 71.46 | 36.57 → 47.22 | +10.65 | 25.04 → 33.21 | 0.331 → 0.482 | 98 / 0 / 2 |
| Spleen | 100 | 230.31 | 71.87 → 76.13 | +4.26 | 63.07 → 66.79 | 0.331 → 0.491 | 81 / 19 / 0 |
| Liver | 100 | 1,284.15 | 91.61 → 91.47 | −0.15 | 85.09 → 84.88 | 0.331 → 0.493 | 49 / 51 / 0 |

The seven cases without gallbladder GT have no Dice/IoU value in the table.
Five of them have false-positive gallbladder predictions under **both**
decoders. Across all seven, predicted voxels increase from **3,026** with
Argmax to **12,145** with RMA. This is an adverse result that the positive-case
Dice average alone would hide. No native-positive target disappeared after
resampling. The case-organ result JSON is saved locally at
`artifacts/monai_amos22_ct_btcv/results_wheel007.json`; data and probability
caches are intentionally not committed.

## Does size explain the gains?

Not on its own. The three smallest targets improve by **+2.97 to +4.61 pp**,
yet the medium-sized pancreas improves the most (**+10.65 pp**), the larger
spleen improves **+4.26 pp**, and the high-baseline liver slightly declines.
Across all 593 GT-positive case-organ observations, the small-target mean gain
is **+4.05 pp**, versus **+2.06 pp** for the two large targets. These are
descriptive groups with different organ identities and baseline Dice, not a
controlled estimate of organ-size causality.

A within-organ split by **median native GT volume** and **median baseline Dice**
is more revealing. Pooling those six within-organ 2×2 tables, the mean gains
are **+8.38 pp** (smaller / lower baseline, 160 observations), **+7.38 pp**
(larger / lower baseline, 137), **+1.29 pp** (smaller / higher baseline, 137),
and **+0.86 pp** (larger / higher baseline, 159). Here “smaller” means below
that organ's median, not smaller than every other organ. Lower baseline Dice
tracks the larger gain more consistently than size in this cohort. The 93
cases with all six GT targets present give a similar direction, but this
observational comparison cannot separate size, baseline quality, and model
calibration. Class-specific adverse cases and the absent-gallbladder behavior
must be retained in any report.

## Interpretation and limitations

Small-versus-large group averages are descriptive, not proof of a size effect:
organ identity, class calibration, and baseline Dice can all differ. The
case-organ JSON supports follow-up analyses including adverse cases and absent
classes. This is cross-dataset use of a BTCV model, not an AMOS-trained model
or a test of a native multiclass decoder. Patient-level overlap with the
checkpoint's upstream source datasets has not been independently audited.

Sources: [AMOS22 challenge](https://amos22.grand-challenge.org/),
[MONAI BTCV Bundle](https://github.com/Project-MONAI/model-zoo/blob/dev/models/swin_unetr_btcv_segmentation/docs/README.md).
