# Frozen MONAI BTCV checkpoint on FLARE22 public labeled CT

This is an external-dataset evaluation of the **official MONAI BTCV Swin UNETR
0.5.8 checkpoint**, not an evaluation on FLARE22's hidden validation/test set.
The public 50 labeled FLARE22 cases are its *training* split, but they were not
used to train or tune the frozen BTCV checkpoint or the RankSEG decoder in this
study. Patient-level overlap with BTCV's original training collection has not
been independently audited.

All six one-vs-rest endpoints use the same 50 CT volumes and the same frozen
14-channel softmax. For each endpoint, Argmax of `[1-p, p]` is compared with
RankSEG 0.0.7 RMA Dice applied to `p` (`safe_screening=True`, other decoder
defaults unchanged). The binary masks are **not** a mutually exclusive
multiclass segmentation. Dice/IoU are means of per-case scores for
GT-positive cases; GT-absent cases and their false positives are reported
separately. Timings include only decoding of already-resident GPU probabilities,
not model inference or file I/O.

| Target | FLARE22 label | BTCV channel |
| :--- | ---: | ---: |
| Gallbladder | 9 | 4 |
| Left adrenal | 8 | 13 |
| Right adrenal | 7 | 12 |
| Pancreas | 4 | 11 |
| Spleen | 3 | 1 |
| Liver | 1 | 6 |

## Cohort integrity and primary analysis

The complete 50-case list was fixed before inference (SHA-256 of sorted IDs:
`b571037423bb9f5680871ea5005f090c31d9738d59a4c4aa8adb0e42fd1b032e`).
The [official Zenodo archive](https://zenodo.org/records/7860267) has MD5
`f6b1a257deb8ad98eb901328b0762cc4`; this run downloaded the pinned
[MedOtter mirror](https://huggingface.co/datasets/MedOtter/FLARE22) at
`ab0b99b53e2183fe59321b5888867c6c7cb0792a`. **All 100 image/label NIfTI
files matched the official ZIP members' uncompressed sizes and CRC32 values.**
Each input's SHA-256 and each cache artifact's SHA-256 are recorded locally.
Zenodo identifies Medical Decathlon as the source of the CT images and
pancreas annotations and AbdomenCT-1K as the source of other organ annotations;
FLARE22 is therefore an additional *annotation/cohort* evaluation, not an
independent CT image source relative to the entire MSD collection.

A pre-inference SHA-256 comparison identified four image-byte duplicates with
the earlier MSD Pancreas evaluation: `FLARE22_Tr_0005` / `pancreas_016`,
`0011` / `pancreas_056`, `0020` / `pancreas_102`, and `0023` / `pancreas_107`.
Their labels are different. The **predeclared primary analysis excludes those
four byte-identical images** to avoid counting previously observed images as
fresh evidence; a
secondary all-50 analysis is below. This is an evaluation-history overlap,
not evidence that the BTCV checkpoint was trained on those cases. Other
patient-level matches that do not preserve image bytes cannot be ruled out.

## Primary result: 46 images without byte-identical MSD matches

Frozen BTCV checkpoint SHA-256:
`52e7c3114444e41bb14f644e0dd2b7d42d70ad4b4dec0c1bfa4a552a4b92a096`.
Python 3.10.12, PyTorch 2.11.0+cu130, MONAI commit `fcee761`, RTX 3090.
All six targets are GT-positive in all 46 cases. Each decoder was warmed twice
per case/organ and timed for nine synchronized, alternating CUDA calls; the
table reports the mean of per-case median milliseconds.

| Target | Mean GT cm³ | Dice: Argmax → RMA | Δ Dice pp | IoU: Argmax → RMA | Decoder ms: Argmax → RMA | Wins / losses |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gallbladder | 35.36 | 71.32 → 74.18 | +2.86 | 58.83 → 61.62 | 0.208 → 0.416 | 43 / 3 |
| Left adrenal | 5.32 | 73.80 → 74.15 | +0.35 | 59.75 → 59.73 | 0.207 → 0.430 | 18 / 28 |
| Right adrenal | 4.48 | 72.47 → 73.96 | +1.48 | 59.43 → 60.55 | 0.207 → 0.419 | 25 / 21 |
| Pancreas | 95.27 | 63.79 → 67.23 | +3.45 | 48.15 → 51.75 | 0.209 → 0.418 | 46 / 0 |
| Spleen | 235.21 | 94.98 → 95.10 | +0.12 | 90.69 → 90.87 | 0.208 → 0.417 | 26 / 20 |
| Liver | 1,636.56 | 96.69 → 96.64 | −0.05 | 93.60 → 93.50 | 0.209 → 0.424 | 1 / 45 |

The four overlapping cases do not change the overall direction: across **all
50**, Dice changes are +2.82 (gallbladder), +0.72 (left adrenal), +1.39 (right
adrenal), +3.38 (pancreas), +0.10 (spleen), and −0.05 (liver) points. No
native-positive target disappeared after model resampling.

This replication does **not** support a generic "smaller organ means larger
gain" claim. The medium-sized pancreas gains most; the tiny left adrenal gains
little on average and loses Dice in 28/46 cases. Within each organ, splitting
at that organ's median baseline Dice gives a descriptive mean change of
**+2.53 pp** over 138 lower-baseline observations versus **+0.20 pp** over 138
higher-baseline observations. Splitting by physical volume as well gives:

| Within-organ size / baseline | Case-organ observations | Mean Δ Dice pp |
| :--- | ---: | ---: |
| Smaller / lower baseline | 74 | +2.82 |
| Larger / lower baseline | 64 | +2.20 |
| Smaller / higher baseline | 64 | +0.07 |
| Larger / higher baseline | 74 | +0.32 |

These observations share cases and differ by organ; they are **descriptive**,
not a causal estimate or a test that baseline Dice is the sole predictor.
RankSEG is not guaranteed to improve a particular image or organ. In
particular, the high-baseline liver is slightly worse on average. The 46-case
primary analysis is a cross-dataset replication, not an untouched FLARE22
challenge test score.

## Reproduce

Download the public 50 cases from Zenodo or the pinned mirror:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MedOtter/FLARE22",
    repo_type="dataset",
    revision="ab0b99b53e2183fe59321b5888867c6c7cb0792a",
    allow_patterns=["images/*.nii.gz", "labels/*.nii.gz"],
    local_dir="artifacts/flare22_mirror",
)
```

With the MONAI BTCV 0.5.8 Bundle in `artifacts/monai_bundles/`:

```bash
python -m rankseg_benchmark.monai.flare22 cache \
  --dataset-root artifacts/flare22_mirror \
  --bundle-dir artifacts/monai_bundles \
  --output-dir artifacts/monai_flare22_ct_btcv --device cuda

python -m rankseg_benchmark.monai.flare22 evaluate \
  --cache-dir artifacts/monai_flare22_ct_btcv \
  --output artifacts/monai_flare22_ct_btcv/results_wheel007.json \
  --device cuda --warmup 2 --repeats 9
```

The cache and detailed per-case result JSON are local ignored artifacts, not
committed. `--limit` and `--allow-partial` exist for smoke testing only; a
partial cache cannot be reported as a complete cohort. See the
[AMOS22 small-organ study](SMALL_ORGANS_AMOS22.md) for the other frozen-BTCV
cross-dataset cohort.
