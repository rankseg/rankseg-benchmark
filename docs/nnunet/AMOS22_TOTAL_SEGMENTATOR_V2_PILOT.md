# Official TotalSegmentator v2 on AMOS22 CT: fixed 10-case pilot

This is a supplementary **second-checkpoint** test of the small-organ
hypothesis, not an AMOS22 full-cohort result. It uses the official
[TotalSegmentator v2.0.0 weights](https://github.com/wasserth/TotalSegmentator/releases/tag/v2.0.0-weights):
`Dataset291_TotalSegmentator_part1_organs_1559subj`, 3D full resolution,
`nnUNetTrainerNoMirroring`, fold 0, `checkpoint_final.pth`. The original model
archive SHA-256 is
`2cde8e6bcfb5b6a02183648a7036c3d6b28bc854576f8e1c34fbd6b4dd5c6c5b`.
This is a widely released checkpoint rather than one trained for the RankSEG
benchmark. Its training subjects are not fully enumerated publicly, so a
patient-level overlap audit with AMOS22 is **not** possible here.

The evaluation data are the 100 labeled CT cases in the AMOS22 official
validation split. To bound the high cost of native-grid 25-channel probability
inference, the pilot selects **every tenth case ID in the sorted fixed 100-case
list**, before model inference and without looking at labels or decoder
outcomes. This yields the ten cases `amos_0008`, `amos_0056`, `amos_0112`,
`amos_0155`, `amos_0203`, `amos_0233`, `amos_0283`, `amos_0309`, `amos_0333`,
and `amos_0364`. It is a deterministic convenience sample, not a random
or representative sample. The separate [MONAI BTCV study](../monai/SMALL_ORGANS_AMOS22.md)
uses **all 100** validation cases.

The six target organs and AMOS label IDs are the same as in that MONAI study:
left adrenal 12, right adrenal 11, gallbladder 4, pancreas 10, spleen 1, and
liver 6. Their TotalSegmentator v2 softmax channels are 9, 8, 4, 7, 1, and 5,
respectively. For each organ we compare binary Argmax of `[1-p, p]` with
RankSEG 0.0.7 RMA Dice (`safe_screening=True`) on the **same native-grid
probability**. This is one-vs-rest evaluation, **not** comparison with the
native mutually exclusive 25-class TotalSegmentator mask. The code still
checks that the returned 25-channel softmax Argmax exactly matches nnU-Net's
returned native hard segmentation for each case.

The nnU-Net `NibabelIOWithReorient` reader loads image and label in the same
canonical array coordinates used by the prediction API. Grid shape and
spacing are checked; the reader's label array for the first case was also
confirmed byte-for-byte against the benchmark's independent
`nnunetv2_reoriented` NIfTI reader. Per-case image and label SHA-256 hashes,
checkpoint/configuration hashes, software versions, and detailed results are
recorded in the ignored local JSON. No image, label, or checkpoint is added
to Git.

## Reproduce

Install the optional nnU-Net v2 dependencies and RankSEG 0.0.7, obtain the
official model archive and the [AMOS22 validation CT images and labels](https://zenodo.org/records/7155725),
then run from this repository root:

```bash
python -m pip install -e ".[nnunet-v2]" "rankseg==0.0.7"
python -m rankseg_benchmark.nnunet.amos_total_organs \
  --dataset-root artifacts/amos22_ct_validation_mirror \
  --model-dir /path/to/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres \
  --output artifacts/amos22_totalseg_v2_pilot10/results_wheel007.json \
  --device cuda --stride 10 --warmup 2 --repeats 9
```

The default `--stride 10` is the fixed pilot. `--stride 1` processes the full
100-case list, but is **not** the result reported here. Decoder timing warms
each organ/shape twice, alternates nine synchronized CUDA Argmax/RMA calls,
and excludes model inference, data transfer, and I/O. Per-case results are
saved atomically so a long run can resume after verifying checkpoint and
source hashes.

## Results and limitations

RankSEG 0.0.7, nnU-Net v2 2.6.2, RTX 3090. The checkpoint file SHA-256 is
`98c429f89fd9b3b08890fbb2b4c08c365ad4655fdcadb45efaa29380129fd890`.
All ten selected cases completed. Dice and IoU are mean **per GT-positive
case**, not pooled voxel scores. The decoder milliseconds are means of
per-case median timings. Inference and I/O are excluded.

| Target | GT-positive cases | Dice: binary Argmax → RMA | Δ Dice pp | IoU: Argmax → RMA | Decoder ms: Argmax → RMA | Wins / losses |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Left adrenal | 10 | 70.77 → 70.82 | +0.05 | 56.46 → 56.52 | 0.346 → 0.976 | 6 / 4 |
| Right adrenal | 10 | 72.21 → 72.33 | +0.11 | 57.06 → 57.19 | 0.346 → 0.967 | 8 / 2 |
| Gallbladder | 9 | 80.42 → 80.66 | +0.23 | 69.57 → 69.82 | 0.346 → 0.970 | 9 / 0 |
| Pancreas | 10 | 80.39 → 80.46 | +0.07 | 69.66 → 69.74 | 0.345 → 0.963 | 8 / 2 |
| Spleen | 10 | 95.86 → 95.87 | +0.02 | 92.09 → 92.12 | 0.345 → 0.960 | 10 / 0 |
| Liver | 10 | 96.95 → 96.95 | +0.01 | 94.08 → 94.09 | 0.344 → 0.968 | 7 / 3 |

The one GT-absent gallbladder case, `amos_0056`, has **823** false-positive
voxels under binary Argmax and **863** under RMA. The positive-case mean alone
would hide that adverse behavior. Other individual losses include pancreas
`amos_0056` (−0.10 Dice pp) and right adrenal `amos_0309` (−0.04 pp).

The same ten CT cases were included in the earlier frozen MONAI BTCV
100-case study. Re-filtering **only those ten case IDs** yields this
descriptive checkpoint contrast:

| Target | MONAI BTCV Argmax → RMA Dice | TotalSegmentator v2 Argmax → RMA Dice |
| :--- | ---: | ---: |
| Left adrenal | 39.05 → 42.61 | 70.77 → 70.82 |
| Right adrenal | 45.17 → 53.01 | 72.21 → 72.33 |
| Gallbladder (9 positive) | 41.37 → 46.15 | 80.42 → 80.66 |
| Pancreas | 29.37 → 36.65 | 80.39 → 80.46 |
| Spleen | 74.71 → 78.85 | 95.86 → 95.87 |
| Liver | 92.71 → 92.55 | 96.95 → 96.95 |

Both decoders within each checkpoint share probabilities, but the two
checkpoints use different preprocessing and evaluation grids. This table is
**not** a controlled model comparison and should not be pooled with the MONAI
100-case or FLARE22 results. It does show that even small organs need not gain
much when another recognized checkpoint already segments them well. Baseline
Dice alone is also not a sufficient rule: calibration, error location, and
case mix can all alter the RankSEG response. The ten-case convenience sample
is too small to estimate a reliable TotalSegmentator cohort effect, and its
training-patient overlap with AMOS22 remains unaudited.
