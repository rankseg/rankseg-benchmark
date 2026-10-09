# Expanded MONAI checkpoint benchmark — 2026-10-07

This evaluation brings **five MONAI targets** under one published-RankSEG-0.0.7
protocol. It adds two released MONAI Bundles to the existing BTCV Swin UNETR
experiment: **Whole Body CT Segmentation 0.2.7** (the 3 mm low-resolution
checkpoint) and **Ventricular Short Axis 3-Label 0.3.5**. Every comparison
uses a frozen checkpoint and one shared probability cache per target. RankSEG
and Argmax see identical probabilities; neither model nor postprocessing was
fitted to these test cases. The published Whole Body repository metadata lists
0.2.8, but the official weight repository exposed 0.2.7 as its newest tag at
the time of this run, so this benchmark pins the downloadable 0.2.7 artifact.

The CT checkpoints use **exactly the same fixed 20 MSD Pancreas and 9 MSD
Spleen volumes**. BTCV Swin UNETR was trained on BTCV and Whole Body on
TotalSegmentator, rather than those declared MSD tasks. The cardiac target uses all
**100 patients × 2 labeled frames = 200 3D frames** from the public ACDC
official training database. Its MONAI checkpoint was trained on private
King's College London data. These are dataset-level provenance declarations,
not an independent patient-identity audit of upstream datasets.

## Results

Scores are foreground-only **means of per-volume Dice**, in percent; each
ACDC frame is one 3D volume and each patient contributes two frames. The RMA
Dice decoder uses `safe_screening=True`, `smooth=0`, and the default
`pruning_prob=0.5`. Times are milliseconds per volume on an RTX 3090, measured
with the **same per-volume-warmed, paired protocol** for Argmax and RMA Dice +
screening. They exclude checkpoint inference, preprocessing, cache I/O,
host-to-device transfers, and metric updates.

| Checkpoint → external cohort | Volumes | Argmax Dice (%) | RMA Dice + screening (%) | Δ Dice (pp) | Decoder time: Argmax → RMA + screening (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| BTCV Swin UNETR → MSD Pancreas | 20 | 50.2953 | 54.8633 | +4.5680 | 0.218 → 0.435 |
| BTCV Swin UNETR → MSD Spleen | 9 | 94.7900 | 94.7516 | −0.0385 | 0.229 → 0.436 |
| Whole Body 3 mm → MSD Pancreas | 20 | 68.7115 | 69.2260 | +0.5145 | 0.050 → 0.317 |
| Whole Body 3 mm → MSD Spleen | 9 | 93.6003 | 93.5601 | −0.0402 | 0.055 → 0.337 |
| Ventricular 2D → ACDC 3D frames | 200 | 88.8790 | 88.8789 | −0.0002 | 0.034 → 0.424 |

These are complete, predeclared cohorts, not selected positive cases. RMA Dice
improves Pancreas but does not improve either Spleen cohort or ACDC. Decoder
time is **higher than Argmax** in every row; this table does not claim an
end-to-end application speedup. The earlier BTCV screening experiment is in
the [BTCV screening report](SCREENING_RESULTS_2026-10-07.md).

The BTCV and Whole Body checkpoints use different native preprocessing and
resampled evaluation grids (1.5 × 1.5 × 2.0 mm versus 3 mm isotropic), so
their absolute Dice values should **not** be interpreted as a controlled
head-to-head model ranking. The paired Argmax-versus-RankSEG comparison
*within* each row is the controlled postprocessing comparison.

## Why Spleen declines and ACDC barely changes

For Whole Body → Spleen, Argmax predicts **99,186** foreground voxels against
**95,721** ground-truth voxels across the nine volumes: it already tends to
oversegment. Screened RMA Dice adds **221** voxels and removes none. Of those
new voxels, **79** correct false negatives and **142** introduce false
positives. All nine cases expand; eight lose Dice and one gains. The added
voxels have mean model foreground probability **0.492**, but only **35.7%**
are foreground in the labels. This local probability-versus-outcome mismatch
explains why a model-probability-based Dice decision expands the mask even
though measured Dice falls. It is not caused by screening: full-sort RMA Dice
also falls, from 93.6003% to 93.5617%. Nine volumes are too few to infer a
general property of the checkpoint's calibration.

For Ventricular → ACDC, screened RMA Dice changes only **1,175 of
408,211,712** labels relative to Argmax (about **0.00029%**); 25 of 200 frames
are exactly unchanged, and at least half the frames have two or fewer altered
voxels. Two frames from patient057 account for 657 of the 1,175 changes. The
per-class Dice values nearly cancel: LV pool 93.3101% → 93.3108%, myocardium
88.0412% → 88.0417%, and RV pool 85.2858% → 85.2840%. The decoder is thus
almost making the same prediction as Argmax; there is little room for an
aggregate change under this frozen checkpoint. An additional interpretation
caveat is that multiclass RMA considers the background class, whereas the
reported metric averages only the three foreground classes; this mismatch
does not by itself establish the cause of the small observed change.

## Timing method and additional checks

The paired timer warms **each volume independently** (two calls per method),
then rotates Argmax, full-sort RMA, and screened RMA over nine synchronized
CUDA calls per method. The main table reports the mean of each volume's
nine-call median. Argmax runs on the complete two-/four-channel probability
tensor; the binary RMA objective uses the target foreground channel as defined
below. Both inputs are resident on GPU before timing. The full-sort comparison
is retained as a correctness control, not the headline runtime comparison.

<details>
<summary>Additional IoU, full-sort Dice, and screening checks</summary>

IoU evaluates the **same Dice-optimized predictions**, not a separate
IoU-objective run. Scores are percentages, with the same foreground-only
per-volume averaging as the main table.

| Target | Argmax IoU | Full-sort RMA Dice / IoU | Screened RMA IoU | Full → screened changed voxels |
| :--- | ---: | ---: | ---: | ---: |
| BTCV → Pancreas | 35.2915 | 54.8695 / 39.1871 | 39.1811 | 67 |
| BTCV → Spleen | 90.1241 | 94.7526 / 90.0559 | 90.0543 | 112 |
| Whole Body → Pancreas | 54.7138 | 69.2266 / 55.3149 | 55.3142 | 1 |
| Whole Body → Spleen | 88.0376 | 93.5617 / 87.9716 | 87.9687 | 3 |
| Ventricular → ACDC | 80.9224 | 88.8789 / 80.9222 | 80.9222 | 28 |

The counts are from the **published 0.0.7 wheel's** fused screening
certificate and include only classes active after pruning. In ACDC, 508 of
800 active class-volume rows had no undecided candidates. Candidate counts
are 41,507 / 210,956,342 active entries (BTCV Pancreas), 4,021 /
100,489,709 (BTCV Spleen), 1,229 / 34,192,965 (Whole Body Pancreas),
228 / 16,982,962 (Whole Body Spleen), and 1,234 / 408,211,712 (ACDC).

</details>

The generic one-pass benchmark runner instead warms only the first few
volumes. New shapes can trigger Triton JIT compilation inside later timed
calls, so its first-pass timing is **not a steady-state latency estimate**.
Use the paired, per-shape-warmed protocol for the decoder-time comparison
reported here.

## Reproducibility and interpretation

- Whole Body model channel 10 is pancreas and channel 1 is spleen. As with the
  BTCV experiment, each 105-channel softmax is collapsed to
  `[1-p(target), p(target)]` for the task's binary ground truth. MSD Pancreas
  labels 1 and 2 are both counted as foreground. The Bundle's actual 0.2.7
  3 mm preprocessing (RAS, spacing, nonzero normalization, intensity scaling)
  was compared against our generator on a real Spleen case and matched
  element-for-element.
- The cardiac Bundle is a 2D network applied slice-by-slice using its official
  `SliceInferer` 256 × 256 ROI. Probabilities are then assembled into each
  native-grid 3D ACDC frame. The Bundle output order is background/LV
  pool/myocardium/RV pool; ACDC's RV/myocardium/LV cavity labels are mapped
  to that order. Its intensity preprocessing was also checked against the
  Bundle config and matched element-for-element on a real frame.
- The tables use the **published PyPI RankSEG 0.0.7 wheel**, installed into
  an isolated temporary import path and verified by its module path. They ran
  on an RTX 3090 with Torch 2.11.0+cu130 and local MONAI
  `0+untagged.3466.gfcee761`. Checkpoint MD5s are
  `dbdebeae650f4fb5ea7f4ad01e0d2431`
  (Whole Body low resolution) and `0ffb4506fbfa4b3709a28a7f71122ec8`
  (Ventricular). Each generated case and manifest records SHA-256 provenance.
- Complete caches, manifests, result JSON files, and per-case paired timings
  remain under ignored `artifacts/monai_*`; no medical images, model weights,
  or probability tensors are committed. Reproduce steady-state timing with
  `python -m rankseg_benchmark.monai.paired --artifact-dir <cache> --output
  <results.json> --warmup 2 --repeats 9` after installing RankSEG 0.0.7.
  The results above use the `paired_argmax_wheel007.json` files and the
  corresponding `wheel007_screened_dice.json` quality reports.
  The [MONAI guide](README.md) gives cache-generation and quality commands.

Official Bundle references: [Whole Body CT](https://github.com/Project-MONAI/model-zoo/blob/dev/models/wholeBody_ct_segmentation/docs/README.md),
[Whole Body labels and metadata](https://github.com/Project-MONAI/model-zoo/blob/dev/models/wholeBody_ct_segmentation/configs/metadata.json),
[Ventricular Short Axis](https://github.com/Project-MONAI/model-zoo/blob/dev/models/ventricular_short_axis_3label/docs/README.md).
