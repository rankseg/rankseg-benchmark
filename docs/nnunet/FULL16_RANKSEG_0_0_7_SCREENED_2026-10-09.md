# Full-16 rerun: RankSEG 0.0.7 with safe screening

Run date: 2026-10-09. This rerun uses the same 16 fixed nnU-Net cohorts,
exported probability arrays, labels, official postprocessing, and evaluation
rules as the [historical Full-16 study](README.md). The only decoder under
evaluation is RankSEG 0.0.7 with `safe_screening=True`; Argmax receives the
same probability arrays. Screening was explicitly enabled for every dataset,
not selected by the library's `auto` policy.

These are our paired inference results on released model outputs, **not**
metrics published by nnU-Net or its model authors. The historical, checksummed
`evidence/nnunet` package remains unchanged. This rerun's detailed local
summaries and case-level CSVs are in the Git-ignored
`artifacts/full16_rankseg007_screened/` directory.

## Dice results

Each record below is one foreground class or evaluated region, **not** a
within-dataset average. For this exploratory classification, high difficulty
means Argmax Dice <70%, medium means 70% to <90%, and low means ≥90%. The
thresholds are descriptive, not a prespecified hypothesis test. Each record
has equal weight in the group summaries; classes from the same dataset are
correlated. `Valid/total` counts cases with defined Dice for that class. The
three BrainTumour rows are nested evaluation regions formed from the model's
exclusive output labels. `Δ` is RankSEG minus Argmax in percentage points.

| Difficulty | Argmax Dice range | Records | Improved / declined | Mean Δ Dice (pp) | Median Δ Dice (pp) |
|---|---:|---:|---:|---:|---:|
| High | <70% | 5 | 5 / 0 | +1.32030 | +1.70436 |
| Medium | 70% to <90% | 10 | 7 / 3 | -0.04636 | +0.02137 |
| Low | ≥90% | 16 | 12 / 4 | +0.01341 | +0.00657 |

| Difficulty | Dataset | Class / evaluated region | Valid/total | Argmax Dice (%) | 0.0.7 + screening Dice (%) | Δ Dice (pp) |
|---|---|---|---:|---:|---:|---:|
| High | MSD Colon | colon cancer primary | 126/126 | 48.91759 | 49.15858 | +0.24100 |
| High | MSD Pancreas | pancreas tumour | 281/281 | 54.65037 | 57.22576 | +2.57539 |
| High | MSD HepaticVessel | hepatic vessel | 303/303 | 64.72835 | 65.04884 | +0.32049 |
| High | MSD Liver | liver tumour | 120/131 | 65.78628 | 67.49064 | +1.70436 |
| High | MSD Prostate | peripheral zone | 32/32 | 66.13404 | 67.89428 | +1.76024 |
| Medium | MSD Lung | lung tumour | 63/63 | 72.28822 | 72.75215 | +0.46393 |
| Medium | MSD HepaticVessel | liver tumour | 303/303 | 72.53273 | 73.27905 | +0.74632 |
| Medium | MSD BrainTumour | enhancing tumour | 478/484 | 79.34149 | 79.40680 | +0.06532 |
| Medium | MSD Pancreas | pancreas | 281/281 | 82.14158 | 81.91609 | -0.22548 |
| Medium | MSD BrainTumour | tumour core | 484/484 | 83.37486 | 83.41277 | +0.03792 |
| Medium | SegTHOR | esophagus | 40/40 | 85.49781 | 85.47893 | -0.01888 |
| Medium | MSD Prostate | transition zone | 32/32 | 85.78014 | 84.06445 | -1.71569 |
| Medium | MSD Hippocampus | Posterior | 260/260 | 88.06751 | 88.07233 | +0.00482 |
| Medium | CHAOS MRI | spleen | 60/60 | 89.42461 | 89.59923 | +0.17462 |
| Medium | MSD Hippocampus | Anterior | 260/260 | 89.74944 | 89.75294 | +0.00350 |
| Low | ACDC | myocardium | 200/200 | 90.82449 | 90.85238 | +0.02789 |
| Low | SegTHOR | trachea | 40/40 | 90.87343 | 90.88131 | +0.00788 |
| Low | MSD BrainTumour | whole tumour | 484/484 | 91.00015 | 90.91901 | -0.08114 |
| Low | ACDC | right ventricle | 200/200 | 91.62896 | 91.37724 | -0.25172 |
| Low | PROMISE12 | prostate | 30/30 | 91.93567 | 91.88181 | -0.05386 |
| Low | CHAOS MRI | left kidney | 60/60 | 92.38598 | 92.74323 | +0.35725 |
| Low | CHAOS MRI | right kidney | 60/60 | 92.91439 | 93.01820 | +0.10381 |
| Low | MSD Heart | left atrium | 20/20 | 93.29433 | 93.29623 | +0.00191 |
| Low | CHAOS MRI | liver | 60/60 | 93.45059 | 93.52071 | +0.07013 |
| Low | Fluo-C3DH-A549 | cell | 90/90 | 93.94401 | 93.95268 | +0.00867 |
| Low | SegTHOR | aorta | 40/40 | 94.37565 | 94.38116 | +0.00551 |
| Low | ACDC | left ventricle cavity | 200/200 | 94.90798 | 94.90951 | +0.00153 |
| Low | SegTHOR | heart | 40/40 | 95.33305 | 95.32943 | -0.00362 |
| Low | MSD Liver | liver | 131/131 | 96.21178 | 96.21886 | +0.00708 |
| Low | CHAOS CT / TotalSegmentator v2 | liver | 20/20 | 97.08608 | 97.09215 | +0.00607 |
| Low | MSD Spleen | spleen | 41/41 | 97.23659 | 97.24370 | +0.00711 |

All five high-difficulty records improve, but the medium group has no clear
average gain: the Prostate transition zone declines by 1.71569 pp. These
groups are too small and dependent to support a claim of statistical
significance or a general rule that harder classes benefit more.

The following predeclared aggregate still averages foreground class means
within each dataset, then weights all 16 datasets equally; changing only the
table display does not change that scientific endpoint. Across 16 datasets and
2,181 cases, dataset-mean Dice is **83.63646%** for
Argmax and **83.84726%** for screened RankSEG, a mean difference of **+0.21080
pp**. Thirteen datasets improve and three decline; the median dataset
difference is **+0.00802 pp**. The exact two-sided sign-test p-value is 0.02127
and the Wilcoxon signed-rank p-value is 0.01309. At a ±1 Dice-pp case threshold,
245 cases improve, 1,803 are within the threshold, and 133 worsen. The worst
case is Colon `colon_107` at -44.64 pp; the best is `colon_108` at +51.42 pp.

The historical Full-16 study, which recorded RankSEG 0.0.5 without screening,
had a +0.20671 pp dataset-mean difference. Thus this rerun is +0.00409 pp
higher, but **this is not a controlled screening effect**: both the RankSEG
version and screening setting changed. Four separately completed 0.0.7
unscreened cohorts are available locally for a partial within-version check;
they are not a second Full-16 aggregate.

### Case-level baseline distribution

At the case level, 168 of 2,181 cases have Argmax foreground-macro Dice below
50%, and 413 are below 70%. These are descriptive counts across heterogeneous
datasets, not the class-record groups above, an inferential subgroup analysis,
or a new primary endpoint. Additional cohorts should be selected by
checkpoint/test-case independence and a fixed protocol, not by observed
RankSEG improvements.

## Timing interpretation

The summary files record synchronized one-pass decoder latency with
probabilities resident on the GPU, excluding label postprocessing. Those
measurements include first-use Triton kernel compilation for new volume shapes.
Consequently, e.g. Liver's 717.78 ms and Spleen's 617.65 ms **per-case medians
from this run are not steady-state decoder times** and should not be used as a
screening speedup/slowdown claim.

To separate this effect, we repeated each decoder on the first discovered real
case of six cohorts on the same RTX 3090. The warm values below are the median
of seven alternating RankSEG/Argmax calls after two warm-up pairs. These are
representative case measurements, **not** Full-16 runtime estimates.

| Real case | Shape (C × Z × Y × X) | Warm Argmax (ms) | Warm screened RMA (ms) |
|---|---:|---:|---:|
| Liver `liver_101` | 3 × 683 × 512 × 512 | 12.799 | 15.574 |
| Lung `lung_006` | 2 × 567 × 512 × 512 | 2.867 | 10.960 |
| Pancreas `pancreas_021` | 3 × 93 × 512 × 512 | 0.602 | 2.439 |
| Spleen `spleen_10` | 2 × 55 × 512 × 512 | 0.297 | 1.521 |
| SegTHOR `Patient_03` | 5 × 147 × 512 × 512 | 1.280 | 4.594 |
| CHAOS CT `CHAOSCT_001` | 25 × 96 × 512 × 512 | 4.185 | 10.733 |

This run does not establish a steady-state Full-16 timing comparison between
screened and unscreened 0.0.7. A future timing study should warm each shape
and measure both modes on the same cases.

## Validation and numerical note

- Every one of the 16 summaries records installed `rankseg=0.0.7`,
  `safe_screening=true`, and CUDA for every RankSEG case; no CPU fallback was
  used. The runtime was Python 3.10.12, PyTorch 2.8.0+cu128, Triton 3.4.0,
  and an NVIDIA GeForce RTX 3090.
- The 2,181 per-case Argmax Dice and IoU values exactly match the unchanged
  historical evidence, confirming identical baseline inputs and evaluation.
- The complete nnU-Net test suite passed: 262 tests. Changed-file lint and
  `git diff --check` also passed.
- Safe screening need not produce a byte-identical mask to the unscreened
  float32 full-sort path near flat RMA objective maxima. In PROMISE12 `Case08`,
  the foreground mask after multiclass assignment differs by 2,067 voxels,
  with screened Dice higher by about 0.496 pp. Direct float64 evaluation of
  the two binary RMA masks gives class-objective differences of order 10⁻⁸.
  This is an example of a tiny objective difference producing a visible GT
  metric change; do not describe the two modes as prediction-identical.

The fixed manifests still contain their historical `rankseg_source_commit`
provenance field. For this rerun, the summary's runtime package version and
explicit decoder flag identify what actually executed; the historical field
must not be read as the source revision of RankSEG 0.0.7. Release tag
`rankseg/v0.0.7` resolves locally to
`36e64890b206e0865aeab8fb4c249483d961839a`.
