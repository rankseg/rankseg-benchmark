# TIAToolbox 2.1.3: all five paired comparisons

RankSEG 0.0.8, Dice/RMA, `safe_screening=True`, `pruning_prob=0.5`,
`smooth=0`. Each pair uses identical saved model probabilities and
unchanged official postprocessing. No retraining, scaling, calibration
or threshold tuning. Scores are computed per image/ROI, then averaged.

## Primary comparisons

| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | Paired 95% CI (pp) | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MicroNet / CoNSeP | Nucleus Dice | 14 | 0.796590 | 0.801132 | +0.454 | +0.226 to +0.662 | 11/14 |
| MicroNet / Kumar | Nucleus Dice | 14 | 0.612634 | 0.649413 | +3.678 | +2.269 to +5.206 | 14/14 |
| HoVer-Net CoNSeP / CoNSeP | Nucleus Dice | 14 | 0.847074 | 0.847281 | +0.021 | -0.059 to +0.098 | 8/14 |
| HoVer-Net Kumar / Kumar | Nucleus Dice | 14 | 0.825251 | 0.826315 | +0.106 | -0.035 to +0.250 | 9/14 |
| FCN ResNet50 UNet / BCSS | Five-class ROI macro Dice | 45 | 0.591457 | 0.603061 | +1.160 | -0.375 to +3.027 | 21/45 |

The two CoNSeP rows share 14 images; the two Kumar rows share another
14 images. MicroNet uses the CoNSeP checkpoint for both cohorts, so
its Kumar row is a cross-dataset evaluation. Nucleus Dice is measured
on `inst_map > 0` after official instance postprocessing. HoVer-Net's
baseline thresholds its nucleus probability at 0.5; the column is
labelled Argmax for consistency with the other comparisons.

## Comparison with the historical 1.6.0 runs

| Checkpoint / cohort | 1.6 Argmax | 2.1.3 Argmax | 1.6 RankSEG gain (pp) | 2.1.3 RankSEG gain (pp) |
| --- | ---: | ---: | ---: | ---: |
| MicroNet / CoNSeP | 0.737604 | 0.796590 | +5.392 | +0.454 |
| MicroNet / Kumar | 0.592362 | 0.612634 | +5.485 | +3.678 |
| HoVer-Net CoNSeP / CoNSeP | 0.838820 | 0.847074 | +0.055 | +0.021 |
| HoVer-Net Kumar / Kumar | 0.827551 | 0.825251 | +0.196 | +0.106 |
| FCN ResNet50 UNet / BCSS | 0.592789 | 0.591457 | -0.097 | +1.160 |

These columns describe different inference pipelines. The old engine
uses black edge padding and the new engine uses white. BCSS additionally
changes its registered stride from 256 to 450 pixels. MicroNet also
changes its returned instance map's hole filling. See the
[migration audit](V2_MIGRATION.md) and the detailed
[MicroNet/CoNSeP report](V2_MICRONET_CONSEP.md).

## Instance metrics

| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | Paired 95% CI (pp) | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MicroNet / CoNSeP | AJI | 14 | 0.492571 | 0.479085 | -1.349 | -1.961 to -0.707 | 2/14 |
| MicroNet / CoNSeP | PQ | 14 | 0.467760 | 0.466972 | -0.079 | -0.446 to +0.329 | 5/14 |
| MicroNet / Kumar | AJI | 14 | 0.415033 | 0.444061 | +2.903 | +1.953 to +3.870 | 13/14 |
| MicroNet / Kumar | PQ | 14 | 0.351425 | 0.372456 | +2.103 | +1.380 to +2.797 | 12/14 |
| HoVer-Net CoNSeP / CoNSeP | AJI | 14 | 0.544137 | 0.544091 | -0.005 | -0.101 to +0.088 | 8/14 |
| HoVer-Net CoNSeP / CoNSeP | PQ | 14 | 0.514868 | 0.514198 | -0.067 | -0.179 to +0.053 | 5/14 |
| HoVer-Net Kumar / Kumar | AJI | 14 | 0.615664 | 0.616056 | +0.039 | -0.138 to +0.226 | 6/14 |
| HoVer-Net Kumar / Kumar | PQ | 14 | 0.599063 | 0.599584 | +0.052 | -0.179 to +0.276 | 9/14 |

AJI and PQ evaluate instances; a foreground Dice increase need not
improve either metric. Both methods retain the same official instance
postprocessor. RankSEG does not optimize AJI or PQ directly.

## BCSS: all five classes

| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | Paired 95% CI (pp) | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BCSS | Tumor | 45 | 0.828064 | 0.826528 | -0.154 | -0.707 to +0.325 | 21/45 |
| BCSS | Stroma | 45 | 0.759316 | 0.749677 | -0.964 | -1.964 to +0.070 | 13/45 |
| BCSS | Inflammatory | 45 | 0.676005 | 0.701085 | +2.508 | +0.809 to +4.541 | 25/45 |
| BCSS | Necrosis | 45 | 0.350710 | 0.419944 | +6.923 | +0.320 to +15.571 | 14/45 |
| BCSS | Other | 45 | 0.343192 | 0.318072 | -2.512 | -4.357 to -0.741 | 12/45 |

All 45 ROIs contribute to every class mean. Raw labels 0 and 7 are
excluded only during scoring. Other is included. A both-empty class
scores 1; a predicted class with no ground-truth pixels scores 0.
The necrosis gain is largely driven by three absent-class Dice switches
from 0 to 1; the [presence audit](V2_MIGRATION.md#bcss-class-presence-audit)
separates this effect from scores on ROIs with annotated necrosis.

## Secondary decoder configurations

| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | Paired 95% CI (pp) | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MicroNet / CoNSeP | Two-channel multiclass Dice | 14 | 0.796590 | 0.800361 | +0.377 | +0.178 to +0.558 | 11/14 |
| MicroNet / Kumar | Two-channel multiclass Dice | 14 | 0.612634 | 0.647358 | +3.472 | +2.082 to +5.014 | 14/14 |
| BCSS | Independent-class multilabel macro Dice | 45 | 0.591457 | 0.608918 | +1.746 | +0.244 to +3.578 | 25/45 |
| BCSS multilabel | Tumor | 45 | 0.828064 | 0.829403 | +0.134 | -0.140 to +0.491 | 26/45 |
| BCSS multilabel | Stroma | 45 | 0.759316 | 0.777072 | +1.776 | +0.933 to +2.767 | 33/45 |
| BCSS multilabel | Inflammatory | 45 | 0.676005 | 0.699864 | +2.386 | +0.605 to +4.451 | 25/45 |
| BCSS multilabel | Necrosis | 45 | 0.350710 | 0.421707 | +7.100 | +0.504 to +15.595 | 16/45 |
| BCSS multilabel | Other | 45 | 0.343192 | 0.316543 | -2.665 | -4.469 to -0.920 | 12/45 |

BCSS multilabel masks can overlap or leave pixels unassigned. They are
a diagnostic and do not form the single-label output required for
ordinary BCSS semantic segmentation.

## Reproduce this table

```bash
python -m rankseg_benchmark.tiatoolbox.table_v2 --output rankseg_benchmark/tiatoolbox/RESULTS_V2.md
```

Only NumPy and the tracked evidence CSVs are needed (including the
historical CSVs for the old/new comparison).
Paired bootstrap intervals use 10,000 image/ROI resamples and seed
20261026, preserving each evaluator's metric and method call order.
Use `--from-artifacts --check-artifacts` to verify the original run.
Full inference instructions and limitations are in [V2_MIGRATION.md](V2_MIGRATION.md).
