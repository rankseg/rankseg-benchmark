# MONAI safe-screening benchmark — 2026-10-07

This is a paired, decoder-only comparison of `safe_screening=False` and
`safe_screening=True` for **RMA Dice, `smooth=0`, foreground-only 3D volumes**.
The BTCV Swin UNETR 0.5.8 checkpoint generated each probability volume once;
both decoder modes then read the same float32 cache. No model retraining,
checkpoint selection, threshold fitting, or test-case-driven parameter tuning
was performed. The fixed MSD Pancreas and Spleen cohorts are external tests by
the [declared dataset provenance](README.md#checkpoint-provenance), not by an
independent patient-identity audit.

The tested RankSEG source was the **local 0.0.7 working tree**, based on commit
`36e64890b206e0865aeab8fb4c249483d961839a` with the uncommitted
stable-increment changes. This is **not a measurement of the published 0.0.7
wheel**. The benchmark checkout was
`f15dd06b39d76478eb4e3697fdd8a042783f1a58`; CUDA ran on an RTX 3090 with
Torch 2.11.0+cu130 and local MONAI `0+untagged.3466.gfcee761`.

## Segmentation quality

Scores are foreground-only **means over complete 3D volumes**, in percent;
Δ is screened minus full sort in percentage points. Argmax uses the same
probabilities as a reference.

| Cohort | Argmax Dice / IoU | Full sort Dice / IoU | Screened Dice / IoU | Δ Dice / IoU (pp) |
| :--- | ---: | ---: | ---: | ---: |
| MSD Pancreas, 20 volumes | 50.29525 / 35.29150 | 54.86957 / 39.18723 | 54.86328 / 39.18107 | −0.00629 / −0.00617 |
| MSD Spleen, 9 volumes | 94.79003 / 90.12413 | 94.75256 / 90.05585 | 94.75158 / 90.05425 | −0.00098 / −0.00160 |

The official benchmark runner independently reproduced the Argmax and
full-sort cohort scores exactly. The small Spleen decline versus Argmax is
present in full sort too; screening does not turn it into a gain.

## Decoder cost and screening extent

Latency is the **mean of each volume's median of nine synchronized, paired
CUDA calls**, with two untimed warmups per mode. Calls were rotated to reduce
order bias. The probability input was already on GPU; model inference,
preprocessing, cache reads, data transfers, metric updates, and diagnostics
were excluded from timing. Memory is the mean **incremental peak allocated
GPU memory** per volume with the input resident, not total process or reserved
CUDA memory.

| Cohort | Full sort → screened (ms/volume) | Speedup | Incremental peak (MiB) | Peak reduction | Pixels needing candidate sort |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Pancreas | 3.429 → 0.573 | 5.98× | 483.0 → 90.6 | 81.23% | 41,507 / 210,956,342 (0.01968%) |
| Spleen | 3.613 → 0.582 | 6.20× | 511.1 → 95.9 | 81.24% | 4,021 / 100,489,709 (0.00400%) |

Candidate counts come from the fused Triton screening kernel outside the
timed region. The complement is certified positive or negative and skipped
by candidate sorting; **no complete volume was class-pruned**. The 99.98%+
candidate reduction should not be interpreted as an equal memory reduction:
outputs, metadata, and other fixed-size workspaces remain necessary.

Screened and full-sort masks were not bitwise identical in the recorded run:
67 pixels differed across Pancreas and 112 across Spleen. An independent float64 evaluation of
the RMA Dice objective found a maximum absolute difference of
`6.33e-8` (Pancreas) and `2.00e-7` (Spleen) per volume; both positive and
negative differences occurred. In a separate 50-repeat check of three
boundary cases, full sort varied by one pixel on `pancreas_117` and
`spleen_6`; screened output was stable in all three. Thus the recorded
different-pixel totals can shift slightly on rerun, and exact pixel equality
is not a supported claim at these large float32 volumes. These observations do not establish a general
accuracy improvement; they show a large speed/memory benefit with a small
measured score change for these two cohorts.

The complete per-case timings, scores, screening counts, and changed-pixel
counts are in the local ignored artifact
`artifacts/monai_screening_2026-10-07/results.json`; independent objective
checks are in `objective_audit.json` alongside it. Probability caches and the
official checkpoint remain under ignored `artifacts/` and are **not committed**.
