# RankSEG mini-benchmark screening acceptance — 2026-09-22

**Status: measurements complete; numerical checks passed.**

## Main findings

All **6,012 inputs** completed for all five methods, with no OOM or skipped
numerical checks. KiTS comprises **210 cases / 5,712 slices / five folds**.
Independent metric, timing, memory, cohort and source/cache audits passed.
The final benchmark repository CPU/CUDA test run passed **115 tests**
(`artifacts/screening-final-acceptance-2026-09-22-v2/tests.log`).

- Normal `safe_screening=True` improves natural-image decoder latency by
  **4.84× / 9.78× / 12.43×** versus ordinary off for VOC / Cityscapes / ADE20K,
  while reducing mean extra peak allocated memory by **88.51% / 87.68% / 88.35%**.
  The largest absolute dataset-mean Dice difference is **0.000355 pp**.
- Against the distinct optimized-full auxiliary control, natural-image speedups
  are **3.28× / 6.39× / 8.15×** and memory reductions **88.51% / 77.59% / 85.74%**.
- Normal KiTS dispatch bypasses screening for **all 5,712 inputs**, retains
  exactly the ordinary off masks and metrics, and runs **1.34×** faster with
  **36.01%** less extra peak memory. That gain comes from the opt-in optimized
  full-sort path, **not** from screening entries out of the sort.
- Forcing KiTS screening reduces extra peak memory from **4.6800 to 1.8291 MiB**
  (**60.92%**) but increases mean time from **0.2249 to 0.3969 ms/slice**
  (**76.50%**). The matched auxiliary full-sort control is **0.2444 ms**;
  forced screening is still **62.40% slower** than that scoped-override control.
  The difference between ordinary bypass and the auxiliary full-sort timing
  also includes control-wrapper overhead, so it is not a mathematical speedup.
  These results support retaining the current production small-input bypass.
- Maximum binary-objective regret is **2.2564 float32 eps** across all paths,
  below the unchanged four-eps budget. For the 300 natural images the production
  maximum is **0.9229 eps**. Forced KiTS screening changes 504 slice masks,
  with aggregate Dice changing only **−0.000066 pp**; it is not bitwise identical.
- Natural-image masks may change too. The worst per-image Dice decline is
  **0.0897 pp** versus ordinary off, or **0.1753 pp** versus optimized full,
  both on Cityscapes. Aggregate stability is not a per-image guarantee.

## Scope and protocol

- VOC, Cityscapes and ADE20K: the first 100 stored images each, selected without looking at outcomes; **subsets**, not their full-dataset README results.
- KiTS: **all 5,712 stored slices in all five folds**. RankSEG receives only the foreground channel in multilabel mode. Argmax retains both original channels.
- All caches use Hugging Face revision `1884f0766268cdc62730e696f24dcc913d551b35`. The missing natural-image prefixes were restored from this same revision; KiTS reused existing local files.
- MONAI Pancreas/Spleen are **not included**: their local probability manifests were unavailable. No substitute nnU-Net caches, checkpoint downloads, or new inference were used.
- Same frozen RankSEG snapshot as the 778-volume nnU-Net acceptance. No core algorithm edits, training, checkpoint changes, or parameter fitting.
- RTX 3090, Torch 2.11.0+cu130, Triton 3.6.0, float32, batch size 1, four CPU threads. Two warm-ups and seven synchronized repeats in rotating method order.
- Timings include benchmark channel routing and output conversion; exclude loading/transfers, compilation, metrics, oracles and screening diagnostics. Means of per-input median decoder wall times, not inference speedups.
- Memory is extra peak PyTorch allocated GPU memory above resident original probabilities, including routing temporaries and output, excluding reserved memory and CUDA context. Tables show means of sample peaks.
- Natural-image scores: mean over GT-present classes per image, then mean over images. KiTS: foreground slice metrics, then mean within case, across cases within fold, and across folds, preserving the repository's empty-slice convention. **Not whole-volume Dice.**
- Natural-image ground-truth values outside [0, C) are excluded, in addition to ignore_index=255, matching the existing accumulator (e.g. VOC caches contain void label 21). KiTS retains its separate existing label-clamping convention. Labels never affect decoding or screening.
- Dice/RMA, smooth=0, pruning_prob=0.5, max_score. IoU is measured but not separately optimized.

## Methods

`full` is normal public `safe_screening=False`; `screened` is normal public `safe_screening=True`.
`full_optimized` is a separate direct-argmax full-sort auxiliary control; `screened_forced` overrides only the small-input bypass.
The ordinary on/off paths do not pay for dispatch monkeypatching. Auxiliary controls use scoped overrides restored after each call.

## Quality, decoder runtime and memory

| Dataset | Method | Dice (%) | IoU (%) | Mean ms/input | Mean peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| pascal_voc | argmax | 88.9594 | 85.5419 | 0.0545 | 2.0000 |
| pascal_voc | full | 89.3309 | 85.9592 | 3.1712 | 256.7588 |
| pascal_voc | full_optimized | 89.3312 | 85.9597 | 2.1502 | 256.7588 |
| pascal_voc | screened | 89.3311 | 85.9595 | 0.6551 | 29.5029 |
| pascal_voc | screened_forced | 89.3311 | 85.9595 | 0.6463 | 29.5029 |
| cityscapes | argmax | 81.3147 | 74.4120 | 0.2279 | 16.0000 |
| cityscapes | full | 81.9421 | 74.9180 | 19.6197 | 1688.0010 |
| cityscapes | full_optimized | 81.9433 | 74.9197 | 12.8309 | 928.0010 |
| cityscapes | screened | 81.9418 | 74.9181 | 2.0068 | 208.0029 |
| cityscapes | screened_forced | 81.9418 | 74.9181 | 1.9875 | 208.0029 |
| ade20k | argmax | 64.3637 | 57.1359 | 0.2472 | 2.3989 |
| ade20k | full | 65.4416 | 58.0087 | 23.2879 | 1981.5250 |
| ade20k | full_optimized | 65.4419 | 58.0090 | 15.2689 | 1619.3016 |
| ade20k | screened | 65.4419 | 58.0088 | 1.8729 | 230.8809 |
| ade20k | screened_forced | 65.4419 | 58.0088 | 1.8539 | 230.8809 |
| kits | argmax | 61.1566 | 54.1836 | 0.0210 | 1.1250 |
| kits | full | 63.5350 | 56.2096 | 0.3021 | 7.3135 |
| kits | full_optimized | 63.5350 | 56.2096 | 0.2444 | 4.6800 |
| kits | screened | 63.5350 | 56.2096 | 0.2249 | 4.6800 |
| kits | screened_forced | 63.5350 | 56.2095 | 0.3969 | 1.8291 |

## Production on/off comparison

All paired samples are retained. Speedup is full / screened; negative memory reduction is a regression.

| Dataset | Samples | Speedup | Mean peak reduction | Dice Δ (pp) | IoU Δ (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pascal_voc | 100 | 4.841× | 88.509% | 0.000198 | 0.000380 |
| cityscapes | 100 | 9.777× | 87.678% | -0.000354 | 0.000026 |
| ade20k | 100 | 12.434× | 88.348% | 0.000247 | 0.000120 |
| kits | 5712 | 1.343× | 36.009% | 0.000000 | 0.000000 |

## Screening proportion versus actual dispatch

Potential H+L counts are measured in a separate untimed forced-certificate call. They exclude class-pruned rows and count class-probability entries, not unique pixels. Production sorting avoidance respects both bypass and workspace fallback.

| Dataset | Potential H+L / active | Production sort avoidance / active | Small-input bypass samples | Observed screening calls | Workspace fallback rows (forced) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pascal_voc | 99.2785% | 99.2785% | 0/100 | 100 | 0 |
| cityscapes | 99.7628% | 99.7628% | 0/100 | 100 | 0 |
| ade20k | 98.7293% | 98.7293% | 0/100 | 100 | 0 |
| kits | 99.9135% | 0.0000% | 5712/5712 | 0 | 0 |

## Correctness and observed differences

Every sample/method is checked against the same exhaustive full-prefix float64 binary-objective oracle, with the unchanged four-float32-eps budget. No epsilon-based full-sort retries are added. This is a binary-objective regression check, not a guarantee of identical final multiclass labels or ground-truth scores.

| Dataset | Method | Passed / failed | Maximum regret (float32 eps) |
| --- | --- | ---: | ---: |
| pascal_voc | full | 100 / 0 | 0.741001 |
| pascal_voc | full_optimized | 100 / 0 | 0.741001 |
| pascal_voc | screened | 100 / 0 | 0.586366 |
| pascal_voc | screened_forced | 100 / 0 | 0.586366 |
| cityscapes | full | 100 / 0 | 0.714323 |
| cityscapes | full_optimized | 100 / 0 | 0.714323 |
| cityscapes | screened | 100 / 0 | 0.640254 |
| cityscapes | screened_forced | 100 / 0 | 0.640254 |
| ade20k | full | 100 / 0 | 0.761497 |
| ade20k | full_optimized | 100 / 0 | 0.761497 |
| ade20k | screened | 100 / 0 | 0.922848 |
| ade20k | screened_forced | 100 / 0 | 0.922848 |
| kits | full | 5712 / 0 | 2.256365 |
| kits | full_optimized | 5712 / 0 | 2.256365 |
| kits | screened | 5712 / 0 | 2.256365 |
| kits | screened_forced | 5712 / 0 | 1.036400 |

pascal_voc, screened vs full: 84/100 changed masks; 3455 differing pixels; worst scored image/slice Dice Δ -0.014303 pp (fold=None, source row=47, case=None).

pascal_voc, screened vs full_optimized: 84/100 changed masks; 3248 differing pixels; worst scored image/slice Dice Δ -0.028643 pp (fold=None, source row=4, case=None).

cityscapes, screened vs full: 100/100 changed masks; 100864 differing pixels; worst scored image/slice Dice Δ -0.089683 pp (fold=None, source row=38, case=None).

cityscapes, screened vs full_optimized: 100/100 changed masks; 94105 differing pixels; worst scored image/slice Dice Δ -0.175245 pp (fold=None, source row=38, case=None).

ade20k, screened vs full: 99/100 changed masks; 6935 differing pixels; worst scored image/slice Dice Δ -0.016617 pp (fold=None, source row=10, case=None).

ade20k, screened vs full_optimized: 99/100 changed masks; 6683 differing pixels; worst scored image/slice Dice Δ -0.016524 pp (fold=None, source row=10, case=None).

kits, screened vs full: 0/5712 changed masks; 0 differing pixels; worst scored image/slice Dice Δ 0.000000 pp (fold=0, source row=0, case=10).

kits, screened vs full_optimized: 0/5712 changed masks; 0 differing pixels; worst scored image/slice Dice Δ 0.000000 pp (fold=0, source row=0, case=10).

## Validation and artifacts

- Full benchmark repository CPU/CUDA regression tests, including end-to-end report generation and out-of-range/void-label handling, pass. Static checks for the new harness/tests pass; run logs retain exact test counts.
- Confusion counts and metric aggregation are independently computed using NumPy and compared with the unchanged repository accumulators. Per-sample timing medians, memory aggregates, numerical status, cohort completeness and source/cache hashes are audited again after the run.
- Shared input tensors are checked for exact non-mutation on every sample. Untimed hooks verify actual production dispatch, and are removed before timing.
- The real-cache runs complete normally or stop visibly on structural errors/OOM; no CPU retry, resized input, or silent dropped sample is allowed. Numerical budget failures remain recorded failures.
- This is empirical acceptance on these inputs, not a proof for all possible inputs. Natural-image subsets cannot replace the published full-dataset evaluation.

Raw reports, per-sample JSONL counts/timings and hashes are in `artifacts/screening-final-acceptance-2026-09-22-v2/` (local, ignored by Git). The initial attempt stopped at VOC's first sample because the new independent audit rejected void label 21. Only the audit was corrected; the complete v2 run starts afresh, and no measurements from the failed attempt are reused.
Reproduce using fresh output paths:

```bash
../env/bin/python -B scripts/benchmark_screening_acceptance.py \
  --rankseg-path ../rankseg-nnunet-benchmark/outputs/screening-final-acceptance-2026-09-22/snapshot \
  --output-dir artifacts/screening-acceptance-new --warmup 2 --repeats 7
CUDA_VISIBLE_DEVICES='' ../env/bin/python -B scripts/summarize_screening_acceptance.py \
  artifacts/screening-acceptance-new --document docs/SCREENING_ACCEPTANCE_NEW.md
```
