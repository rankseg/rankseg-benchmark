# Screening pipeline optimization — 2026-09-21

The subsequent [2026-09-22 multiclass memory update](SCREENING_MEMORY_2026-09-22.md)
uses this report's optimized decoder as its baseline. The measurements below
remain the historical 2026-09-21 comparison.

## Scope

This is a **paired implementation comparison against the previous screened
decoder**, not a comparison against full sort or argmax. It does not replace
the complete 778-volume cohort evaluation. No model was retrained or rerun;
the existing held-out probability caches and label mappings were reused.

The default `safe_screening=False` remains unchanged. The optimized path is
still opt-in RMA Dice with `smooth=0`. Original probabilities, the `+1` in the
RMA denominator, class pruning, multiclass eligibility and incremental-score
assignment are retained. There is no epsilon-based tie pass or retry.

Implemented changes:

1. Fuse full-image sum, maximum and forced-positive statistics. Prepared CUDA
   screening no longer copies complete active probability rows.
2. Count candidates while constructing their mask, then compact indices with
   GPU block offsets in stable row-major order. CPU group-size metadata remains;
   this is not a synchronization-free implementation.
3. Separate the padded-group budget (1,048,576 elements) from the real unpadded
   candidate-row budget (4,194,304). Keep the 75% candidate-fraction fallback.
4. Fuse multiclass incremental scoring, eligibility and first-class argmax.
   The uniquely selected pixel statistics retain their original PyTorch
   reductions. Large unsupported class tiles use portable assignment.
5. Reuse the global maximum already read during input validation to skip
   candidate construction when all classes are pruned. Multiclass `max_score`
   still evaluates the original probabilities, not raw-probability argmax.

Full-image fused sums may round differently from `torch.sum`. Binary and final
multiclass masks are therefore **not promised to be bitwise identical**.

## Protocol and artifacts

- RTX 3090, one timed process at a time, four PyTorch CPU threads.
- Each input is shared by both implementations: three warm-ups, seven timed
  repeats per method, alternating before/after order. Reported dataset latency
  is the mean of per-input median synchronized wall times, including host
  scheduling. Loading, transfer, compilation, ground-truth metrics, official
  nnU-Net postprocessing and float64 oracles are outside timing.
- nnU-Net: PyTorch 2.8.0+cu128 / Triton 3.4.0; first held-out case in each of five
  folds for four datasets, plus the known `liver_43` workspace regression case.
  Cache/fold provenance is checked by `prepare_cases`.
- Mini benchmark: PyTorch 2.11.0+cu130 / Triton 3.6.0; fixed first 100 cached
  examples each for VOC, Cityscapes and ADE20K, using dataset revision
  `1884f0766268cdc62730e696f24dcc913d551b35`. KiTS uses the first 20 slices in
  each fold (100 total), foreground-only single-channel decoding. These are
  **sampled slices, not the complete KiTS case/fold evaluation**.
- Independent full-prefix float64 objective checks run for both decoders on
  every measured input. The regression limit is four float32 epsilons; this
  is an empirical check, not a universal numerical proof.
- Incremental peak memory excludes resident probabilities and the CUDA context.

Local raw records, source hashes, source snapshots and ablation files are under
`outputs/screening-update-2026-09-21/` (ignored artifacts, no checkpoints):

| Artifact | Purpose |
|---|---|
| `nnunet.json` | Final 21-volume paired comparison, metrics and objective checks |
| `mini-final-b1.json` | Final forced-screening mini comparison, B=1 |
| `kits-dispatch-b1.json`, `kits-dispatch-b4.json` | Production dispatch controls |
| `synthetic-final.json` | Final paired synthetic checks and timings |
| `liver_43-counts.json` | Matched candidate counts; collected outside timing |
| `baseline/rankseg/`, `optimized/rankseg/` | Before/after source snapshots |

`mini-b1.json` and the `rankseg-stage*-synthetic.json` files are intermediate
ablations, not the final numbers below. Final report source hashes were checked
against the final RankSEG source files.

## nnU-Net subset: further speedup over previous screening

Milliseconds per complete volume; the Liver row includes the extra regression
case and must not be interpreted as an unbiased full-cohort estimate.

| Dataset | Volumes | Previous screening | Optimized screening | Further speedup |
|---|---:|---:|---:|---:|
| Pancreas | 5 | 15.041 | 10.605 | 1.42x |
| HepaticVessel | 5 | 7.307 | 5.168 | 1.41x |
| Lung | 5 | 40.992 | 23.250 | 1.76x |
| Liver | 5 + 1 regression | 60.126 | 39.924 | 1.51x |

Mean foreground Dice/IoU on this subset, after identical configured official
postprocessing. Changes are **percentage points**, optimized minus previous
screening; they are not improvements over argmax.

| Dataset | Dice before → after (%) | Dice change | IoU before → after (%) | IoU change |
|---|---:|---:|---:|---:|
| Pancreas | 52.80236 → 52.80236 | 0 | 40.01227 → 40.01227 | 0 |
| HepaticVessel | 71.96881 → 71.96868 | -0.000139 | 58.41519 → 58.41506 | -0.000129 |
| Lung | 63.75571 → 63.75571 | 0 | 54.13971 → 54.13971 | 0 |
| Liver | 81.46927 → 81.49497 | +0.025702 | 74.32665 → 74.37100 | +0.044349 |

All 21 volumes passed the objective checks; the largest optimized-decoder
regret was **0.4811 float32 eps**. Pancreas and Lung final masks were identical;
HepaticVessel differed at one voxel, Liver at 22,422 voxels across its six
volumes. The largest absolute per-volume foreground-mean change was 0.1337 Dice
points / 0.2290 IoU points (Liver). Small binary objective regret does **not**
bound the final multiclass metric change by epsilon.

Maximum incremental peak allocation observed within each subset:

| Dataset | Before (MiB) | After (MiB) |
|---|---:|---:|
| Pancreas | 1,430.0 | 962.5 |
| HepaticVessel | 859.0 | 578.0 |
| Lung | 6,037.3 | 3,828.5 |
| Liver | 11,645.5 | 5,994.0 |

### `liver_43`: avoid an unnecessary full-row fallback

The tumour row has 40,632,320 probabilities: H=22,620, L=38,417,806 and
M=2,191,894. The certificate removes **94.6055%**. These counts did not change;
the workspace policy did. Previously M exceeded the padded-group limit and
triggered a full 40,632,320-element sort. Now the row is sorted independently:
**2,191,894 sorted entries, no fallback**.

Whole-volume latency decreased from **31.760 to 16.667 ms (1.91x)**, and peak
allocation from 2,054.8 to 1,356.3 MiB. This is a targeted regression case,
not evidence that every Liver case has the same gain.

## Mini benchmark

Forced screening, B=1; milliseconds per image/slice. All 400 inputs passed
their objective checks. Maximum optimized-decoder regret was **0.9229 eps**.

| Dataset | Samples | Previous screening | Optimized screening | Further speedup |
|---|---:|---:|---:|---:|
| VOC | 100 | 1.428 | 1.059 | 1.35x |
| Cityscapes | 100 | 7.904 | 4.604 | 1.72x |
| ADE20K | 100 | 6.983 | 4.664 | 1.50x |
| KiTS, forced screening | 100 slices | 0.292 | 0.311 | 0.94x — slower |

Mean per-image Dice/IoU changes (percentage points):

| Dataset | Dice change | IoU change |
|---|---:|---:|
| VOC | -0.000541 | -0.000928 |
| Cityscapes | +0.000579 | +0.000680 |
| ADE20K | +0.000362 | +0.000344 |
| KiTS sampled slices | +0.000049 | +0.000082 |

Across the natural-image samples the largest absolute per-image changes were
0.02857 Dice points / 0.05010 IoU points. Metrics use the existing mini
benchmark's per-image GT-present-class convention; the sampled KiTS values
must not be substituted into its complete case/fold-weighted result table.

### Keep the KiTS small-input bypass

Forced screening remains about **6.6% slower** than previous forced screening
on these KiTS slices, and is slower than sorting directly. The existing
small-input bypass is unchanged. With actual production dispatch:

| Batch | Before ms/slice | After ms/slice | Output |
|---|---:|---:|---|
| 1 | 0.1751 | 0.1581 | Bitwise identical |
| 4 | 0.1460 | 0.1462 | Bitwise identical |

B=1 benefits from the all-pruned shortcut; B=4 is effectively unchanged
(0.16% timing difference). Both controls passed objective checks; the largest
regret was 1.152 eps. No claim of universal speedup is made.

## Verification

- Full RankSEG suite, CUDA enabled: **1,180 passed, 11 skipped**, coverage 90.00%.
- PyTorch 2.8 screening regression suite: **359 passed, 3 skipped** (before the
  final one-test addition checking validation-range reuse).
- PyTorch 2.8 large-memory row-indexing/public-assignment suite: **43 passed**,
  including D=846*512*512=221,773,824 and small-tile CUDA launch limits.
- Screening proportion collector: **39 tests passed**.
- Default `safe_screening=False`: **216 before/after exact comparisons passed**
  across CPU/CUDA, Dice/IoU, three dtypes, both output modes, three smoothing
  values and three pruning thresholds.
- Sphinx HTML build with warnings as errors passed.

Focused tests cover strided inputs, ragged/padded candidates, exact bounds,
all-pruned and fully resolved rows, independent row/group budgets, stable GPU
packing, optional Triton fallback, exact multiclass scoring with fixed
statistics, int64 void labels and more than 256 classes.

Example rerun from the benchmark repository (use the corresponding environment
for each suite; do not run timed jobs concurrently):

```bash
env/bin/python -B scripts/benchmark_screening_update.py \
  --baseline outputs/screening-update-2026-09-21/baseline \
  --suite nnunet --output /tmp/screening-nnunet-new-run.json

../env/bin/python -B scripts/benchmark_screening_update.py \
  --baseline outputs/screening-update-2026-09-21/baseline \
  --suite mini --output /tmp/screening-mini-new-run.json
```

These tests and samples provide regression evidence, not a proof for every
input or a rerun of all 778 nnU-Net volumes / all 5,712 KiTS slices. A complete
cohort rerun remains the next validation step before changing the default.
