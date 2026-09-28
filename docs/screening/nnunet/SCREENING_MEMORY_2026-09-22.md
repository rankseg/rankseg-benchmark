# Multiclass screening memory optimization — 2026-09-22

This is a paired comparison against the **already optimized screening decoder
from 2026-09-21**, not against full sort or argmax. It reuses the same 21-volume
nnU-Net engineering subset and 400 mini-benchmark samples. It does not replace
the complete-cohort evaluation. No model inference, retraining, checkpoint or
probability-cache changes were made.

## What changed

The previous multiclass conversion called `overlap_preds.sum(dim=1)`. On the
tested CUDA versions, this materialized a full `int64` copy of the boolean
`B*C*D` mask. Together with the mask and the `B*D` count result, its peak was
approximately `B*D*(9*C+8)` bytes, independent of the number of sorting
candidates retained by screening.

The opt-in CUDA/Triton Dice path (`safe_screening=True`, `smooth=0`) now:

1. Dispatches before creating the portable dense count/unique/score tensors.
2. Stores a one-byte zero/unique/overlapping status per pixel, saturated at two
   (including the 256-class case, which must not wrap to zero).
3. Reduces unique-pixel counts and probability sums in bounded tiles, without
   a full unique mask or floating-point product tensor. Counts accumulate in
   integers; probability sums retain the working dtype. Large volumes use
   hierarchical reductions with bounded reduction tiles.
4. Reuses the existing fused incremental-score/eligibility/first-argmax kernel.

Original probabilities, full-image means, binary screening, class pruning,
overlap eligibility, all-pruned `max_score` fallback and `void` labels retain
their roles. Unsupported shapes and unavailable Triton use the portable path.
The default `safe_screening=False`, IoU and positive-smoothing paths are unchanged.

The unique-probability reduction order changes, so near-tied multiclass labels
are **not guaranteed to be bitwise identical**. No epsilon tie pass, extra
inference retry or score-based screening-eligibility restriction was added.

## Measurement protocol

- One RTX 3090 timed process at a time; four PyTorch CPU threads.
- nnU-Net: PyTorch 2.8.0+cu128 / Triton 3.4.0. First held-out case per fold for
  each dataset, plus the known `liver_43` workspace regression case.
- Mini: PyTorch 2.11.0+cu130 / Triton 3.6.0. The same first 100 cached examples
  each for VOC, Cityscapes and ADE20K; KiTS uses the first 20 foreground-only
  single-channel slices per fold, 100 total. Screening is forced for this
  KiTS control; production small-input dispatch is not changed.
- Three warm-ups, **11 alternating timed repeats** per method. Dataset latency
  is the mean of per-input median synchronized wall times.
- Memory is the mean of per-input peak **additional PyTorch allocated GPU
  memory**, including the result but excluding resident probabilities and the
  CUDA context. It is not reserved memory or model-inference memory.
- Loading, transfers, compilation, float64 oracles, metrics and configured
  official nnU-Net postprocessing are outside timing.

## Runtime and memory

Both columns compare the previous screened implementation with this update.
Memory percentages use the ratio of mean per-input peaks.

| Dataset | Samples | Runtime before → after (ms) | Speedup | Peak memory before → after (MiB) | Memory reduction |
|---|---:|---:|---:|---:|---:|
| nnU-Net Pancreas | 5 volumes | 10.645 → 6.348 | 1.68x | 894.3 → 306.7 | 65.71% |
| nnU-Net HepaticVessel | 5 volumes | 5.185 → 3.202 | 1.62x | 410.2 → 141.1 | 65.61% |
| nnU-Net Lung | 5 volumes | 23.238 → 9.918 | 2.34x | 2,631.3 → 1,113.3 | 57.69% |
| nnU-Net Liver | 5 + 1 regression volumes | 39.618 → 21.219 | 1.87x | 3,240.5 → 1,111.1 | 65.71% |
| VOC | 100 images | 1.017 → 0.755 | 1.35x | 49.3 → 10.6 | 78.55% |
| Cityscapes | 100 images | 4.565 → 2.705 | 1.69x | 358.0 → 76.7 | 78.58% |
| ADE20K | 100 images | 4.622 → 2.329 | 1.98x | 407.2 → 90.5 | 77.78% |
| KiTS, forced screening control | 100 slices | 0.294 → 0.293 | 1.00x | 0.269 → 0.269 | 0% |

All **321 multiclass samples** had lower median latency, with a minimum per-input
speedup of **1.23x**; this is not a universal latency guarantee for arbitrary
inputs. KiTS does not enter the modified multiclass conversion and is effectively
unchanged: 48/100 control samples measured slightly slower, but none by more than
3.4%, and the aggregate difference was about 0.15%.

The largest observed Liver allocation decreased from **5,994.0 to 2,055.3 MiB**,
saving about **3.85 GiB**. A paired synthetic `[1,3,16_777_216]` sparse input
decreased from **560.0 to 192.0 MiB**, and from **6.344 to 3.675 ms (1.73x)**.
For three classes the new low-candidate peak is approximately `12*D` bytes
(binary mask, status and int64 output), instead of `35*D`.

Screening proportion remains a reduction in sortable class-probability entries,
not total allocated memory. Dense binary masks, candidate masks during screening,
and final class-index outputs still exist even when almost all candidates are
screened out. Neither the old nor new memory reduction should be equated with
the screening percentage.

## Correctness and metrics

Independent full-prefix float64 binary objective checks passed for all 421
samples: maximum regret was **0.4811 float32 eps** for nnU-Net and **0.9229 eps**
for the mini benchmark. These are empirical regression checks, not a universal
numerical proof or a bound on ground-truth metric changes.

Mean metric changes below are **percentage points, after minus before**. Medical
metrics use the same foreground labels and official postprocessing as before;
mini metrics use the existing GT-present-class convention.

| Dataset | Changed final pixels, before postprocessing | Mean ΔDice (pp) | Mean ΔIoU (pp) |
|---|---:|---:|---:|
| Pancreas | 19 | -0.000368 | -0.000419 |
| HepaticVessel | 8 | +0.000103 | +0.000153 |
| Lung | 0 | 0 | 0 |
| Liver | 308 | -0.000514 | -0.000641 |
| VOC | 1,458 | +0.000714 | +0.001327 |
| Cityscapes | 33,068 | +0.000416 | +0.000722 |
| ADE20K | 2,241 | -0.000163 | -0.000250 |
| KiTS | 0 | 0 | 0 |

The largest absolute per-volume changes were **0.003083 Dice pp / 0.003847 IoU pp**;
the largest per-image changes in the mini subset were **0.029769 Dice pp /
0.052205 IoU pp**. Average stability must not be mistaken for identical masks.

### Existing long-prefix CUDA variability

An additional audit found that `liver_43` binary masks can also vary across
repeated calls to the **unchanged old decoder**. Isolating `torch.cumsum` on one
fixed 2,191,894-element candidate sequence reproduced different float32 prefixes
and selected candidate counts. Across eight alternating old/new calls, all
binary objectives remained within **0.4179 float32 eps** of the float64 oracle.
This is existing floating-point scan variability, not a change to screening
certificates introduced by the multiclass optimization.

Consequently, the paired final-label differences above should not all be
attributed to the new unique-pixel reductions. Reruns can produce slightly
different boundary counts. The additional assignment audit fixes **the same
binary mask and full-image means** for both implementations to isolate this
update; it independently checks eligibility and float64 incremental scores at
every changed label. No deterministic-scan override or retry was added to the
inference path.

This fixed-mask audit completed for all **421 samples**:

- Binary masks matched exactly in **420/421** paired calls. The exception was
  `liver_43` (82 binary pixels in this audit), with both binary objective regrets
  below 0.336 eps. All 400 mini-benchmark binary masks matched exactly.
- Holding the binary mask fixed, nnU-Net had **27 changed multiclass pixels**
  (19 Pancreas, 8 HepaticVessel); Lung and Liver had none. In particular, the
  paired Liver changes in the table above came from the existing binary scan
  variability, not the new multiclass statistics in this fixed-mask check.
- The mini benchmark had **36,767 changed multiclass pixels**, matching the
  paired run. Every changed choice in both suites was eligible under the
  original pruning/overlap rules.
- Maximum absolute incremental-score regret against the independent float64
  oracle was **1.4202 eps** for nnU-Net and **1.4851 eps** for mini, below the
  four-epsilon audit limit. For comparison, the old mini assignment reached
  **1.5705 eps** at these changed positions. These are incremental-score checks,
  distinct from the binary Dice-objective regrets reported above.

Regression checks:

- Full primary CPU/CUDA suite: **1,257 passed, 11 skipped**, coverage **88.92%**.
- PyTorch 2.8 screening suite with large-memory tests enabled: **440 passed**,
  including the 221,773,824-pixel Liver indexing/assignment regression.
- Before/after exact checks: **216 default-path**, **180 unsupported metric/smooth**,
  **18 CPU-screened**, and **36 binary-screening** configurations.
- New tests cover float32/float64 sums against an independent reference, exact
  counts above `2**24`, channel/spatial strides, tile tails, hierarchical reductions,
  256-class status saturation, all-pruned samples, int64-extreme void labels,
  unsupported-shape fallback and an explicit peak-memory regression threshold.
- Assignment tests separately validate statistics and require exact agreement
  with dense scoring given those statistics. An independent float64 incremental
  score oracle also checks rounding-sensitive choices and eligibility.
- Sphinx warnings-as-errors build passed.

## Reproduction and artifacts

Raw records and before/after source snapshots are in the ignored local directory
`outputs/screening-memory-2026-09-22/`; this contains no checkpoints:

- `nnunet.json`, `mini.json`: final paired performance/accuracy reports.
- `synthetic.json`: synthetic paired comparisons.
- `baseline/rankseg/`, `optimized/rankseg/`: source snapshots, hashed in reports.
- `full-tests.log`, `torch28-tests.log`, `compatibility.json`, `sphinx.log`: validation results.
- `binary-repeatability.json`, `check_repeatability.py`: isolated existing CUDA scan variability.
- `nnunet-assignment-audit.json`, `mini-assignment-audit.json`, `audit_assignment.py`:
  fixed-binary eligibility/float64 incremental-score audits on the same caches.

From `rankseg-nnunet-benchmark`:

```bash
env/bin/python -B scripts/benchmark_screening_update.py \
  --baseline outputs/screening-memory-2026-09-22/baseline \
  --suite nnunet --repeats 11 --output /tmp/screening-memory-nnunet.json

../env/bin/python -B scripts/benchmark_screening_update.py \
  --baseline outputs/screening-memory-2026-09-22/baseline \
  --suite mini --repeats 11 --output /tmp/screening-memory-mini.json
```

Use unused output paths and the corresponding cached-data environments; run GPU
jobs serially. See the [previous optimization report](SCREENING_OPTIMIZATION_2026-09-21.md)
for the preceding implementation comparison and dataset-selection details.
