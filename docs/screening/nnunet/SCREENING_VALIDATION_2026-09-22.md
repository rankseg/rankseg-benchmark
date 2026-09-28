# Fused probability validation — 2026-09-22

This small paired benchmark compares against the preceding screened decoder,
including compact candidate replay and the unique-class assignment shortcut.
It is **not** a full-cohort evaluation or a comparison with unscreened full sort.
No checkpoint, inference, probability-cache or postprocessing changes were made.

## Implementation

Eligible float32/float64 CUDA inputs now check probability values during the
existing full-image screening statistics scan, instead of a separate `aminmax`
scan. Eligibility requires unsmoothed Dice screening, a nonempty batch, no
small-input bypass, available Triton, and a layout viewable as probability rows
without copying. CPU, low-precision and other solver paths retain ordinary value
validation. Structure/dtype validation is still performed once at entry.

- The existing statistics reduction expressions, tile sizes and rounding remain
  unchanged. Per-block integer error codes are additional reductions of the
  already loaded probabilities, including pruned classes and spatial tails.
- Explicit NaN/Inf detection has higher priority than range errors; no reliance
  is placed on floating-point maximum propagating NaN. Out-of-range finite values
  still raise the original error. Parameter-error precedence is preserved.
- The validation host transfer remains, returning row error codes and maxima.
  Candidate construction and sorting cannot start before validation succeeds.
  Full-size binary mask allocation is deferred until candidate construction.
- Globally pruned inputs still skip candidates and sorting. Multiclass fallback
  deliberately retains its original `torch.sum`, because reusing a differently
  rounded fused sum could change near-tied fallback labels.
- Candidate ordering, sorting, `cumsum`, objectives, tie policy, class eligibility
  and output dtype/shape are unchanged. No inference retry or new dispatch
  threshold was added.

The benchmark loader also now isolates the snapshot's validation functions,
when present, so its baseline does not accidentally use the new validator.

## Protocol

- RTX 3090, four CPU threads, serial GPU jobs.
- nnU-Net: PyTorch 2.8.0+cu128 / Triton 3.4.0. One representative each from
  Pancreas, HepaticVessel, Lung and Liver, plus the `liver_43` regression case:
  **5 volumes** total. These are the same held-out caches used previously.
- Mini: PyTorch 2.11.0+cu130 / Triton 3.6.0. **10 examples per dataset**, with
  KiTS using two foreground-only single-channel slices per fold: 40 total.
- Both sides use screening, with three warm-ups and 11 alternating repeats.
  Times are means of per-input median synchronized wall times. Data loading,
  compilation, exact-statistics audits, independent objective checks, metrics
  and official postprocessing are excluded.
- Mini explicitly forces screening. The KiTS numbers therefore do not describe
  its normal small-input full-sort bypass, which remains in place.
- Memory is peak additional PyTorch allocated GPU memory, including output but
  excluding resident probabilities, reserved memory and CUDA context.

## Results

| Dataset | Inputs | Before → after (ms) | Runtime reduction | Extra peak memory before → after (MiB) |
|---|---:|---:|---:|---:|
| nnU-Net Pancreas | 1 volume | 4.502 → 4.204 | 6.6% | 280.00 → 280.00 |
| nnU-Net HepaticVessel | 1 volume | 2.004 → 1.899 | 5.2% | 109.00 → 109.00 |
| nnU-Net Lung | 1 volume | 12.186 → 10.897 | 10.6% | 1,560.00 → 1,560.00 |
| nnU-Net Liver | 2 volumes | 17.404 → 15.965 | 8.3% | 1,257.00 → 1,257.00 |
| VOC | 10 images | 0.686 → 0.653 | 4.8% | 7.503 → 7.503 |
| Cityscapes | 10 images | 2.239 → 2.087 | 6.8% | 56.003 → 56.003 |
| ADE20K | 10 images | 2.133 → 1.955 | 8.4% | 52.191 → 52.191 |
| KiTS, forced screening | 10 slices | 0.323 → 0.312 | 3.4% | 0.15059 → 0.15083 |

All 35 multiclass inputs had lower median runtime. This optimization primarily
saves input bandwidth and validation overhead, not overall peak memory.

### All-pruned tradeoff

This is **not universally faster**. An all-pruned input previously needed only
value validation before its shortcut; the fused path has already calculated
statistics by that point. Synthetic all-pruned cases added roughly **34–49 us**:

| All-pruned input | Output | Before → after (ms) |
|---|---|---:|
| 3 classes × 1,048,576 pixels | multilabel | 0.065 → 0.110 |
| 3 classes × 16,777,216 pixels | multiclass | 3.313 → 3.357 |
| 21 classes × 262,145 pixels | multilabel | 0.076 → 0.124 |
| 150 classes × 131,073 pixels | multiclass | 0.714 → 0.759 |

The small absolute cost can be a large relative slowdown for very short calls.
The forced KiTS subset included one all-pruned slice at **0.050→0.105 ms**
(about 2.1x slower), despite the dataset-average improvement. Small-input production bypasses do not enter
fused validation. A separate 147,456-pixel bypass control measured 0.186→0.188 ms
when active and 0.040→0.043 ms when pruned; bypass dispatch and results are
unchanged, but Python-side setup is not claimed to have zero overhead.

The 18-case synthetic matrix uses 21 alternating repeats and checks both active
and all-pruned cases. Every paired synthetic output matches exactly. Active
screened cases improved by approximately 3–19% in speed ratio. Some all-pruned
multilabel peaks increased by a few KiB of retained statistics; there is no new
full-image workspace. Keep this tradeoff in mind for workloads dominated by
empty multilabel predictions.

## Correctness and existing Liver repeatability

All **45 real inputs** had bitwise-identical full-image sums, maxima, forced
counts/masses and lower bounds versus the preserved implementation.

Independent end-to-end calls matched on **44/45 inputs**. The exception was
`liver_43`, with 2,050 differing final pixels. An additional isolated audit found:

- All **2,553,134 retained candidate indices** matched, in the same order.
- Holding identical floating-point prefix arrays fixed produced **zero final
  differences**. Every sorted scan input was checked bitwise before replay.
- Six repeats of the **old decoder itself** differed from a fixed old output by
  308, 308, 0, 0, 1,828 and 357 pixels. Repeated CUDA scans of a fixed candidate
  array also varied. This reproduces the existing PyTorch 2.8 scan limitation;
  it is not evidence of a changed screening formula.
- Prefix replay is diagnostic only. No override or retry was added to inference.

Dice and IoU were identical on the other 44 inputs. Averaged over the two Liver
volumes, the independent-call differences were +0.001762 Dice percentage points
and +0.002507 IoU percentage points; these are **repeatability effects, not
accuracy gains**. Full-prefix float64 binary objective checks passed for all 45
inputs, with maximum regret 0.4179 float32 eps for nnU-Net and 0.6672 eps for mini.
These objective checks do not relax the bitwise statistics requirement.

## Tests

- Full primary CPU/CUDA suite: **1,731 passed, 14 skipped**, coverage **87.68%**.
- PyTorch 2.8/Triton 3.4 screening suite, large tests enabled: **917 passed**.
- New validation test file: 195 cases, including exact float32/float64 statistics,
  boundary/adjacent values, negative subnormals, NaN/Inf and error priority,
  strided layouts, adaptive tiles, nondefault streams, invalid values in pruned
  rows, public functional/module entrypoints, bypasses and deferred allocation.
- The large test rejects NaN in the last element of a 3-class,
  221,773,824-pixel-per-class tensor before candidate construction.
- **450 exact before/after compatibility checks**: default paths, unsupported
  metric/smooth combinations, CPU screening and binary screening.
- Sphinx warnings-as-errors and whitespace checks pass.

These checks provide empirical evidence on the tested configurations, not a
universal bitwise-repeatability guarantee for floating-point CUDA scans.

## Reproduction

Sources and raw results are in `outputs/screening-validation-2026-09-22/`:
`baseline/`, `optimized/`, `nnunet.json`, `mini.json`, `synthetic.json`,
`liver-audit.json`, `summary.json`, test logs and diagnostic scripts. Files
suffixed `-v1` are development trials, not the final measurements above.
`summarize.py` verifies report/snapshot/current core source hashes.

From `rankseg-nnunet-benchmark`, use fresh output paths:

```bash
env/bin/python -B scripts/benchmark_screening_validation.py \
  --baseline outputs/screening-validation-2026-09-22/baseline \
  --suite nnunet --repeats 11 --output /tmp/validation-nnunet-new.json

../env/bin/python -B scripts/benchmark_screening_validation.py \
  --baseline outputs/screening-validation-2026-09-22/baseline \
  --suite mini --mini-limit 10 --kits-per-fold 2 --repeats 11 \
  --output /tmp/validation-mini-new.json
```
