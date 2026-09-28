# Fused candidate workspace (P2) — 2026-09-22

This compares against the preceding screened decoder, **including P1 fused
validation**, compact candidate replay and unique-class assignment. It does not
compare with full sort or argmax. Checkpoints, inference, probability caches and
official postprocessing are unchanged.

## What changes

The CUDA candidate path previously assembled each group with concatenated
indices, per-candidate `owner`/`position` arrays, padding assignments and separate
statistics gathers. Mask reconstruction then built a selected-by-rank mask and
its inverse permutation before a final scatter.

P2 replaces that data movement with two kernels:

1. Gather candidate probabilities directly into the same bounded sort workspace,
   fill padding with exactly `-1`, and copy the group's mean/count/mass statistics.
   Reads respect probability and statistics strides. No extra full-probability
   `rows.reshape(-1)` copy is required inside CUDA grouping.
2. Use the existing sort permutation to write each real candidate's boolean
   decision directly to its packed index. Padding cannot write, every real
   candidate writes once, and noncandidate/forced/pruned/fallback positions are
   untouched. No atomics or inverse-permutation workspace is needed.

Host metadata contains only row/start/length per group row. Group sizes and
widths, candidate order, padding, workspace budgets and singleton 1-D sort
shapes are unchanged. Group statistics are copied without arithmetic; integer
counts stay int64. Row-count metadata is not a new constexpr specialization for
every group size. Temporary gathered values are released after sorting.

Sorting, floating-point prefix scans, score formulas, direct argmax, multiclass
eligibility and all-pruned/void behavior are unchanged. All pre-existing CUDA
function bodies are AST-identical to the preserved version; only the gather and
scatter helpers are new. CPU/no-Triton paths retain the portable implementation.
This is not a synchronization-free pipeline: candidate counts and per-group
metadata still cross the host/device boundary.

## Protocol

- RTX 3090; four CPU threads; GPU jobs run serially.
- Medical: PyTorch 2.8.0+cu128 / Triton 3.4.0. One representative each from
  nnU-Net Pancreas, HepaticVessel, Lung and Liver, plus `liver_43`: **5 volumes**.
- Mini: PyTorch 2.11.0+cu130 / Triton 3.6.0. Ten cached examples each for VOC,
  Cityscapes and ADE20K, plus two foreground-only KiTS slices per fold: **40
  samples**. KiTS screening is forced; normal small-input bypasses are unchanged.
- Three warm-ups, 11 alternating repeats for real data. Dataset times are means
  of per-input median synchronized wall times, not complete-cohort estimates.
- The 18-case synthetic matrix and 12 dense/ragged whole-decoder cases use 21
  alternating repeats. Twelve additional movement-only cases use fixed sort
  permutations and selected volumes, in float32 and float64.
- Loading, compilation, diagnostic hooks, independent float64 objective checks,
  metrics and official postprocessing are excluded from decoder timing.
- Whole-decoder memory is peak additional PyTorch allocated GPU memory, including
  output but excluding resident probabilities, reserved memory and CUDA context.
  Movement-only memory additionally excludes resident packed indices, binary
  masks, sort permutations and selected volumes. These are different measures.

## Results

### Real cached probabilities

Times are milliseconds per sample. A positive time reduction means faster;
negative values are measured slowdowns. Memory is the mean of per-sample
whole-decoder peak additional allocations, in MiB.

| Dataset | Samples | Before → P2 (ms) | Time reduction | Peak MiB, before → P2 |
| --- | ---: | ---: | ---: | ---: |
| ADE20K | 10 | 1.961 → 1.550 | 21.0% | 52.191 → 51.997 |
| Cityscapes | 10 | 2.082 → 1.699 | 18.4% | 56.003 → 56.003 |
| Pascal VOC | 10 | 0.637 → 0.589 | 7.5% | 7.503 → 7.503 |
| KiTS, forced screening | 10 | 0.301 → 0.311 | −3.4% | 0.151 → 0.153 |
| nnU-Net Pancreas | 1 | 4.209 → 4.107 | 2.4% | 279.253 → 279.253 |
| nnU-Net HepaticVessel | 1 | 1.908 → 1.929 | −1.1% | 109.003 → 109.003 |
| nnU-Net Lung | 1 | 10.924 → 10.937 | −0.1% | 1559.753 → 1559.753 |
| nnU-Net Liver | 2 | 16.002 → 15.975 | 0.2% | 1257.003 → 1257.003 |

All 30 VOC/Cityscapes/ADE20K samples became faster in this run. The improvement
is mainly for multirow candidate groups. Singleton groups have fewer old
temporary operations to remove; their new metadata/statistic allocations can
offset the savings. KiTS was slower on 9/10 samples when screening was forced;
its normal small-input bypass is unchanged. The small medical differences do
not establish a meaningful general speedup or slowdown for those datasets.

The whole-decoder peak is mostly unchanged for these caches: reducing candidate
temporaries does not necessarily reduce a peak set by dense masks/output or
another stage. This should not be advertised as a large real-cohort memory
reduction.

### Candidate-heavy whole-decoder stress cases

These synthetic cases deliberately retain more candidates or unequal candidate
lengths. All six multiclass results are shown below; the corresponding six
multilabel runs are also present in `groups.json`. They are not real-data
accuracy measurements.

| C × D | Candidate distribution | Before → P2 (ms) | Speedup | Peak MiB, before → P2 |
| --- | --- | ---: | ---: | ---: |
| 3 × 1,048,576 | Dense | 2.002 → 1.442 | 1.39× | 83.36 → 59.36 |
| 3 × 1,048,576 | Unequal lengths | 1.081 → 0.914 | 1.18× | 26.28 → 18.78 |
| 21 × 262,145 | Dense | 2.370 → 1.312 | 1.81× | 92.37 → 68.37 |
| 21 × 262,145 | Unequal lengths | 1.372 → 0.966 | 1.42× | 27.87 → 21.12 |
| 150 × 131,073 | Dense | 6.960 → 3.681 | 1.89× | 154.58 → 116.17 |
| 150 × 131,073 | Unequal lengths | 2.618 → 1.812 | 1.44× | 90.35 → 70.84 |

Across all 12 dense/unequal-length cases, whole-decoder speedups are
**1.18–2.01×**, peak reductions **21.6–28.8%**, and masks are identical. The
separate 18-case sparse/all-pruned/bypass matrix also has identical masks:
active non-bypassed multiclass speedups are 1.04–1.35×, multilabel speedups
1.14–1.52×; all-pruned and small-input bypass controls are approximately unchanged.

### Isolated gather/writeback workspace

With sort permutations and selected volumes held fixed, the eight multirow
float32/float64 cases improve by **2.17–4.60×**, and movement-only peak temporary
memory falls by **77.4–88.5%**. For example, float32 with three rows and padded
width 262,145 changes from **0.436 → 0.095 ms** and **26.25 → 4.00 MiB**.

The four singleton cases range from 0.95–1.01×; width-32 memory is unchanged,
while width-262,145 temporary memory falls from about 6.25 to 2.00 MiB. These
stage-only savings exclude sorting/scoring and must not be equated with the
whole-decoder peak above.

## Correctness checks

For each real sample's untimed diagnostic call, every gathered sort input and
group statistic is checked exactly against an independent PyTorch construction.
Every writeback is checked against a full cloned mask updated by the original
inverse-permutation rule. This checks unchanged noncandidate pixels as well as
selected candidates, without an epsilon tolerance for movement.

Full-image statistics are also compared bitwise against the preserved baseline.
Independent full-prefix float64 objective checks remain in the end-to-end
benchmark. They validate the existing numerical solver; they do not relax the
exact movement checks.

The completed real-data audit contains **142 gather/writeback pairs**: 74
singleton and 68 multirow groups, across 44 samples that enter candidate solving.
The remaining KiTS sample has no candidate group. All gathered values, copied
statistics and full-mask writebacks match exactly. Full-image statistics match
the baseline on all 45 samples. The largest independent objective regret is
0.668 float32 eps (below the existing four-eps check).

Independent end-to-end masks match on **44/45 samples**. `liver_43` differs at
2,050 pixels; the two-sample Liver mean Dice/IoU differences are +0.00176/+0.00251
percentage points, respectively. All other dataset metrics are unchanged.
This exception was isolated rather than hidden behind a mask tolerance:

- All 2,553,134 packed candidate indices and sorted scan inputs match exactly.
- Replaying the baseline's exact floating prefixes through P2 gives **zero
  differing pixels**. This override is diagnostic only, not production behavior.
- Six independent repeats of the unchanged old decoder differ from its reference
  mask by 0, 308, 308, 410, 308 and 308 pixels. Its long CUDA `cumsum` outputs also
  vary on repeated identical inputs (up to 0.3125 absolute prefix-sum difference
  in this audit).

Thus the audit identifies the existing long CUDA prefix scan as the source of
the discrepancy; the new gather/writeback operations are exact. This does not
promise bitwise repeatability for independent long-scan decoder calls.

## Regression tests

- Full primary CPU/CUDA suite: **1,922 passed, 16 skipped**, coverage **86.75%**.
- PyTorch 2.8/Triton 3.4 screening suite, large tests enabled: **1,110 passed**.
- New test file: **193 cases**, covering float32/float64, contiguous/channel-
  strided/spatial-strided/expanded/offset inputs, strided statistics, empty rows,
  odd tails, reordered groups, singleton/128-row groups, and nondefault streams.
- Exhaustive writeback checks cover all 24 four-element permutations and five
  selection volumes, for candidate lengths 0, 1, 2 and 4: **480 combinations**.
- Counts above `2**33` are copied exactly. Large-memory tests cover
  221,773,824-pixel rows and logical packed indices above `2**31`, not merely
  byte addresses. The latter uses 129 rows × 16,777,216 pixels.
- Allocation-path guards reject old owner/inverse-permutation operations and
  the extra full probability flatten/copy on the CUDA path.
- **450** additional before/after compatibility checks match exactly: 216
  default-path, 180 other-metric/smoothing, 18 CPU screening and 36 binary
  screening cases.
- Sphinx warnings-as-errors and whitespace checks pass.

These checks are not a claim of bitwise repeatability for every floating-point
CUDA scan. Long-prefix behavior is audited separately when comparing independent
end-to-end calls.

## Reproduction

Raw reports, diagnostic scripts and source snapshots are under
`outputs/screening-candidate-workspace-2026-09-22/`. `baseline/` and `optimized/`
preserve the two core versions. `summarize.py` checks source hashes and unchanged
score/grouping function ASTs before aggregating results. Files suffixed `-v1`
are development trials, not the final results.

From `rankseg-nnunet-benchmark`, use fresh output paths:

```bash
env/bin/python -B scripts/benchmark_candidate_workspace.py \
  --baseline outputs/screening-candidate-workspace-2026-09-22/baseline \
  --suite nnunet --repeats 11 --output /tmp/workspace-nnunet-new.json

../env/bin/python -B scripts/benchmark_candidate_workspace.py \
  --baseline outputs/screening-candidate-workspace-2026-09-22/baseline \
  --suite mini --mini-limit 10 --kits-per-fold 2 --repeats 11 \
  --output /tmp/workspace-mini-new.json
```
