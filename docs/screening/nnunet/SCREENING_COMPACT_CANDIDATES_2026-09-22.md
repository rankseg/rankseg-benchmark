# Compact screening candidates — 2026-09-22

This compares compact candidate replay with the **previous screened decoder**,
including the completed unique-class assignment shortcut. It is not a comparison
with full sort or argmax. The 21 nnU-Net volumes and 400 mini-benchmark samples
are engineering subsets, not a rerun of complete cohorts. Models, checkpoints,
probability caches, pruning, screening bounds and official postprocessing are
unchanged.

## Change and exactness contract

Previously, CUDA candidate construction stored both the binary mask H and a
dense boolean candidate mask M. Packing subsequently read M to produce indices.
The decoder now keeps H, the already computed lower bounds, and integer block
candidate counts. It does not allocate M.

During stable index packing, only nonempty blocks of retained rows reread the
original probabilities. They replay the same comparisons against the stored
lower bound and forced-positive threshold, in the same working dtype. Pruned,
empty and workspace-fallback rows are skipped. Integer prefix offsets and
row-major candidate order are unchanged. The compact descriptor retains the
exact tile size used during counting, including adaptive tiles for large inputs.

This does **not** recompute floating-point statistics or change sorting, prefix
sums, scoring, class assignment, epsilon rules, fallback budgets or dispatch
thresholds. There is no new kernel launch or host synchronization. Original
probabilities remain unchanged and remain available for final multiclass
assignment. The old dense form remains an internal diagnostic option.

The acceptance criterion is **exact candidate membership and order**, not
approximately equal objectives. The unchanged solver's independent objective
checks are additional validation, not a relaxed tolerance for this change.

## Protocol

- RTX 3090; GPU workloads run serially, with four CPU threads.
- nnU-Net: PyTorch 2.8.0+cu128 / Triton 3.4.0. First held-out case per fold
  for Pancreas, HepaticVessel, Lung and Liver, plus `liver_43`.
- Mini: PyTorch 2.11.0+cu130 / Triton 3.6.0. First 100 cached examples each
  for VOC, Cityscapes and ADE20K; KiTS uses 20 foreground-only single-channel
  slices per fold, 100 in total.
- Both sides use screening. Mini runs force screening to measure the changed
  path; normal production small-input bypasses are unchanged. In particular,
  the KiTS savings below do not apply when its normal bypass selects full sort.
- Three warm-ups and 11 alternating timed repeats. Dataset latency is the
  mean of per-input median synchronized wall times. Loading, compilation,
  candidate audits, float64 objective oracles, metrics and postprocessing are
  excluded from timing.
- Memory is **peak additional PyTorch allocated GPU memory**, including output
  but excluding resident input probabilities, reserved memory and CUDA context.
  Dataset memory values are means of per-input peaks, not maximum-case peaks.

## Whole-decoder results

| Dataset | Samples | Before → after (ms) | Runtime reduction | Peak memory before → after (MiB) | Memory reduction |
|---|---:|---:|---:|---:|---:|
| nnU-Net Pancreas | 5 volumes | 4.708 → 4.616 | 2.0% | 306.7 → 306.7 | 0% |
| nnU-Net HepaticVessel | 5 volumes | 2.441 → 2.405 | 1.5% | 141.1 → 141.1 | 0% |
| nnU-Net Lung | 5 volumes | 8.996 → 8.735 | 2.9% | 1,113.3 → 1,113.3 | 0% |
| nnU-Net Liver | 5 + 1 volumes | 15.591 → 15.236 | 2.3% | 1,111.1 → 1,111.1 | 0% |
| VOC | 100 images | 0.759 → 0.757 | 0.2% | 10.57 → 7.50 | 29.0% |
| Cityscapes | 100 images | 2.530 → 2.483 | 1.9% | 76.67 → 56.03 | 26.9% |
| ADE20K | 100 images | 2.144 → 2.082 | 2.9% | 90.92 → 51.37 | 43.5% |
| KiTS, forced binary screening | 100 slices | 0.311 → 0.310 | 0.4% | 0.269 → 0.149 | 44.9% |

Every paired input had an equal or lower measured memory peak. All 221
nnU-Net/Cityscapes/ADE20K inputs had lower median latency. VOC and KiTS were
effectively unchanged in aggregate: 46/100 VOC and 37/100 KiTS inputs had
slightly higher medians, with maximum increases of 3.34% and 2.27%, respectively.
Sub-percent aggregate differences should not be interpreted as reliable speedups.

Removing a full `C × D` byte mask does not necessarily reduce the overall peak.
For these low-class-count medical cases, final binary masks, status and int64
labels already dominate that peak. High-class-count cases previously peaked
earlier while both H and M were live, so removing M lowers their overall peak.
Dense candidate sorting can also dominate memory independently of M.

## Correctness

- **407 exact candidate audits**: same H masks, forced counts/masses, block
  counts and packed candidate indices in exactly the same order. The other
  14 inputs were globally pruned KiTS slices and bypassed candidate creation.
  No audited real row exceeded the candidate workspace budget.
- **421/421 final predictions matched bitwise** in this run; Dice and IoU were
  unchanged on every paired example, including `liver_43`.
- Full-prefix float64 binary objective checks passed for all 421 samples.
  Maximum regret was 0.4811 float32 eps for nnU-Net and 0.9229 eps for mini.
  These checks concern the existing binary solver, not candidate replay.
- Full primary CPU/CUDA suite: **1,537 passed, 13 skipped**, coverage **87.84%**.
- PyTorch 2.8/Triton 3.4 screening suite, with large tests enabled:
  **722 passed**, including 221,773,824-pixel rows and candidates in the last
  block of a multi-gigabyte tensor.
- New compact-candidate test file: **114 tests**, covering float32/float64,
  exact thresholds and adjacent floats, odd tails, channel/spatial strides,
  prepared/unprepared statistics, pruning, fallback rows, empty packing,
  adaptive tiles, nondefault streams, input immutability and peak memory.
- **450 exact before/after compatibility checks**: 216 default-path, 180
  unsupported metric/smooth, 18 CPU screening and 36 binary-screening cases.
- Sphinx HTML builds with warnings treated as errors; whitespace checks pass.

The earlier [unique-class report](SCREENING_UNIQUE_FASTPATH_2026-09-22.md)
documents existing CUDA prefix-scan variability on `liver_43`. Exact final
agreement in this run does not remove that repeatability limitation. This
change introduces neither a deterministic-scan override nor an inference retry.

## Adversarial runtime checks

A 25-case candidate-stage matrix covers 1–150 rows, 147,456–16,777,216 pixels,
no candidates, clustered candidates, scattered candidates, dense candidates
and workspace fallback. All old/new packed indices match exactly. Stage timing
includes candidate construction, host count transfer and packing, but excludes
precomputed statistics and the resident binary mask.

The worst candidate-stage case, 150 rows with dense candidates, was **13.1%
slower**. Replaying comparisons rereads floating-point probabilities instead of
one-byte M values, so this is not a universally faster packing kernel. Conversely,
large sparse/empty/fallback cases reached roughly 1.17–1.19x stage speedups.
Their large stage-memory savings must not be confused with overall peak savings.

To check the actual impact, 12 additional **whole-decoder** cases used scattered
and dense candidates at 3, 21 and 150 classes, in float32 and float64, with 31
alternating repeats and independent objective checks. All final masks matched.
The largest whole-decoder slowdown was **0.8% for float32** and **2.2% for
float64**; the dense 150-class cases were 0.55% and 0.79% slower, respectively.
Two additional 3-class, 16,777,216-pixel sparse cases improved by about 1–2%.
No data-dependent tuning rule or dense-path dispatch was added from these trials.

These are empirical checks on one GPU, not a guarantee for every input or device.
The practical benefit here is substantially lower high-class-count memory with
small or favorable observed end-to-end runtime changes.

## Reproduction and artifacts

Raw records and preserved sources are under
`outputs/screening-compact-candidates-2026-09-22/`:

- `baseline/`, `optimized/`: source snapshots; the only core source differences
  are `_screening.py` and `_screening_cuda.py`.
- `nnunet.json`, `mini.json`, `summary.json`: per-input timing samples, peaks,
  probability/label provenance, metrics, candidate audits and source hashes.
- `stress.json`, `stress-pipeline.json` and their generation scripts.
- `full-tests.log`, `torch28-tests.log`, `compatibility.json`, `sphinx.log`.
- `summarize.py`: verifies snapshot/report/current source identity and aggregates
  results without changing the original measurements.

From `rankseg-nnunet-benchmark`, use fresh output paths:

```bash
env/bin/python -B scripts/benchmark_compact_candidates.py \
  --baseline outputs/screening-compact-candidates-2026-09-22/baseline \
  --suite nnunet --repeats 11 --output /tmp/compact-nnunet-new.json

../env/bin/python -B scripts/benchmark_compact_candidates.py \
  --baseline outputs/screening-compact-candidates-2026-09-22/baseline \
  --suite mini --repeats 11 --output /tmp/compact-mini-new.json
```
