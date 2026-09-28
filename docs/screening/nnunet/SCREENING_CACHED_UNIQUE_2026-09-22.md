# Cached unique-class identities — 2026-09-22

This compares against the preceding screened decoder, including fused validation
and the P2 candidate workspace optimization. It is **not** a comparison against
argmax or unscreened full sort. Checkpoints, model inference, probability caches,
postprocessing and evaluation case selection are unchanged.

## Change and exactness contract

The existing unique-statistics pass already reads every class's binary mask.
Previously it stored only zero/one/multiple selections per pixel; the final
assignment pass reread the masks to recover the unique class identity.

For up to 254 classes, that same one-byte buffer now stores:

| Code | Meaning |
| --- | --- |
| 0 | Unassigned pixel |
| 1 | Overlapping pixel |
| `2 + class_index` | Uniquely selected class |

One int32 reduction carries both count and identity: each selected class `c`
contributes `((c + 2) << 8) + 1`. With at most 254 selected classes, the low eight
bits contain the exact count without carry; even the largest possible summed
tag is below `2**24`, within int32. Only when the count equals one are the high
bits interpreted as an identity. Overlap sums are never used as class IDs.

All-unique assignment tiles decode the stored identity directly, with no second
class-by-pixel mask read or class reduction. Mixed tiles still execute the old
scoring function. The predicates it uses, selected versus unassigned, are
unchanged by the new encoding. At 255/256 classes, the old encoding and
mask-based identity lookup are retained; larger unsupported shapes still use
the existing portable fallback. Private helpers retain their original default
encoding for callers that do not request cached identities.

No public API, shape contract, screening bound, candidate order, sort, prefix
scan, score expression, tie rule or pruning/void behavior changes. Unique counts
remain exact integers. The unique-probability sum expression, reduction tile
sizes and hierarchical floating reductions are unchanged. There is no new
full-image buffer and no new host synchronization.

## Paired real-cache benchmark

- RTX 3090, four CPU threads; final GPU jobs run serially.
- Medical: Torch 2.8.0+cu128 / Triton 3.4.0, one held-out nnU-Net case per fold
  for each of four datasets, plus `liver_43`: **21 volumes**. These are subsets,
  not complete-cohort averages.
- Mini: Torch 2.11.0+cu130 / Triton 3.6.0, 100 cached examples per dataset:
  **400 samples**. KiTS is foreground-only multilabel, with screening forced,
  so it is an unchanged control for this multiclass-only optimization.
- Three warm-ups and 11 alternating timed repeats. Times below are means of
  per-input median synchronized decoder wall times, in ms/sample.
- Loading, compilation, exact diagnostic checks, metrics, independent objective
  checks and official postprocessing are excluded from decoder timings.
- Memory is peak additional PyTorch allocated GPU memory, including decoder
  output but excluding resident probabilities, CUDA context and reserved memory.

| Dataset | Samples | Before → cached IDs (ms) | Time reduction | Peak MiB, before → after |
| --- | ---: | ---: | ---: | ---: |
| nnU-Net Liver | 6 | 13.853 → 8.525 | 38.5% | 1111.34 → 1111.34 |
| nnU-Net Pancreas | 5 | 4.077 → 2.591 | 36.5% | 306.90 → 306.90 |
| nnU-Net HepaticVessel | 5 | 2.285 → 1.596 | 30.2% | 141.10 → 141.10 |
| nnU-Net Lung | 5 | 7.767 → 7.357 | 5.3% | 1113.70 → 1113.70 |
| Cityscapes | 100 | 1.810 → 1.635 | 9.7% | 56.00 → 56.00 |
| ADE20K | 100 | 1.476 → 1.417 | 3.9% | 51.21 → 51.21 |
| Pascal VOC | 100 | 0.595 → 0.574 | 3.5% | 7.50 → 7.50 |
| KiTS, unchanged binary control | 100 | 0.297 → 0.297 | approximately 0% | 0.150 → 0.150 |

All **321 multiclass samples** became faster in this run. KiTS varied in both
directions, with an aggregate difference of 0.01%; its modified-kernel hit count
is zero. Peak allocated memory is identical before/after for every real sample.

### Is the work merely moved to statistics?

Separate CUDA-event timings on frozen masks/statistics check this explicitly.
Medical dataset-mean stage times are shown below; do not add these independently
measured medians to reconstruct wall time.

| Dataset | Statistics before → after (ms) | Assignment before → after (ms) | Combined statistics + assignment before → after (ms) |
| --- | ---: | ---: | ---: |
| Liver | 3.462 → 3.371 | 6.501 → 1.278 | 9.899 → 4.576 |
| Pancreas | 0.998 → 0.972 | 1.820 → 0.322 | 2.794 → 1.267 |
| HepaticVessel | 0.467 → 0.455 | 0.840 → 0.167 | 1.283 → 0.596 |
| Lung | 3.695 → 3.570 | 1.480 → 1.184 | 5.125 → 4.709 |

The saved assignment work is not offset by a statistics slowdown in these
medical measurements. The optimization reduces repeated mask reads and class
reductions, not the size of the dense output. It should be presented as a
runtime improvement, not a further whole-decoder memory reduction.

## Correctness and adverse cases

All **421 real samples** have identical final masks and unchanged Dice/IoU in
this run. Each of the 321 multiclass samples additionally receives an untimed
same-input audit:

- Old and new pixel classifications, per-class unique counts and probability
  sums are compared exactly, without a floating tolerance.
- Assignment is compared on the same binary masks/statistics for `max_score`
  and `void`, including both int64 void-index extremes.
- The complete statistics-plus-assignment entrypoints are also compared.

Independent full-prefix float64 objective checks still pass; maximum regret
over all real cases is 0.923 float32 eps, below the existing four-eps limit.
Those checks concern the unchanged numerical binary solver; they do not relax
the exact statistics/assignment checks above. Existing long CUDA prefix scans
can still vary between independent decoder calls, so this observed 421/421
agreement is not a universal bitwise-repeatability guarantee.

Synthetic stress tests include all-unique, clustered 99%-unique, alternating
50%-unique, all-overlapping and all-unassigned masks. A 30-case float32 matrix
and two 65-case extended matrices (float32/float64, including 254/255/256 classes)
use 31 alternating repeats. All **160 cases** preserve statistics and labels
exactly, with unchanged measured statistics-plus-assignment peak allocations.

There is **no universal speedup**: tiles without unique-only regions cannot
benefit from the cached shortcut. The worst measured combined-stage slowdowns
are **5.9% for float32** (64 classes, all unassigned) and **4.7% for float64**
(64 classes, alternating unique/overlap). These are isolated-stage results, not
whole-decoder regressions; no such slowdown occurred in the 321 real multiclass
samples above. Retaining the existing mixed-tile score function does not imply
identical compiler scheduling or zero encoding overhead.

## Regression tests

- Full primary CPU/CUDA suite: **2,127 passed, 16 skipped**, coverage **86.42%**.
- Torch 2.8/Triton 3.4 screening suite with large-memory tests enabled:
  **1,315 passed**. The public cached assignment path is exercised on
  221,773,824-pixel volumes; existing greater-than-int32 logical indexing tests
  also pass.
- New cached-identity test file: **205 cases**. Coverage includes float32/float64,
  1–256 classes with explicit 254/255/256 boundaries, every class identity,
  all-selected count overflow guards, tile tails, strided/expanded inputs,
  hierarchical reductions, counts above float32's exact-integer range,
  nondefault streams, pruning, void-index extremes, exact/adjacent score ties,
  and all 512 three-class/three-pixel binary masks.
- **450** exact compatibility comparisons: 216 default-path, 180 other metric/
  smoothing, 18 CPU-screened and 36 binary-screened cases.
- Sphinx warnings-as-errors and whitespace/source-scope checks pass.

## Reproduction

The source baseline/optimized snapshots, raw timing samples, diagnostic scripts and reports are in
`outputs/screening-cached-unique-2026-09-22/`. `summarize.py` verifies source hashes
and modification scope. Only `_screening_cuda.py` changes relative to the
baseline; the original scoring functions and floating hierarchical reduction
are AST-identical. The `*-interrupted` run was stopped to avoid possible GPU
contention and is excluded from all reported results.

From `rankseg-nnunet-benchmark`, using fresh output paths:

```bash
env/bin/python -B scripts/benchmark_cached_unique.py \
  --baseline outputs/screening-cached-unique-2026-09-22/baseline \
  --suite nnunet --repeats 11 --output /tmp/cached-unique-nnunet-new.json

../env/bin/python -B scripts/benchmark_cached_unique.py \
  --baseline outputs/screening-cached-unique-2026-09-22/baseline \
  --suite mini --mini-limit 100 --kits-per-fold 20 --repeats 11 \
  --output /tmp/cached-unique-mini-new.json
```
