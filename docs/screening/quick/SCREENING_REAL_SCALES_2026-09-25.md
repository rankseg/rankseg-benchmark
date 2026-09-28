# CPU screening on uniformly subsampled real probabilities — 2026-09-25

**Status: complete. Benchmark tooling/report only; no RankSEG implementation,
default or `auto` policy changed.**

Subsequent work: [candidate-retry removal and CUDA real-scale measurements](SCREENING_CUDA_REAL_SCALES_2026-09-25.md).
The CPU results below retain their original source hashes; the follow-up changes
candidate handling and separately measures CUDA rather than extrapolating from CPU.

This follow-up to the [CPU crossover calibration](SCREENING_CPU_CALIBRATION_2026-09-24.md)
uses **only real cached model probabilities**, with 40 source inputs and 12
spatial sizes. No synthetic dense-candidate distributions are included.

On this CPU, **every tested configuration at D≥4,096 is a stable win**.
Real small-input regressions remain: a high screening proportion alone does not
guarantee lower runtime. These measurements support 4,096 as a candidate CPU
boundary on the measured workloads, not a universal crossover.

## Protocol and source selection

- Intel i9-12900K; Python 3.10.12; PyTorch 2.11.0+cu130. All tensors/execution on CPU; no CUDA or Triton execution.
- RMA/Dice, float32, B=1, smooth=0, pruning_prob=0.5, unassigned_policy=max_score. Compare public `RankSEG.predict` with `safe_screening=False` versus `True`.
- VOC (C=21), Cityscapes (C=19), ADE20K (C=150): **10 images each**. Divide each pinned first-100 cache into 10 equal strata and select the center row: **5, 15, 25, 35, 45, 55, 65, 75, 85, 95**, zero-based.
- KiTS: **two slices per fold, five folds**, using the quarter and three-quarter cache-row positions. Keep only foreground channel 1 (C=1, multilabel). The resulting 10 slices cover **nine distinct case IDs**, not 10 independent cases. This is slice-stratified, not case-balanced selection.
- Cache revision: `1884f0766268cdc62730e696f24dcc913d551b35`. All 40 source profile IDs differ from the eight real profiles in the previous calibration. Selection uses neither labels nor timing outcomes.
- For each source, generate a seeded random permutation of spatial positions. Take the first D positions **uniformly without replacement**, sort their indices into original spatial order, and gather the same positions from every channel. Seed: **20260925**, with a deterministic per-profile SHA-256-derived seed.
- Subsets are **nested across D** and identical across thread settings. Preserve probabilities exactly: no interpolation, renormalization or duplicate pixels. This scales the number of positions, not model inference resolution or a contiguous crop. All requested sizes fit all source inputs.
- Main run: **4 threads**, D=64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384, 32,768, 65,536, 131,072.
- Follow-up: **1 and 8 threads**, D=512, 1,024, 2,048, 4,096, 8,192, 16,384, using the same input hashes. No parallel benchmark jobs.
- Three rounds; rotate thread blocks and shuffle workloads. Each block: two warm-ups and seven timed predictions per method, alternating method order. Measure only postprocessing on resident tensors; exclude extraction, loading, hashing, counting and correctness checks.
- Per input/configuration: median of three per-round timing medians. Dataset speedup is **mean off time / mean on time**, not the mean of individual speedup ratios.
- A **stable win** requires off/on >1.05 in every round; a **stable loss** requires <0.95 in every round. Others are mixed/near. This reporting margin is not a statistical confidence interval.

Total: **960 configurations**, **2,880 timing blocks**, **40,320 timed predictions**.

KiTS source rows, for exact reproduction:

| Fold | First row / case | Second row / case |
| ---: | --- | --- |
| 0 | 238 / 5 | 714 / 5 |
| 1 | 313 / 78 | 941 / 52 |
| 2 | 354 / 96 | 1,063 / 118 |
| 3 | 367 / 128 | 1,103 / 135 |
| 4 | 154 / 203 | 462 / 199 |

## Speedup by scale

Four threads. **Exclude inputs where all classes are pruned** in this table so
that their early-exit gains do not inflate the pixel-screening comparison.
All 10 natural-image inputs remain eligible at every size; the last column
states how many KiTS inputs remain. D is positions **per class**, not B×C×D.

| D | VOC | Cityscapes | ADE20K | KiTS foreground | KiTS active / 10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.009× | 0.921× | 1.596× | 0.985× | 1 |
| 128 | 0.964× | 0.898× | 1.657× | 0.862× | 4 |
| 256 | 0.991× | 0.957× | 1.567× | 0.913× | 4 |
| 512 | 1.159× | 1.035× | 1.695× | 0.955× | 8 |
| 1,024 | 1.429× | 1.218× | 1.886× | 1.028× | 8 |
| 2,048 | 1.397× | 1.196× | 2.346× | 1.186× | 9 |
| 4,096 | 1.426× | 1.271× | 2.992× | 1.623× | 9 |
| 8,192 | 1.595× | 1.340× | 5.684× | 2.360× | 9 |
| 16,384 | 1.849× | 1.564× | 4.527× | 3.699× | 9 |
| 32,768 | 2.229× | 1.823× | 2.873× | 5.104× | 9 |
| 65,536 | 3.044× | 2.743× | 2.586× | 8.574× | 9 |
| 131,072 | 2.849× | 2.539× | 2.555× | 15.604× | 9 |

These are actual public-path gains, not isolated sorting-kernel gains: partial
class pruning and final multiclass assignment still contribute to the times.

### Absolute times, including all 10 inputs per dataset

Four threads; mean per-input postprocessing milliseconds, **off → on**.
Unlike the preceding table, this includes entirely class-pruned KiTS inputs.
For example, at D=64 this gives KiTS 2.476× overall, but only 0.985× on its one
active input. At D=4,096, the corresponding figures are 1.745× and 1.623×.

| D | VOC ms | Cityscapes ms | ADE20K ms | KiTS ms |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.2284 → 0.2264 | 0.2655 → 0.2884 | 0.5773 → 0.3618 | 0.0708 → 0.0286 |
| 128 | 0.2666 → 0.2766 | 0.3048 → 0.3393 | 0.7441 → 0.4491 | 0.0787 → 0.0550 |
| 256 | 0.3318 → 0.3349 | 0.3821 → 0.3993 | 0.8477 → 0.5411 | 0.0827 → 0.0551 |
| 512 | 0.4698 → 0.4052 | 0.5191 → 0.5015 | 1.0644 → 0.6279 | 0.0986 → 0.0909 |
| 1,024 | 0.7436 → 0.5203 | 0.7819 → 0.6419 | 1.4311 → 0.7588 | 0.1175 → 0.1000 |
| 2,048 | 0.8620 → 0.6168 | 0.9504 → 0.7943 | 2.4438 → 1.0418 | 0.1653 → 0.1297 |
| 4,096 | 1.1733 → 0.8230 | 1.3225 → 1.0409 | 5.4311 → 1.8150 | 0.2515 → 0.1441 |
| 8,192 | 1.9112 → 1.1982 | 1.9456 → 1.4516 | 16.3537 → 2.8769 | 0.4175 → 0.1627 |
| 16,384 | 3.6171 → 1.9559 | 3.5945 → 2.2985 | 35.1935 → 7.7745 | 0.7381 → 0.1839 |
| 32,768 | 8.2477 → 3.7005 | 7.9550 → 4.3632 | 72.9393 → 25.3918 | 1.3662 → 0.2450 |
| 65,536 | 21.2025 → 6.9662 | 21.1668 → 7.7158 | 173.1561 → 66.9655 | 2.5032 → 0.2659 |
| 131,072 | 47.7997 → 16.7759 | 44.9997 → 17.7211 | 346.5606 → 135.6504 | 4.6483 → 0.2726 |

### Per-input consistency and thread verification

Do not infer uniform improvement from dataset averages alone:

| D | Configurations | Stable win | Mixed / near | Stable loss |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 40 | 29 | 2 | 9 |
| 128 | 40 | 20 | 8 | 12 |
| 256 | 40 | 18 | 13 | 9 |
| 512 | 120 | 93 | 21 | 6 |
| 1,024 | 120 | 111 | 0 | 9 |
| 2,048 | 120 | 104 | 16 | 0 |
| 4,096 | 120 | 120 | 0 | 0 |
| 8,192 | 120 | 120 | 0 | 0 |
| 16,384 | 120 | 120 | 0 | 0 |
| 32,768 | 40 | 40 | 0 | 0 |
| 65,536 | 40 | 40 | 0 | 0 |
| 131,072 | 40 | 40 | 0 | 0 |

The 120-configuration rows cover 40 inputs × 1/4/8 threads; the other rows
cover four threads only. Overall: **855 wins, 60 mixed/near, 45 losses**.
At D=2,048 all configurations have faster median times, but the weakest round
is only 1.0007×. At D=4,096 the weakest round is 1.1254×.

Speedup at D=4,096, excluding entirely class-pruned inputs as above:

| Source | 1 thread | 4 threads | 8 threads |
| --- | ---: | ---: | ---: |
| VOC | 2.242× | 1.426× | 1.428× |
| Cityscapes | 1.793× | 1.271× | 1.269× |
| ADE20K | 3.829× | 2.992× | 2.550× |
| KiTS foreground | 1.620× | 1.623× | 1.614× |

The small-input losses are real, not synthesized adversarial cases. For
example, KiTS fold 2 row 1,063, D=128, four threads: **0.094392 → 0.145376 ms**,
0.649×, an extra **50.984 µs**. The same source at D=256 is 0.677×. Neither
uses a candidate-budget fallback. These observations are consistent with
screening overhead outweighing the small amount of sorting work; this run
does not separately profile that overhead.

## How much was screened?

H and L are positions resolved positive/negative for the per-class binary
optimization; M is the unresolved candidate set. Report **(H+L)/(H+L+M)**,
excluding whole-class-pruned entries from the denominator. These are
class-position counts, not unique multiclass pixels, final multiclass
eligibility, or memory-saving percentages.

| D | VOC screened | Cityscapes screened | ADE20K screened | KiTS screened |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 99.812% | 99.683% | 99.135% | 100.000% |
| 1,024 | 99.623% | 99.848% | 98.748% | 99.951% |
| 4,096 | 99.518% | 99.843% | 98.859% | 99.927% |
| 16,384 | 99.602% | 99.842% | 98.836% | 99.912% |
| 65,536 | 99.563% | 99.832% | 98.811% | 99.914% |
| 131,072 | 99.573% | 99.831% | 98.830% | 99.918% |

Detailed counts at D=4,096, summing each of the 10 inputs once:

| Source | Active entries | H | L | M | Whole-class-pruned / all entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| VOC | 122,880 | 40,185 | 82,103 | 592 | 85.714% |
| Cityscapes | 520,192 | 40,211 | 479,163 | 818 | 33.158% |
| ADE20K | 327,680 | 38,716 | 285,226 | 3,738 | 94.667% |
| KiTS | 36,864 | 650 | 36,187 | 27 | 10.000% |

**Zero candidate-budget fallback rows** across all 960 configurations.
Screening proportions remain high at small D, where runtime may still regress.
No new RSS or GPU-memory measurements were performed in this experiment.

## Correctness and audit

- **1,920/1,920 independent float64 binary-objective checks passed**, checking both paths for every configuration against exhaustive per-class volume optima. Unchanged acceptance budget: four float32 eps. Maximum observed regret: **0.887693 eps**.
- Per-method outputs are bitwise repeatable across three rounds at fixed thread count. This is not on/off bitwise equivalence: **44 configurations differ**, with at most **0.022126%** differing output elements in any configuration. The objective checks concern the per-class binary optimization, not ground-truth Dice/IoU or a proof about final multiclass accuracy.
- All requested configurations and rounds are present exactly once. Timings are positive and finite; medians, ratios and objective status are recomputed by an audit script. Active entries partition exactly into H, L and M, separately from whole-class pruning.
- Identical input hashes across rounds/thread settings; matching source-cache and sampling-order identities across runs. RankSEG source hashes unchanged before/after benchmarking and at final aggregation.
- Full mini-benchmark test suite: **198 passed, 16 skipped**, including sampling, nested-subset, source-selection, all-pruned aggregation, report-integrity and failure-propagation tests.

These are regression checks on measured inputs, not a universal correctness
proof. Spatial subsampling does not constitute a new full-cohort Dice/IoU
evaluation. No new inference, checkpoints, downloads or core changes were needed.

## Implications for CPU `auto`

1. Use real-workload evidence as the primary practical guide. Here, 65,536 would exclude many consistently useful cases.
2. **4,096 is the smallest tested common boundary above which every measured size/input/thread configuration is a stable win.** The 1/8-thread validation covers through 16,384; larger sizes were measured at four threads only. This is one CPU host and four cache sources, not a hardware-independent guarantee.
3. At smaller sizes, some datasets already benefit, especially ADE20K, but single-channel KiTS and some natural-image inputs still regress. The all-pruned early exit should be considered separately from pixel screening.
4. Keep an explicit user override. These CPU measurements do not determine CUDA/Triton or CUDA/PyTorch dispatch boundaries.

No automatic threshold or default has been implemented by this experiment.

## Reproduction and local artifacts

From `rankseg-benchmark`, using the existing pinned caches and sibling source
checkouts. Output directories must be new; the scripts refuse overwrites.

```bash
../env/bin/python scripts/calibrate_screening_cpu.py \
  --real-only --samples-per-dataset 10 --sampling uniform --sampling-seed 20260925 \
  --threads 4 \
  --dims 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 \
  --rounds 3 --repeats 7 \
  --output-dir artifacts/screening-real-scales-replay/primary

../env/bin/python scripts/calibrate_screening_cpu.py \
  --real-only --samples-per-dataset 10 --sampling uniform --sampling-seed 20260925 \
  --threads 1 8 --dims 512 1024 2048 4096 8192 16384 \
  --rounds 3 --repeats 7 \
  --output-dir artifacts/screening-real-scales-replay/thread-check

../env/bin/python scripts/summarize_real_screening_scales.py \
  artifacts/screening-real-scales-replay/primary \
  artifacts/screening-real-scales-replay/thread-check \
  --output artifacts/screening-real-scales-replay/summary.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../env/bin/python -m pytest -q
```

Original raw artifacts are local and ignored, under
`artifacts/screening-real-scales-2026-09-25/`: `primary/results.json`,
`primary/timings.jsonl`, `thread-check/results.json`,
`thread-check/timings.jsonl`, and `summary.json`. They retain per-input/per-round
timings, counts, objective diagnostics and provenance hashes.
