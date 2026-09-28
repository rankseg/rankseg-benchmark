# Candidate-retry removal and real-image CUDA scale measurements — 2026-09-25

**Status: complete. Candidate-count retries removed; CUDA size dispatch and
the public `safe_screening=False` default unchanged.**

Unlike the [CPU real-scale results](SCREENING_REAL_SCALES_2026-09-25.md), these
GPU measurements show substantial small-input slowdowns on real probabilities.
Large natural-image inputs benefit strongly, but the foreground-only KiTS
slices remain faster with optimized full sort. Removing every CUDA size gate
is therefore not supported by these data.

## Implementation change

- Removed both per-row candidate retry conditions: M > 0.75D and M > 4,194,304.
- Once screening starts, **all unresolved candidates are solved**, regardless of their count/fraction. The caller no longer receives fallback flags or reruns full sort for those rows.
- Retained grouping controls for padded multi-row workspaces: at most 1,048,576 padded elements, a 2× length ratio, and 128 rows per group. A longer real row runs alone without padding. These limits do not cap a singleton row or guarantee a total-memory ceiling.
- No changes to screening certificates, score formulas, class pruning, direct argmax, or multiclass eligibility/assignment. The private helper now returns only masks; associated benchmark diagnostics were updated. Historical source snapshots with the old constants remain supported by the count collector.
- Allocation failures are not silently retried with full sort. Unsupported objectives, explicit screening-off, and the **existing pre-screening CUDA size dispatch** still use their usual paths.
- Removing the absolute candidate cap also required hardening the CUDA scorer: candidate blocks use grid.x rather than grid.y, and very long rows reduce block maxima in bounded stages with int64 original indices. Every candidate volume is still evaluated; hierarchical max/first-index reduction preserves exact ties. Ordinary candidate sizes do not gain extra kernel launches. This is execution tiling, not a new screening cutoff.

This removes a post-screening retry, not the separate pre-screening size gate
being evaluated below. No `auto` mode or new threshold was implemented.

## Real inputs and sampling

Same 40 cache sources, selection and seeded nested spatial subsets as the CPU
experiment: 10 VOC images, 10 Cityscapes images, 10 ADE20K images, and 10 KiTS
slices (two per fold, nine distinct cases). See the linked CPU report for exact
source rows and cache provenance. Seed **20260925**; cache revision
`1884f0766268cdc62730e696f24dcc913d551b35`.

Select spatial positions uniformly without replacement, using the same positions
for every channel. Keep original probabilities exactly, without interpolation
or renormalization. KiTS uses foreground channel 1 only. No synthetic inputs
are used for these performance measurements; synthetic fixtures are used only
in correctness tests. No new inference, downloads or checkpoints.

Requested D: **64, 256, 1,024, 4,096, 16,384, 65,536, 131,072, 262,144,
524,288, 1,048,576**, plus each complete cached input. D is spatial positions
per class, not B×C×D. Sizes above a source's actual length are recorded as
unavailable rather than manufactured; a full input matching an existing D is
measured once, not duplicated.

| Source | B, C used | Full cached probability-map shape |
| --- | --- | --- |
| VOC | 1, 21 | 512×512 |
| Cityscapes | 1, 19 | 1024×2048 |
| ADE20K | 1, 150 | 512×512, 512×648, 512×659, 512×683, or 512×729 |
| KiTS foreground | 1, 1 | 384×384 |

These are input-size experiments, not resized-image inference or new full-cohort
Dice/IoU measurements. All batches have B=1; batch-size policy is not calibrated.

## Fair CUDA controls

RTX 3090; Python 3.10.12; PyTorch **2.11.0+cu130**; Triton **3.6.0**; four host
PyTorch threads. Float32, RMA/Dice, smooth=0, pruning_prob=0.5, max_score.

| Method | Operation |
| --- | --- |
| `full` | Public `safe_screening=False` reference |
| `full_optimized` | Existing optimized CUDA full-sort path, forced independently of size |
| `screened` | Screening on, bypassing only the existing small-input dispatch |
| `production` | Public `safe_screening=True`, with current dispatch unchanged |

All methods use the same public `RankSEG.predict` wrapper and resident tensor.
Private dispatch overrides are set/restored **outside** the timed intervals;
the timing does not include benchmark routing/context-manager work. Untimed
hooks verify whether screening really ran. Production outputs must be bitwise
equal to those of their selected controlled route on each input.

- Three rounds, shuffled input order, rotated/alternating method order.
- Three warm-ups per method/block, seven synchronized wall-time measurements per method/round. Compilation, transfer, cache loading, counting and correctness checks are untimed. No concurrent GPU benchmark jobs.
- Per input: median of the three round medians. Dataset speedup is **mean reference time / mean screening time**, not mean per-input ratios.
- Peak memory: incremental `torch.cuda.max_memory_allocated()` above the resident baseline during a separate warmed call. Includes output/workspaces; excludes input storage, CUDA context and allocator-reserved-but-unused memory. For each input/method use the maximum across rounds, then average across inputs.
- Stable win: reference/screened >1.05 in every round; stable loss: <0.95 in every round; otherwise mixed/near. This margin is not a confidence interval.

Total: **355 unique configurations**, **1,065 timing blocks**, **29,820 timed
predictions**, four methods per configuration.

## Speedup against optimized full sort

The relevant dispatch comparison is `full_optimized / screened`: comparing
only against the slower legacy reference could incorrectly favor screening.
Exclude inputs with every class pruned in this table, so their early-exit gains
do not inflate the candidate-screening result. Natural-image columns each use
10 inputs; the last column gives the active KiTS sample count.

| D | VOC | Cityscapes | ADE20K | KiTS foreground | KiTS active / 10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.894× | 0.788× | 0.669× | 0.865× | 1 |
| 256 | 0.686× | 0.650× | 0.475× | 0.757× | 4 |
| 1,024 | 0.596× | 0.520× | 0.454× | 0.701× | 8 |
| 4,096 | 0.531× | 0.450× | 0.410× | 0.570× | 9 |
| 16,384 | 0.540× | 0.400× | 1.166× | 0.573× | 9 |
| 65,536 | 1.099× | 0.706× | 3.578× | 0.545× | 9 |
| 131,072 | 1.873× | 1.178× | 5.989× | 0.565× | 9 |
| 262,144 | 3.379× | 2.161× | 9.766× | — | — |
| 524,288 | — | 3.812× | — | — | — |
| 1,048,576 | — | 5.515× | — | — | — |

Every configuration at D=1,024 and 4,096 is a stable loss against optimized
full sort. Screening is not uniformly beneficial even where the dataset
average exceeds 1×: VOC at D=65,536 has three slower input medians; Cityscapes
at D=131,072 has two, and ADE20K at D=16,384 has one.

Illustrative mean active-input milliseconds, **optimized full → screening**:

| D | VOC | Cityscapes | ADE20K | KiTS foreground |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.2956 → 0.5572 | 0.2979 → 0.6626 | 0.3015 → 0.7352 | 0.1890 → 0.3318 |
| 16,384 | 0.3248 → 0.6016 | 0.3266 → 0.8171 | 0.9764 → 0.8372 | 0.2030 → 0.3541 |
| 65,536 | 0.6536 → 0.5946 | 0.6182 → 0.8752 | 3.2788 → 0.9163 | 0.2057 → 0.3774 |
| 131,072 | 1.1363 → 0.6068 | 1.0693 → 0.9076 | 6.2575 → 1.0449 | 0.2111 → 0.3740 |

All configurations including all-pruned inputs:

| D | Configurations | Stable win | Mixed / near | Stable loss |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 40 | 1 | 13 | 26 |
| 256 | 40 | 0 | 2 | 38 |
| 1,024 | 40 | 0 | 0 | 40 |
| 4,096 | 40 | 0 | 0 | 40 |
| 16,384 | 40 | 9 | 2 | 29 |
| 65,536 | 40 | 17 | 2 | 21 |
| 131,072 | 40 | 28 | 2 | 10 |
| 262,144 | 30 | 30 | 0 | 0 |
| 524,288 | 10 | 10 | 0 | 0 |
| 1,048,576 | 10 | 10 | 0 | 0 |

The 25 additional full-input configurations have source-specific lengths.
Across all 355 configurations: **120 wins, 21 mixed/near, 214 losses** against
optimized full sort. Against legacy `full`, the counts are 178/4/173 instead;
the choice of reference matters. There are 22 entirely class-pruned configurations.

## Complete cached images: latency and memory

These rows use complete cached probability maps, not sampled D. Natural-image
rows contain all 10 inputs; KiTS contains its nine active foreground slices.

| Source | Legacy full ms | Optimized full ms | Forced screening ms | Production on ms | Optimized / screening |
| --- | ---: | ---: | ---: | ---: | ---: |
| VOC | 3.1273 | 2.1039 | 0.6226 | 0.6052 | 3.379× |
| Cityscapes | 19.2904 | 12.5078 | 1.6394 | 1.6154 | 7.630× |
| ADE20K | 22.1245 | 14.3244 | 1.3373 | 1.3181 | 10.711× |
| KiTS foreground | 0.2666 | 0.2114 | 0.3749 | 0.2201 | 0.564× |

Production already screens these full natural images and bypasses screening
for these KiTS slices. Forced and production timings are separate measurements,
not claims of different algorithms when their routes match. All 30 full natural
images are stable wins; **all nine active full KiTS slices are slower**.
On active KiTS, removing its existing bypass would change approximately
**0.2201 → 0.3749 ms** in this experiment.

Memory is shown alongside screening proportion, with the same active-input selection:

| Source | Screened / active entries | Optimized full peak MiB | Screening peak MiB | Peak reduction |
| --- | ---: | ---: | ---: | ---: |
| VOC | 99.568% | 232.759 | 7.502 | 96.777% |
| Cityscapes | 99.771% | 776.001 | 56.002 | 92.783% |
| ADE20K | 98.897% | 1,397.388 | 49.422 | 96.463% |
| KiTS foreground | 99.918% | 4.556 | 0.152 | 96.662% |

KiTS illustrates a real **memory–runtime tradeoff**: strong relative memory
savings, but roughly 0.163 ms more than optimized full sort. Including its
one all-pruned slice gives 0.1970 → 0.3509 ms and 4.114 → 0.151 MiB; the
direction of the runtime result does not change.

The screening fraction is (H+L)/(H+L+M), summed across active class-position
entries, separately from whole-class pruning. It is not unique-pixel removal,
multiclass eligibility or a memory-saving percentage. At D=4,096 the four
fractions are already 99.518%, 99.843%, 98.859%, and 99.927%, despite the runtime
losses. Avoiding almost all sorting does not eliminate kernel launches,
candidate packing, grouping and metadata synchronization. This is consistent
with the observed small-input overhead, but this run does not attribute time
to individual kernels.

## Correctness and verification

- **1,420/1,420 independent float64 binary-objective checks passed**, covering every configuration and all four methods. Maximum regret **0.835693 float32 eps**, within the unchanged four-eps regression budget. The oracle exhaustively evaluates sorted prefixes, without screening.
- Every method's outputs repeat bitwise over three rounds at fixed input. Production matches its selected controlled route exactly. Screening and legacy full are not promised bitwise equal: 106 configurations differ, with at most **0.222922%** differing output elements in a configuration. These are on/off differences, not a before/after attribution to retry removal.
- Core/helper/harness hashes verified; input hashes agree across rounds, and transfers do not change probabilities. All eight source-cache files were rehashed after completion. All 280 shared sampled-input hashes match the preceding CPU experiment.
- The auditor verifies complete/unique configuration coverage, measured dispatch, H/L/M partitions, no candidate retry, finite timings/memory, recomputed medians/ratios, and consistent objective pass/fail flags. All 1,065 raw journal blocks match the final report.
- Core full suite on PyTorch 2.11/CUDA, including opt-in large-volume indexing and actual 1 GiB score-grid tests: **2,211 passed, 8 skipped**, coverage **85.92%**. CPU-only: **851 passed, 1,368 skipped**, portable coverage **92.60%**; CUDA-dependent tests are skipped there.
- Mini benchmark suite with CUDA: **232 passed**. Related nnU-Net benchmark/count/audit tests in its dedicated PyTorch 2.8 environment: **197 passed**. Documentation builds with Sphinx `-W --keep-going`.
- PyTorch 2.8/Triton compatibility: **106 core CUDA/no-retry tests passed**, including the actual score-grid boundary. CUDA scorer hardening was followed by a complete benchmark replay: all **355 × 4 method output hashes** match the earlier run, including the screened outputs. The tables above use this final-code replay, not the preliminary timings.
- New regression tests cover candidate fractions above 75%, more than 4,194,304 candidates, singleton/group padding limits, pruned/resolved rows, strided inputs, multiclass assignment, float32/float64, and both Triton and portable CUDA. Float64 constant-block fixtures additionally check their analytically known exact optimal mask, avoiding accumulation error in a naive repeated-.49 oracle.

The objective budget is a numerical regression criterion, not a universal proof
or a bound on ground-truth multiclass Dice/IoU. No new quality claims are made.

## Dispatch implications

1. **Keep the distinction between CPU and CUDA.** Real CPU workloads support trying screening without a D gate; the GPU small-input data do not.
2. The existing single-channel bypass helps these real KiTS slices. Their maximum D is 147,456, so this run does **not** establish 262,144 as the exact crossover for larger single-channel volumes.
3. Natural-image channels exceed the existing 16-row gate. Their small sampled inputs expose overhead that the current dispatch does not avoid. Candidate boundaries differ: at tested sizes all VOC inputs win by 131,072, Cityscapes by 262,144, and ADE20K by 65,536. These are dataset-specific observations, not recommended universal constants.
4. A latency-oriented CUDA policy should consider size together with channel/batch structure; these results do not justify deleting every gate or introducing one universal D cutoff. Memory-constrained users may accept a small-input runtime penalty for the measured memory savings.
5. Only CUDA **with Triton** is performance-calibrated here. Portable CUDA correctness is tested, but its performance thresholds cannot be inferred from these measurements.

CUDA thresholds, default behavior and the not-yet-implemented `auto` policy
remain unchanged pending a separate policy decision.

## Reproduction

From `rankseg-benchmark`, with CUDA device access and the pinned local caches:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../env/bin/python scripts/calibrate_screening_cuda.py \
  --include-full \
  --output-dir artifacts/screening-cuda-real-scales-replay
```

The output directory must not exist. Local ignored artifacts are under
`artifacts/screening-cuda-real-scales-2026-09-25-final/`: `results.json`,
`timings.jsonl`, and `summary.json`. They include every timing sample, actual
dispatch, counts, objective diagnostics, unavailable sizes and provenance hashes.
No checkpoints or raw probability caches are added to git.
