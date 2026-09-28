# CPU screening comparison — 2026-09-24

CPU screening is useful on these real caches: all four dataset means improve at
1, 4 and 8 threads. It does **not** require Triton. Tiny inputs and dense-candidate
fallbacks can nevertheless regress, so these results support including CPU in a
future `auto` policy, not enabling screening unconditionally on every CPU input.

No RankSEG algorithm, dependency, dispatch threshold or default was changed by
this experiment. Measurements use the current local source checkout, not an
assertion about the published 0.0.6 wheel.

## Protocol and scope

- Intel Core i9-12900K; Linux; Python 3.10.12; PyTorch 2.11.0+cu130, **CPU tensors only**.
- Float32, batch size 1, RMA/Dice, `smooth=0`, `pruning_prob=0.5`, `max_score`.
- Ordinary public `safe_screening=False` versus `True`; no forced dispatch or kernel overrides.
- VOC, Cityscapes and ADE20K: first **10 cached images each**. KiTS: **10 uniformly spaced slices per fold**, including both endpoints, across all five folds, **50 slices total**. Selection does not inspect labels or outcomes. These are subsets, not full-cohort release scores.
- All inputs use pinned cache revision `1884f0766268cdc62730e696f24dcc913d551b35`. No downloads, inference or checkpoint changes.
- KiTS RankSEG receives foreground channel 1 in multilabel mode; argmax receives both cached channels. Natural images use all channels in multiclass mode.
- Each of the 80 inputs is evaluated at 1/4/8 threads. One warm-up per method, five measured calls per method, rotating method order. Reported latency is the mean of per-input medians. Timing jobs run serially; RSS sampling and test suites run separately.
- Timings include public decoding, channel routing and output conversion; exclude cache loading, metric computation, input hashing, screening diagnostics and float64 oracles. They are **postprocessing**, not model-inference timings.
- Natural-image Dice/IoU use image means over GT-present classes, preserving the repository's ignore rules. KiTS scores here are **sampled foreground-slice means**, not case/fold-balanced or whole-volume Dice. IoU is evaluated but not separately optimized.
- Public on/off differences include the complete opt-in implementation, including avoiding work on pruned classes and its direct-argmax search. They do not isolate the pixel-screening certificate's contribution alone.

## Real-input runtime

### Four threads

| Dataset | Inputs | Argmax ms | Screening off ms | Screening on ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| VOC | 10 images | 17.5078 | 102.1902 | 43.2681 | 2.362× |
| Cityscapes | 10 images | 145.7769 | 870.5169 | 492.1910 | 1.769× |
| ADE20K | 10 images | 35.0249 | 868.5792 | 351.5118 | 2.471× |
| KiTS | 50 slices | 9.3301 | 5.2796 | 0.3349 | 15.765× |

Argmax is the existing generic `probs.argmax(1)` implementation. In particular,
the KiTS reference is not a specialized two-channel comparison kernel; these
measurements do not claim RankSEG is faster than every possible argmax implementation.

### Thread sensitivity

| Dataset | 1 thread off → on (ms) | Speedup | 8 threads off → on (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| VOC | 205.1382 → 68.2719 | 3.005× | 78.5825 → 38.6138 | 2.035× |
| Cityscapes | 1958.6909 → 960.4738 | 2.039× | 728.9469 → 414.5733 | 1.758× |
| ADE20K | 2132.0900 → 793.7173 | 2.686× | 654.1929 → 279.7198 | 2.339× |
| KiTS | 5.3771 → 0.5867 | 9.165× | 5.3116 → 0.3286 | 16.165× |

The CUDA small-input bypass does not apply on CPU. KiTS therefore actually
executes screening when its foreground class is active. A prior CUDA result in
which forcing KiTS screening was slower cannot be transferred to this CPU path.

## Screening counts

Counts below cover each selected input once, not three times for three thread
settings. The partitions agree across thread counts. One entry is one probability
for one class at one spatial position, **not a unique multiclass pixel**.

| Dataset | Active entries | Certified positive H | Certified negative L | Unresolved M | (H+L)/active | Whole-class pruning / all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VOC | 6,029,312 | 2,577,070 | 3,423,326 | 28,916 | 99.5204% | 89.05% |
| Cityscapes | 228,589,568 | 20,803,203 | 207,361,957 | 424,408 | 99.8143% | 42.63% |
| ADE20K | 38,299,648 | 3,011,960 | 34,878,606 | 409,082 | 98.9319% | 92.01% |
| KiTS | 6,193,152 | 228,069 | 5,959,790 | 5,293 | 99.9145% | 16.00% |

There were **zero candidate-budget fallback rows** on the real sample set.
Actual screening dispatch was checked separately from timed calls. Whole-class
pruning is a separate mechanism from H/L screening; the latter denominator
excludes already-pruned classes. Original probabilities remain available for
multiclass assignment.

## Output quality and numerical checks

### Four-thread scores

| Dataset | Argmax Dice / IoU (%) | Off Dice / IoU (%) | On Dice / IoU (%) | Dice Δ (pp) | IoU Δ (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| VOC | 96.801432 / 94.056276 | 96.822833 / 94.091513 | 96.821257 / 94.088932 | −0.001576 | −0.002582 |
| Cityscapes | 82.273381 / 75.919114 | 82.270310 / 75.901256 | 82.275102 / 75.906726 | +0.004792 | +0.005471 |
| ADE20K | 58.982678 / 51.823310 | 61.353411 / 53.728973 | 61.355102 / 53.731610 | +0.001691 | +0.002637 |
| KiTS | 64.431710 / 58.141307 | 66.991208 / 60.179624 | 66.991222 / 60.179649 | +0.000014 | +0.000025 |

These subsets are not selected to demonstrate ground-truth quality improvements.
For example, Cityscapes RankSEG is slightly below argmax here. Screening is a
computational change, not a guarantee of improved empirical Dice.

- **480/480 real-input binary-objective checks passed**: both methods, all 80 inputs, all three thread counts. Each uses an independent exhaustive float64 full-prefix oracle and the unchanged four-float32-eps regression budget. Maximum regret: **1.113718 eps**, on both methods.
- Natural-image output differences at four threads: VOC **301 pixels / 8 images**, Cityscapes **16,141 pixels / 10 images**, ADE20K **577 pixels / 10 images**. KiTS changes **4 pixels / 3 slices**.
- Worst four-thread per-input Dice changes: VOC **−0.023619 pp**, Cityscapes **−0.017632 pp**, ADE20K **−0.002810 pp**, KiTS **−0.002589 pp**. Small mean changes are not per-case guarantees.
- KiTS boundary masks also vary slightly with CPU thread count: 9 changed pixels at one thread, 4 at four threads, 5 at eight threads, relative to the corresponding off path. All objective checks still pass.
- All input hashes remain unchanged; a post-run audit verifies identical input/label hashes across thread settings, selection coverage, timing medians, metrics, count partitions and aggregate summaries.

The numerical budget is a regression criterion, **not a universal error proof**,
nor a promise of bitwise-equal binary or final multiclass masks.

## CPU memory: approximate RSS, separate from timing

First selected input per dataset, four threads, three fresh processes per
method. Inputs/imports are resident before baseline; the output remains alive
through the final sample. Reported values are medians of **incremental observed
peak process RSS**, with the range across the three repeats.

| Dataset | Off MiB, median (range) | On MiB, median (range) |
| --- | ---: | ---: |
| VOC | 281.59 (239.59–281.84) | 133.85 (105.10–190.11) |
| Cityscapes | 1699.09 (1694.12–1701.43) | 582.54 (580.30–582.59) |
| ADE20K | 2207.60 (2205.26–2210.20) | 725.11 (725.10–725.36) |
| KiTS | 14.95 (8.90–15.48) | 8.09 (7.95–8.86) |

Cityscapes and ADE20K show approximately **65.7% and 67.2%** lower observed extra
RSS on these representative inputs. VOC and especially KiTS are noisier; the
KiTS screened operation yielded only two RSS samples per repeat. Treat those
values as rough observations, not precise memory-reduction estimates.

Sampling targets 1 ms, but OS scheduling can delay it (largest recorded gap:
71.83 ms in one Cityscapes off repeat). RSS includes allocator/page residency
effects and can miss short-lived peaks. It is **not** PyTorch live allocation,
not a whole-cohort memory average, and not directly comparable to the GPU
allocated-memory measurements. Near-100% screening does not imply near-100%
memory reduction: input, output and multiclass workspaces remain necessary.

## Synthetic boundaries

**120 configurations**: 1/4/8 threads × 1/3 channels × D =
64/4,096/65,536/262,144/1,048,576 × sparse/uniform/dense-candidate/all-pruned
profiles. Single-channel output is multilabel; three-channel output is multiclass.
Seed 3401, two warm-ups, nine rotating repeats. These are stress distributions,
not additional datasets; three-channel uniform values are independent probabilities.

All **240/240** binary-objective checks pass; maximum regret is **0.478159 eps**.
The dense-candidate profile is 0.49 everywhere with one 0.51 entry per class,
deliberately triggering the existing candidate-budget fallback.

Selected four-thread examples:

| Channels / D | Profile | Off ms | On ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1 / 64 | Uniform | 0.0867 | 0.1352 | 0.642× |
| 3 / 64 | Uniform | 0.1966 | 0.3681 | 0.534× |
| 1 / 4,096 | Dense candidates | 0.1335 | 0.1952 | 0.684× |
| 1 / 65,536 | Dense candidates | 0.8086 | 0.9035 | 0.895× |
| 1 / 65,536 | Uniform | 3.5609 | 0.8228 | 4.328× |
| 1 / 262,144 | Sparse | 3.0172 | 0.2800 | 10.775× |
| 3 / 1,048,576 | Uniform | 160.1445 | 98.2420 | 1.630× |

Thus CPU screening has clear useful regimes, but input size alone cannot
guarantee a speedup. More CPU architectures, thread settings, distributions and
dtypes are needed before claiming a universal threshold. GPU thresholds should
not be reused unchanged.

## Reproduction and artifacts

From `rankseg-benchmark`, using the existing workspace environment and sibling
RankSEG/nnU-Net benchmark checkouts:

```bash
../env/bin/python scripts/benchmark_screening_cpu.py \
  --output-dir artifacts/screening-cpu-replay
../env/bin/python scripts/benchmark_screening_cpu_synthetic.py \
  --output artifacts/screening-cpu-replay/synthetic.json
../env/bin/python scripts/measure_screening_cpu_memory.py \
  --input-dir artifacts/screening-cpu-replay
../env/bin/python scripts/audit_screening_cpu.py artifacts/screening-cpu-replay
```

Scripts refuse to overwrite existing final outputs. The real-input runner
records cache hashes, selected row indices, source hashes and per-input records.
Existing pinned caches, PyArrow and the nnU-Net oracle helpers are required;
the separate memory sampler also requires `psutil`.

Local raw artifacts (ignored, not committed):
`artifacts/screening-cpu-2026-09-24/summary.json`, `records.jsonl`,
`synthetic.json`, `memory.json`, and the four representative memory-input files.
No checkpoints are included.

## Final validation

- Core CPU suite: **841 passed, 1,343 skipped**, **92.71% CPU coverage**, using the existing `--cov-config=.coveragerc.cpu` configuration. CUDA-only tests are not part of this CPU run.
- Full mini-benchmark suite: **132 passed, 16 skipped**; includes 33 new CPU-harness regression tests.
- Read-only real-result audit: **12 dataset/thread groups, 240 sample/thread pairs**, all coverage, hash, partition and summary checks passed.
- Across real and synthetic comparisons: **720/720 objective checks passed** for the two paths combined. No OOM, no modified probability inputs, and no RankSEG source changes.
- The first core test invocation omitted the CPU coverage configuration: all 841 tests passed, but the default coverage gate included unexecuted CUDA code and failed at 73.01%. Rerunning with the repository's existing CPU configuration passed the 85% gate; no coverage settings or thresholds were edited.

The evidence favors a CPU-capable `auto` policy with a separately measured
small-input rule. It does not justify globally forcing `True`, copying CUDA
cutoffs, or promising identical masks. This experiment leaves the policy decision
and implementation unchanged.
