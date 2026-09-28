# Exact unique-class assignment fast path — 2026-09-22

This compares the new assignment shortcut against the **already memory-optimized
screened decoder**, not against full sort or argmax. It reuses the 21-volume
nnU-Net engineering subset and 400 mini-benchmark samples. These are not new
complete-cohort results. No inference, checkpoint, probability-cache or
postprocessing changes were made.

## Change and correctness contract

Only the final CUDA/Triton unsmoothed Dice multiclass assignment kernel changes.
Screening certificates, candidate sorting, prefix sums, binary masks, full-image
means and unique-pixel statistics are unchanged.

- If every valid pixel in a tile has exactly one selected binary class, emit
  that class directly. No probabilities, incremental scores or active-class
  statistics are needed in that branch. Padding does not prevent the shortcut.
- Mixed tiles use the original vectorized score expression, division rounding,
  overlap eligibility, first-class tie order and all-pruned/void handling. The
  scored helper's body is AST-identical to the pre-change assignment body.
- Keep the original scoring in an internal helper. Above 64 padded classes,
  isolate that helper as a GPU device function to avoid the large register
  interference seen in the inline trial. This is not an additional kernel
  launch. Tile sizes and warp configuration remain unchanged.
- No probability-argmax substitution, new epsilon rule, approximate score,
  runtime autotuner, extra kernel launch or host synchronization is introduced.

A uniquely selected class is the only eligible class at that pixel, so direct
assignment preserves the result. This is about the **final binary masks**, not
screening candidates: screened-negative positions can still be assigned through
the existing unassigned-pixel `max_score` fallback.

The acceptance criterion here is **bitwise identical class labels for identical
binary masks and statistics**, not an epsilon objective tolerance. This differs
from the preceding optimization, which changed the unique-sum reduction order.

## Protocol

- RTX 3090; GPU benchmarks run serially, with four CPU threads.
- nnU-Net: PyTorch 2.8.0+cu128 / Triton 3.4.0; first held-out case per fold for
  each dataset, plus `liver_43`.
- Mini: PyTorch 2.11.0+cu130 / Triton 3.6.0; 100 cached examples each for VOC,
  Cityscapes and ADE20K. KiTS uses 100 foreground-only, single-channel slices
  and forced screening as an unchanged-path control.
- Three warm-ups and 11 alternating timed repeats. Dataset latency is the mean
  of per-input median synchronized wall times. Loading, compilation, independent
  oracles, exact audits, metrics and official nnU-Net postprocessing are excluded.
- Additional assignment-only CUDA-event timings use identical frozen inputs.
- Memory is peak additional PyTorch allocated GPU memory, including output but
  excluding resident probabilities, reserved memory and CUDA context.

## Final runtime and memory

| Dataset | Samples | Before → after (ms) | Speedup | Peak memory, unchanged (MiB) |
|---|---:|---:|---:|---:|
| nnU-Net Pancreas | 5 volumes | 6.321 → 4.644 | 1.36x | 306.7 |
| nnU-Net HepaticVessel | 5 volumes | 3.190 → 2.447 | 1.30x | 141.1 |
| nnU-Net Lung | 5 volumes | 9.916 → 8.974 | 1.10x | 1,113.3 |
| nnU-Net Liver | 5 + 1 regression volumes | 21.275 → 15.515 | 1.37x | 1,111.1 |
| VOC | 100 images | 0.760 → 0.722 | 1.05x | 10.6 |
| Cityscapes | 100 images | 2.723 → 2.476 | 1.10x | 76.7 |
| ADE20K | 100 images | 2.355 → 2.080 | 1.13x | 90.9 |
| KiTS, unchanged binary control | 100 slices | 0.302 → 0.300 | 1.00x | 0.269 |

All 321 multiclass samples had lower median latency; the smallest observed
per-input speedup was 1.026x. Assignment-only dataset-mean timings improved
1.61–2.42x. KiTS does not enter the modified kernel; its aggregate latency
difference was -0.43%, with individual timing fluctuations in both directions.
Every paired case had exactly the same measured peak allocated memory.

These measurements do not guarantee a universal speedup. The shortcut helps
when entire tiles contain only uniquely selected pixels. Both the initial
per-pixel mixed-tile variant and an inline high-class variant had substantial
synthetic regressions and were discarded. The final implementation retains
the original mixed-tile computation and isolates its high-class workspace.
Across the extended 60-case matrices, the worst measured assignment-kernel
overhead was **8.1% for float32** and **3.7% for float64** (not end-to-end
decoder slowdowns). These deliberately adversarial cases
include alternating unique/overlapping pixels, all-overlapping/all-unassigned
tiles and odd spatial lengths. The shortcut is not advertised as free when
none of its tiles can take the fast path.

The resource diagnostic explains why isolation matters. At `D=65537`, float32
and the original launch configuration, the rejected inline variant raised
128-class register usage from 152 to 191, while the final version restored 152.
For 255 classes, reported spills rose from 4 to 48 in that trial and returned
to 4 in the final version. This removed a roughly 2x synthetic slowdown.
These are compiler/device-specific observations, not universal resource bounds.

## Exactness and the existing Liver repeatability limitation

For **all 321 multiclass samples**, the new and old assignment entrypoints
produced identical labels from the same binary masks and full-image means.
Each sample also passed direct kernel comparisons with identical statistics
under `max_score` and `void`, including both extreme int64 void labels:
**963 exact kernel comparisons**, all with zero differing pixels.

Independent end-to-end calls matched exactly on **420/421 samples**: all 400
mini samples and 20/21 nnU-Net volumes. Dice and IoU were unchanged on these
cases. The exception was the existing `liver_43` CUDA prefix-scan variability:

- Its independent end-to-end calls differed at 308 final pixels in this run;
  its fixed-mask assignment comparisons still had **zero** differences.
- A binary-only repeatability diagnostic reproduced differences between calls
  to the **old decoder itself**. On one fixed 2,191,894-element candidate array,
  repeated `torch.cumsum` calls varied by up to 0.25 in float32 prefixes and
  selected candidate counts 908565, 908640 or 908850.
- Across eight old/new binary calls, all objective regrets were below 0.418
  float32 eps. This diagnostic uses the binary path, unchanged across trials.
- Averaged over the six Liver volumes, the independent-call metric differences
  were +0.000514 Dice percentage points and +0.000641 IoU percentage points.
  These are repeatability effects, **not accuracy gains from this shortcut**.

Full-prefix float64 binary objective checks passed on all 421 samples; maximum
regret was 0.4811 float32 eps for nnU-Net and 0.9229 eps for mini. Those are
checks of the unchanged binary stage, not relaxed tolerances for assignment.
No deterministic-scan override or retry was added to inference.

## Regression tests

- Full primary CPU/CUDA suite: **1,424 passed, 12 skipped**, coverage **88.16%**.
- PyTorch 2.8 screening suite with large-memory tests enabled: **608 passed**,
  including unique and overlapping assignment at **221,773,824 pixels**.
- New fast-path test file: **167 tests**, covering float32/float64, 1–256
  channels, partial tiles, strided inputs, class zero, overlaps, unassigned and
  all-pruned inputs, ties/adjacent floats, extreme void labels and input
  immutability. It exhaustively checks all 512 three-class/three-pixel binary
  masks for both dtypes and both unassigned policies.
- Exact before/after compatibility: **450 checks** — 216 default-path, 180
  unsupported metric/smooth, 18 CPU screening and 36 binary-screening cases.
- Sphinx HTML builds with warnings treated as errors.

The default `safe_screening=False`, CPU/no-Triton fallback, binary output, IoU
and positive smoothing are not changed by this update. Testing is empirical
evidence, not a claim that every possible hardware/input combination is proven.

## Reproduction and artifacts

Preserved sources and raw outputs are under
`outputs/screening-unique-fastpath-2026-09-22/`:

- `baseline/`, `optimized/`: before/after core sources.
- `nnunet.json`, `mini.json`, `summary.json`: final paired timings, peaks,
  provenance, metrics and exact-assignment audit records, with source hashes.
- `synthetic-assignment.json`, `synthetic-float64.json`,
  `synthetic-float32-extended.json`: synthetic assignment checks and timings.
- `binary-repeatability.json`: old-decoder and isolated-prefix repeatability.
- `full-tests.log`, `torch28-tests.log`, `compatibility.json`, `sphinx.log`.
- `kernel-resources.json`: compiler metadata for the high-class regression.
- Files suffixed `-v1`, `-v2`, `-v3`, `-v4` and `tuning.json` are development trials,
  **not** the final results above.

From `rankseg-nnunet-benchmark`, use fresh output filenames:

```bash
env/bin/python -B scripts/benchmark_unique_assignment.py \
  --baseline outputs/screening-unique-fastpath-2026-09-22/baseline \
  --suite nnunet --repeats 11 --output /tmp/unique-nnunet-new.json

../env/bin/python -B scripts/benchmark_unique_assignment.py \
  --baseline outputs/screening-unique-fastpath-2026-09-22/baseline \
  --suite mini --repeats 11 --output /tmp/unique-mini-new.json
```
