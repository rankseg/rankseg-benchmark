# Complete nnU-Net cohorts — screening benchmark

Updated **2026-09-21**. This report now uses one setup: complete whole-volume
nnU-Net OOF cohorts. The previous subsets and natural-image/KiTS results have been
removed from this document, not deleted from their original result files.

## Setup

RTX 3090 (24 GiB), float32, PyTorch 2.8.0+cu128, Triton 3.4.0. One complete 3D
volume per call. All native channels, including background, are retained.
Every RankSEG path uses Dice / RMA, smooth=0, pruning_prob=0.5 and max_score.
The fixed official postprocessing is applied identically to every method.

| Path | Description |
| :--- | :--- |
| Argmax | Voxel-wise argmax of the same cached probabilities |
| Full | Original default RMA full sort |
| Optimized full | Direct-argmax full sort, with screening forcibly bypassed |
| Screened | Direct argmax with fused screening; small-input bypasses disabled, workspace fallbacks retained |

Two warmups and five synchronized timed repeats with rotating method order.
Latency is the mean of per-volume medians, excluding inference, I/O, transfers,
compilation, official postprocessing, metrics and the numerical oracle.
Dice / IoU are averaged over cases for each foreground label, then over labels;
both-empty labels remain undefined. IoU is an outcome, not a separately optimized
objective. See [reproduction protocol](SCREENING_BENCHMARK.md).

## Complete nnU-Net cohorts

The fresh run was launched on **2026-09-21**, using the same current RankSEG source snapshot, including the
[very-wide row-indexing fix](SCREENING_INDEX_FIX_2026-09-20.md), for all four datasets.
Jobs run serially in the order Liver → Pancreas → HepaticVessel → Lung. Source hashes
are checked before and after each job; changes abort the sequence rather than mix versions.

| Dataset | Volumes |
| :--- | ---: |
| Pancreas | 281 |
| HepaticVessel | 303 |
| Liver | 131 |
| Lung | 63 |
| Total | 778 |

This is the rerun setup, not a completed-results claim. Job completion and validation
states are recorded separately in `outputs/screening-complete-2026-09-21/run_status.json`.
Each completed dataset is independently audited automatically; a numerical failure
remains `failed_objective`, even when all cases were processed.

These four known-positive Full-16 datasets form a targeted reproduction, not a new
unbiased dataset selection or a replacement for the published 16-dataset evaluation.
No training, checkpoint selection or probability generation is repeated.

Fresh measurements and independent audits will be written to
`outputs/screening-complete-2026-09-21/`. Earlier complete results are retained in
`outputs/screening-full-cohorts/`; they are not relabeled as fresh measurements.
The original published Full-16 results are unchanged.

## Screening proportions

An untimed follow-up pass is queued after all four benchmark jobs finish, so it
does not compete for GPU resources or alter the running source snapshot. It reuses
the same probability caches and CUDA certificate kernels; source, cache, shape and
environment checks must match the completed benchmark.

Per-case/per-class counts will distinguish:

- **Pixel screening:** confirmed positive (H) and negative (L) entries, versus
  undecided candidates (M). Primary proportion is `(H + L) / (H + L + M)`, summed
  over active classes and volumes before division.
- **Class pruning:** excluded whole-class rows, reported separately, not counted
  as pixel-level screening. Background and each foreground label are also summarized
  separately; counts concern class-probability entries, not unique spatial voxels.
- **Actual sorting reduction:** workspace fallback sorts the whole affected row,
  so its H/L certificates do not count as effective sorting savings. Actual real
  entries sent to sort and padded workspace elements are recorded separately.

The supplemental files will be
`outputs/screening-complete-2026-09-21/screening-proportions/{dataset}.json`, linked
to the benchmark by its SHA-256 and case IDs. Existing timing/metric reports are not
rewritten. These are pending measurements, not already collected full-cohort counts.

## Correctness and failure reporting

- Audit every cached case against the declared held-out fold assignments and recorded
  inference lineage before selection. This is a cache-provenance audit, not independent
  proof of model training or patient identity.
- Verify binary objectives against an independent float64 full-prefix oracle.
  The numerical regression budget remains **four float32 eps**, not a universal
  mathematical bound or an inference tie rule.
- This run explicitly uses `--objective-failure-policy record`: a finite excess is
  retained as a **failed** check for that method, and all remaining methods/cases are
  still evaluated. Structural errors, nonfinite regrets and unexpected inference
  errors still stop the run. The default policy remains `stop`.
- A finished run is not automatically a passed validation. The independent audit
  reports `failed_objective` if any recorded objective check failed, even when
  confusion counts, aggregation and timings validate successfully.
- CUDA OOM is recorded without CPU retry. Full-cohort comparisons must retain every
  case for the compared methods; do not substitute a smaller paired cohort silently.
  Oracle OOM is unverified, never passed.
- Screening may change masks and individual-case Dice/IoU. Small objective regret
  does not guarantee identical masks or bound ground-truth metric changes.

The preceding Liver attempt stopped at `liver_19`: both full-sort controls had
4.398215 eps regret, while a separate screened check had 0.212232 eps regret.
The new run does not change the algorithm or relax the threshold to hide this;
it measures the complete cohort and retains each method's failures explicitly.

The updated benchmark/auditor passed **87 CUDA tests** (CPU: **85 passed, 2 skipped**),
including checks that recorded failures are never relabeled as passed, later methods
and cases still run, and strict/default validation continues to reject failures.

## Evidence and prior results

All probabilities, checkpoints and raw reports remain local and Git-ignored.
The previous mixed report is archived at
`outputs/screening-engineering/SCREENING_RESULTS_PRE_COMPLETE_2026-09-21.md`;
its original raw measurements remain in `outputs/screening-engineering/`.
Neither the old failed Liver reports nor completed prior results were overwritten.
