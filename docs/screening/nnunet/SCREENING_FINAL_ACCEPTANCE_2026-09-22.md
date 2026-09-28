# Screening final acceptance — 2026-09-22

**Status: measurements complete; screened-path acceptance checks passed on these cohorts.**

## Acceptance outcome

- Screened numerical checks: **passed**.
- Independent metrics/timing audits: **passed**.
- Complete screening diagnostics: **yes**.
- Argmax is included as a reference, with every foreground confusion count compared against the original published benchmark.
- This is empirical acceptance of the frozen implementation on the stated cohorts, not proof for every possible input.
- Screening and unscreened decoding need not be bitwise identical; binary objective tolerance and ground-truth metrics are reported separately.
- Any baseline OOM or objective-budget failures below remain unresolved baseline limitations, not passed tests.

Machine-readable progress is recorded in
`outputs/screening-final-acceptance-2026-09-22/run_status.json`.

## Scope and protocol

This freezes the current RankSEG implementation after fused validation,
candidate workspace fusion and cached unique-class identities. It reruns the
four previously selected complete nnU-Net OOF cohorts, not a new outcome-selected
subset and not the published Full-16 evaluation.

| Cohort | Volumes |
| --- | ---: |
| MSD Pancreas | 281 |
| MSD HepaticVessel | 303 |
| MSD Liver | 131 |
| MSD Lung | 63 |
| Total | 778 |

Every method receives the same cached probabilities, native channel mapping and
whole-volume shape, with background retained in decoding. The fixed official
postprocessing is applied exactly once and identically to all methods. No
training, checkpoint changes, inference or probability regeneration occurs.
Held-out fold placement and recorded inference lineage are audited; this is not
independent proof of the original model training or patient identity.

| Method | Interpretation |
| --- | --- |
| Argmax | Voxel-wise argmax, reference |
| Full | Normal public RMA with `safe_screening=False` |
| Screened | Normal public RMA with `safe_screening=True`, production dispatch |
| Optimized full | Auxiliary direct-argmax full-sort control, screening forcibly bypassed |

All RankSEG paths use Dice/RMA, smooth=0, pruning_prob=0.5 and multiclass
max-score assignment. **IoU is measured, not separately optimized.** The auxiliary
optimized-full control is not presented as ordinary `safe_screening=False`.

RTX 3090, float32, Torch 2.8.0+cu128 and Triton 3.4.0. Two warm-ups and five
synchronized repeats, rotating method order. GPU jobs run serially. Timing is
decoder wall time, excluding inference, loading/transfers, compilation, official
postprocessing, metric evaluation, screening diagnostics and numerical oracles.
Report means of per-volume median times, not model-inference speedups.

Dice/IoU are case means within each foreground label, then means over labels.
Both-empty labels remain undefined. Memory is additional peak PyTorch allocated
GPU memory including output but excluding resident probabilities, CUDA context
and reserved memory. Report both average per-case peaks and the maximum peak.

## Checks and failure accounting

- Every selected case must be retained. OOM is reported explicitly, with no CPU
  retry, resized input or silent reduction to an easier cohort.
- Original full sort previously OOMed on some large cases. Full-cohort aggregates
  and paired-subset comparisons will be separated; incomplete default full-sort
  results must not be labeled complete-cohort metrics.
- Recompute metric aggregation independently, and compare every argmax confusion
  count against the original published-case evidence.
- Check binary objectives against an exhaustive float64 full-prefix oracle with
  the unchanged four-float32-eps regression budget. `record` continues after a
  finite budget excess but preserves a failed result. Oracle OOM remains unverified.
- The bounded oracle sorts in the input dtype, converts the sorted values to
  float64, accumulates every prefix and evaluates scores in bounded chunks.
  Float32-to-float64 conversion preserves value order. This only reduces audit
  workspace; it does not alter RankSEG, skip candidate volumes or relax the budget.
- Compare masks and foreground metrics across methods. An eps-level objective
  difference does not imply identical labels or bound ground-truth metric changes;
  numerical checks and observed Dice/IoU changes are reported separately.
- A separate untimed pass records H/L/M screening proportions, class pruning,
  workspace fallbacks and actual sorting reduction. Counts concern class-probability
  entries, not unique spatial voxels, and are not percentages of total memory saved.

## Reproduction

RankSEG runs from the frozen package in
`outputs/screening-final-acceptance-2026-09-22/snapshot/`. Source hashes are checked
between jobs; all raw reports and logs go to the same artifact directory. Prior
reports and original published results are not overwritten.

The launcher is:

```bash
env/bin/python -B scripts/run_screening_acceptance.py \
  --output-dir outputs/screening-final-acceptance-2026-09-22 \
  --rankseg-path outputs/screening-final-acceptance-2026-09-22/snapshot
```

For a rerun, use a fresh output directory and a verified source snapshot. Existing
status/reports are protected against overwrite. A finished sequence is not by
itself a passed acceptance; individual validation and method statuses must be read.

Generate the final tables after the serial sequence finishes (or wait for it):

```bash
CUDA_VISIBLE_DEVICES='' env/bin/python -B scripts/summarize_screening_acceptance.py \
  outputs/screening-final-acceptance-2026-09-22 --wait-timeout 21600 \
  --document docs/SCREENING_FINAL_ACCEPTANCE_2026-09-22.md
```

The reporter verifies the frozen source hashes, all four cohort sizes, confusion
counts, method coverage, recorded failures and screening-report provenance. It
generates `summary.json`, `RESULTS.md` and the tables in this document. It refuses
to replace an existing result or a document no longer marked pending. Its wait
loop reads only the small status file, performs no CUDA work and does not inspect
per-case running logs. A timeout/error does not generate a passing report; inspect
`summary.log` alongside `run_status.json` if final tables are absent.

## Regression-test evidence

- This acceptance harness: **174 passed** before the full-cohort run, including
  CPU/CUDA comparisons of the bounded and original exhaustive float64 oracles,
  source/provenance controls, decoder dispatch and failure accounting.
- Final screening-proportion collector tests: **39 passed**, including rejection
  of production small-input bypass cases that would misstate actual sort savings.
- Independent report-generation tests: **23 passed**, covering metric/memory/timing
  consistency, incomplete cohorts, explicit OOM subsets, numerical failures,
  pending-document protection and waiting for completion without premature success.
- The unchanged core snapshot retains the immediately preceding validation:
  **2,127 passed / 16 skipped** in the primary CPU/CUDA suite and **1,315 passed**
  in the Torch 2.8/Triton 3.4 screening suite with large-memory tests enabled.
  These are prior core-suite runs, not newly rerun counts; this acceptance adds
  the four complete-cohort comparisons below without changing core code.

The test groups overlap and must not be summed as a count of distinct tests.

## Completed measurements

Each complete method is aggregated over the entire cohort. Dashes mark an incomplete
default-full control; its available subset is compared separately below. Memory is
additional peak allocated MiB (mean of case peaks / largest case peak).

| Dataset | Method | Completed | Dice (%) | IoU (%) | Mean decoder ms | Peak MiB: mean / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| liver | Argmax | 131/131 | 81.00 | 73.36 | 5.40 | 1756.7 / 5922.0 |
| liver | Full (screening off) | 93/131 | — | — | — | OOM: 38 |
| liver | Optimized full (auxiliary) | 131/131 | 81.85 | 74.19 | 262.99 | 7611.0 / 16780.0 |
| liver | Screened (production) | 131/131 | 81.85 | 74.18 | 9.91 | 1343.0 / 2961.0 |
| pancreas | Argmax | 281/281 | 68.40 | 57.06 | 0.64 | 200.9 / 4506.0 |
| pancreas | Full (screening off) | 280/281 | — | — | — | OOM: 1 |
| pancreas | Optimized full (auxiliary) | 281/281 | 69.57 | 57.90 | 56.03 | 1618.4 / 12770.0 |
| pancreas | Screened (production) | 281/281 | 69.57 | 57.90 | 2.58 | 285.7 / 2253.0 |
| hepaticvessel | Argmax | 303/303 | 68.63 | 55.17 | 0.45 | 139.4 / 362.0 |
| hepaticvessel | Full (screening off) | 303/303 | 69.17 | 55.64 | 51.03 | 2233.7 / 5799.0 |
| hepaticvessel | Optimized full (auxiliary) | 303/303 | 69.17 | 55.64 | 41.27 | 1186.5 / 3080.0 |
| hepaticvessel | Screened (production) | 303/303 | 69.16 | 55.64 | 2.10 | 209.4 / 543.0 |
| lung | Argmax | 63/63 | 72.29 | 60.24 | 1.41 | 560.5 / 1272.0 |
| lung | Full (screening off) | 63/63 | 72.75 | 60.82 | 176.79 | 6165.9 / 13992.0 |
| lung | Optimized full (auxiliary) | 63/63 | 72.75 | 60.82 | 150.21 | 3363.2 / 7632.0 |
| lung | Screened (production) | 63/63 | 72.75 | 60.81 | 5.40 | 770.7 / 1749.0 |

### Matched comparisons

Default-full rows use only cases completed by both methods and explicitly show the
smaller paired cohort. Optimized full is a distinct auxiliary control, not a
replacement silently relabeled as normal screening-off behavior.

| Dataset | Comparator | Paired volumes | Screened speedup | Mean peak reduction | Dice Δ (pp) | IoU Δ (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| liver | Default full | 93 | 31.88× | 90.63% | 0.00175 | 0.00185 |
| liver | Optimized full | 131 | 26.54× | 82.35% | -0.00490 | -0.00753 |
| pancreas | Default full | 280 | 26.67× | 90.62% | -0.00014 | -0.00000 |
| pancreas | Optimized full | 281 | 21.68× | 82.35% | -0.00040 | -0.00022 |
| hepaticvessel | Default full | 303 | 24.34× | 90.63% | -0.00141 | -0.00206 |
| hepaticvessel | Optimized full | 303 | 19.69× | 82.36% | -0.00211 | -0.00310 |
| lung | Default full | 63 | 32.72× | 87.50% | -0.00163 | -0.00231 |
| lung | Optimized full | 63 | 27.80× | 77.08% | -0.00144 | -0.00192 |

### Screening proportion versus measured memory reduction

Screening proportions exclude pruned whole classes and count class-probability entries.
Sorting avoidance includes workspace fallbacks. Neither is a whole-memory percentage.

| Dataset | H+L / active entries | Effective sorting avoidance | Class-pruned / all entries | Peak reduction vs optimized full |
| --- | ---: | ---: | ---: | ---: |
| liver | 99.98% | 99.98% | 5.27% | 82.35% |
| pancreas | 100.00% | 100.00% | 2.66% | 82.35% |
| hepaticvessel | 99.99% | 99.99% | 0.51% | 82.36% |
| lung | 100.00% | 100.00% | 0.78% | 77.08% |

### Numerical checks

All recorded failures remain failures. Completing measurements does not waive the
four-eps budget. Baseline failures and screened-path checks are listed separately.

| Dataset | Method | Passed / failed / unverified completed cases | Maximum regret (float32 eps) |
| --- | --- | ---: | ---: |
| liver | Full (screening off) | 91 / 2 / 0 | 4.398215 |
| liver | Optimized full (auxiliary) | 128 / 3 / 0 | 5.836032 |
| liver | Screened (production) | 131 / 0 / 0 | 0.984809 |
| pancreas | Full (screening off) | 280 / 0 / 0 | 1.670840 |
| pancreas | Optimized full (auxiliary) | 281 / 0 / 0 | 1.555248 |
| pancreas | Screened (production) | 281 / 0 / 0 | 0.952679 |
| hepaticvessel | Full (screening off) | 303 / 0 / 0 | 1.565506 |
| hepaticvessel | Optimized full (auxiliary) | 303 / 0 / 0 | 1.459166 |
| hepaticvessel | Screened (production) | 303 / 0 / 0 | 0.806734 |
| lung | Full (screening off) | 63 / 0 / 0 | 1.004778 |
| lung | Optimized full (auxiliary) | 63 / 0 / 0 | 1.023696 |
| lung | Screened (production) | 63 / 0 / 0 | 0.753036 |

Recorded failure: `liver/liver_19/full`, 4.398215 eps.

Recorded failure: `liver/liver_19/full_optimized`, 4.398215 eps.

Recorded failure: `liver/liver_4/full_optimized`, 5.836032 eps.

Recorded failure: `liver/liver_86/full`, 4.229581 eps.

Recorded failure: `liver/liver_86/full_optimized`, 4.181882 eps.
