# Complete-cohort screening benchmark

This engineering experiment compares decoding paths on four complete nnU-Net OOF
cohorts. It does not retrain a model, select new checkpoints, generate new probabilities,
or replace the published Full-16 evaluation. The current results document is
[SCREENING_RESULTS_2026-09-20.md](SCREENING_RESULTS_2026-09-20.md).

The subsequent frozen-code final acceptance, including argmax and normal
production dispatch, is tracked separately in
[SCREENING_FINAL_ACCEPTANCE_2026-09-22.md](SCREENING_FINAL_ACCEPTANCE_2026-09-22.md).
Check its status before treating it as a completed result. The forced-screening
protocol below describes the earlier experiment, not the new production-dispatch run.

## Fixed setup

- Argmax; original full sort; optimized direct-argmax full sort; fused screening.
- Dice / RMA / multiclass, smooth=0, pruning_prob=0.5, max_score.
- All native channels, including background; original channel-label mappings.
- Whole 3D volumes, not slices; identical fixed official postprocessing, exactly once.
- Screening is forced regardless of the small-input cutoffs; workspace fallbacks remain.
  Optimized full forcibly bypasses screening. This is not automatic dispatch timing.
- RTX 3090, float32, PyTorch 2.8.0+cu128, Triton 3.4.0 for the recorded setup.
- Two warmups, five synchronized timed repeats in rotating order. Report the mean
  of per-volume median decoding times; exclude inference, I/O, transfers, compilation,
  postprocessing, metric evaluation and the numerical oracle.
- For each foreground label, average Dice/IoU over cases, then average labels.
  Both-empty labels are undefined. Dice is optimized; IoU is an additional outcome.

## Complete cohorts

| Dataset | Volumes | Manifest under `configs/` |
| :--- | ---: | :--- |
| Pancreas | 281 | `Task007_Pancreas_ensemble_oof.yaml` |
| HepaticVessel | 303 | `Task008_HepaticVessel_ensemble_oof.yaml` |
| Liver | 131 | `Task003_Liver_ensemble_oof.yaml` |
| Lung | 63 | `Task006_Lung_ensemble_oof.yaml` |

Use `--cases-per-fold 0` for every dataset: **778 volumes** in total. These are
targeted replications of known-positive datasets, not an unbiased new multi-dataset
evaluation. No outcome-based case selection is allowed.

From this repository, with a local RankSEG checkout containing experimental screening:

```bash
PYTHONPATH=src env/bin/python -m rankseg_nnunet_bench.screening_benchmark \
  --manifest configs/Task003_Liver_ensemble_oof.yaml \
  --rankseg-path ../rankseg --cases-per-fold 0 --warmup 2 --repeats 5 \
  --objective-failure-policy record \
  --output outputs/screening-complete-2026-09-21/liver.json
```

Use a distinct output filename for each dataset. Existing reports are never
overwritten. Run datasets serially: the controls temporarily change private dispatch
settings and restore them in `finally`. Do not run competing GPU jobs during timings.

## Numerical checks and independent audit

The default objective-failure policy is `stop`. The explicit `record` policy above
continues only after a finite regret exceeds the unchanged **four-float32-eps**
budget. It records the failure and checks subsequent methods/cases. Structural
invariants, nonfinite regrets and unexpected execution errors still stop the run.
No algorithm or threshold is changed. `complete=true` means all selected cases were
processed, not that every numerical check passed.

Audit a completed report against original per-case counts and published metrics:

```bash
PYTHONPATH=src env/bin/python -m rankseg_nnunet_bench.screening_audit \
  outputs/screening-complete-2026-09-21/liver.json \
  --reference-csv outputs/Task003_Liver_ensemble_oof/case_label_metrics.csv \
  --published-summary evidence/datasets/Task003_Liver/summary.json \
  --methods argmax full_optimized screened --report-objective-failures
```

Without `--report-objective-failures`, failed numerical checks are rejected.
With it, complete measurements can be independently aggregated, but numerical
failures produce `audit=failed_objective`, never `passed`. Every failure is listed,
including failures in omitted controls. The flag does not allow missing cases,
invalid counts, incomplete requested methods, oracle OOM or inconsistent summaries.

The original full sort may OOM on large volumes. The explicit three-method audit
above still requires every case for argmax, optimized full and screened, and reports
the original control's OOM count separately. Use the audit's full-cohort aggregates,
not the benchmark's four-way paired-complete summary. Never silently retry on CPU
or exclude a difficult case from a full-cohort claim.

## Untimed screening proportions

Do not change frozen benchmark sources or run competing CUDA work during timings.
After the complete benchmark sequence finishes, run the supplementary collector:

```bash
PYTHONPATH=src env/bin/python scripts/collect_screening_proportions.py \
  --benchmark-report outputs/screening-complete-2026-09-21/liver.json \
  --rankseg-path ../rankseg \
  --output outputs/screening-complete-2026-09-21/screening-proportions/liver.json
```

Repeat for the other three datasets. The collector verifies the full cohort and
matching sources, environment, probability hashes and restored shapes. It calls the
same CUDA certificate backend but does not rerun sorting, inference, Dice/IoU or the
float64 objective oracle. It never overwrites the benchmark report or an existing
statistics file.

Each case/class records H (forced positive), L (forced negative), M (undecided),
class-pruned entries, fallback rows and actual real sort entries. Case-level padded
sort workspace is separate from real candidates. Summary ratios use summed counts:

| Field | Definition |
| :--- | :--- |
| `screening_fraction_active` | `(H + L) / active_entries`, before workspace fallback |
| `effective_sort_avoidance_active` | `1 - actual_sort_entries / active_entries`, counting a fallback row as a full sort |
| `class_pruned_fraction_all` | `class_pruned_entries / total_entries` |
| `actual_sort_fraction_all` | `actual_sort_entries / total_entries`, including the separate effect of class pruning |

Entries mean class probabilities at voxels, not unique spatial voxels. Pruned classes
are outside the H/L/M partition; an all-pruned denominator is `null`, not 100% pixel
screening. `by_label` keeps background and each foreground label separate.
Screened-negative entries remain available to final multiclass assignment.

## Provenance and preserved evidence

Audit all five held-out folds against the declared nnU-Net v1 default assignments
and recorded inference commands. For ensembles, both components must use the same
held-out fold. Recorded commands are parsed, never executed; this is a cache-lineage
audit, not independent proof of training execution or patient identity.

Source, manifest, probability and label hashes are recorded. Original channel
mappings, crop restoration and release-corrected labels are retained. Moved legacy
symlinks are resolved read-only within this checkout, never rewritten. Liver's raw
exports under `fold_*/not_postprocessed/` are supported; duplicates and missing
cases are rejected.

No checkpoints or probability arrays are added to Git. Prior subset/natural-image
measurements remain in `outputs/screening-engineering/`; prior complete and failed
attempts remain in `outputs/screening-full-cohorts*/`. Reserved MONAI cache directories
are unchanged and are not included in this experiment. The earlier CUDA indexing
fix and its tests are described in the
[indexing regression note](SCREENING_INDEX_FIX_2026-09-20.md).
