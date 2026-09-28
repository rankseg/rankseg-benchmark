# Screening engineering experiments

These reports compare decoder implementations, runtime, memory, and screening
proportions. They are separate from the three suites' primary argmax-versus-RankSEG
quality results. Historical dates, cohorts, versions, and source hashes remain
attached to each report. Auto-default reports are historical experiments:
**RankSEG 0.0.7 defaults to `safe_screening=False`.**

- Quick: [acceptance and real-cache results](quick/SCREENING_AUTO_DEFAULT_ACCEPTANCE_2026-09-25.md),
  [CPU scale study](quick/SCREENING_REAL_SCALES_2026-09-25.md),
  [CUDA scale study](quick/SCREENING_CUDA_REAL_SCALES_2026-09-25.md).
- nnU-Net: [benchmark guide](nnunet/SCREENING_BENCHMARK.md),
  [complete cohorts](nnunet/SCREENING_RESULTS_2026-09-20.md),
  [representative volumes and large Liver regression](nnunet/SCREENING_AUTO_DEFAULT_ACCEPTANCE_2026-09-25.md).

Quick drivers remain in `scripts/`; medical drivers are in `scripts/nnunet/`.
See [migration notes](../MIGRATION.md) for local workspace setup. Recorded
commands may reference old paths or frozen, unversioned snapshots; migration
does not rerun or relabel those experiments.
