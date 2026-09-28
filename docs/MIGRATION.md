# Unified benchmark layout

Quick, MONAI, and nnU-Net now live under one package. This is a repository
reorganization, not a new experiment or a change to RankSEG's decoder.
See the [migration acceptance results](MIGRATION_CHECKS_2026-09-25.md) for validation.

| Previous entry | Current entry | Compatibility |
| --- | --- | --- |
| `rankseg-bench --dataset ...` | `rankseg-bench quick --dataset ...` | Flag-only syntax still lists/runs all old targets |
| `rankseg-monai-cache ...` | `rankseg-bench monai cache ...` | Both supported |
| `rankseg-monai-demo ...` | `rankseg-bench monai demo ...` | Both supported |
| MONAI through `rankseg-bench --dataset ...` | `rankseg-bench monai evaluate --dataset ...` | Both supported |
| `rankseg-nnunet-bench ...` | `rankseg-bench nnunet ...` | Both supported |
| `rankseg_nnunet_bench.*` | `rankseg_benchmark.nnunet.*` | Legacy imports alias canonical modules |

Install this merged checkout with `python -m pip install -e .`. The optional
`.[monai]`, `.[demo]`, `.[nnunet]`, and `.[nnunet-v2]` extras remain separate;
Quick does not import or require the medical inference packages. Use separate
virtual environments when reproducing different historical software stacks.
Do not co-install the old standalone nnU-Net benchmark distribution: both
provide the legacy namespace and executable. Leave the old environment intact,
or use a fresh environment for this checkout.

## Existing large nnU-Net workspace

The old repository, its virtual environment, images, checkpoints, probabilities,
and outputs were **not moved or deleted**. To reuse its local data from the
merged checkout, set the original workspace location explicitly:

```bash
export RANKSEG_NNUNET_WORKSPACE=/absolute/path/to/rankseg-nnunet-benchmark
rankseg-bench nnunet evaluate configs/nnunet/Task003_Liver_ensemble_oof.yaml --device cuda
```

Only built-in `configs/nnunet/` manifests redirect their relative `work/`,
`outputs/`, `artifacts/`, and `data/` paths. Absolute input paths and custom manifests
keep their original meaning. Without the variable, these directories are
relative to the merged repository. No automatic inference or data copying is
performed by setting this variable. Evaluation writes to the configured output
directory; use `--output-dir` when preserving a previous run.

Run pipeline scripts as `bash scripts/nnunet/run_task003_pipeline.sh` in an
activated, compatible nnU-Net environment. `RANKSEG_NNUNET_PYTHON` can select
its interpreter; nnU-Net executables must also be on PATH. Historical reference
requirements are preserved in `environments/nnunet/`; their RankSEG 0.0.5 pin is
intentional and does not describe the newly published 0.0.7 benchmark results.

## Evidence and history

`evidence/nnunet/` is a byte-preserved copy of the published Full-16 evidence,
including its original README and manifest. Commands embedded inside that
snapshot describe the old layout; the current verification command is:

```bash
rankseg-bench nnunet verify-evidence evidence/nnunet
```

Verification checks every checksum and rebuilds the aggregate in a temporary
directory, requiring byte-identical results. Old paths/source hashes in evidence
and dated screening reports are historical provenance, not newly generated runs.
Current scripts are under `scripts/nnunet/`; dated kernel-comparison scripts
still require their recorded snapshots and local caches.

Before migration, both repositories contained uncommitted experiment files.
They were included from the working trees, not silently dropped by importing
only committed files. The old nnU-Net repository remains untouched. No commit,
push, repository archival, or large-data migration is performed by this change.
