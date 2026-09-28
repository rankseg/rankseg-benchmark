# License and migration attribution

The original rankseg-benchmark code is BSD-3-Clause (see LICENSE).

The following material was imported from rankseg-nnunet-benchmark, commit
`5db223fc2c8790508461af68ab73725ff986a6f2` plus the local screening additions
present on 2026-09-25, and retains Apache-2.0 (LICENSES/Apache-2.0.txt):

- `rankseg_benchmark/nnunet/` (and legacy `rankseg_nnunet_bench/` aliases);
- `configs/nnunet/`, `registry/nnunet/`, `evidence/nnunet/`, `results/nnunet/`;
- `scripts/nnunet/`, `tests/nnunet/`, `docs/nnunet/`, `docs/screening/nnunet/`;
- `environments/nnunet/`.

Migration changes relocate modules, imports, manifests, scripts, documentation,
and workspace resolution; CLI imports are lazy. Scientific metric/aggregation
definitions and published evidence bytes are unchanged. New migration support
within the nnU-Net directories follows their Apache-2.0 license.

The source repository is retained, not deleted or archived by this migration.
No datasets, checkpoints, or probability arrays are redistributed here.
