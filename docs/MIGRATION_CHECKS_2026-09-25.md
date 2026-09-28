# Repository reorganization acceptance

The unified checkout keeps Quick, MONAI, and nnU-Net under
`rankseg_benchmark/`. No RankSEG core source, checkpoint, training split,
scientific metric, or previously published result was changed.

## Tests

Tests used Python 3.10.12, with the published RankSEG 0.0.7 package explicitly
selected for GPU and extracted-sdist runs. The nnU-Net environment uses Torch
2.8/Triton 3.4; Quick/MONAI uses Torch 2.11/Triton 3.6. GPU: RTX 3090.

| Suite | CPU, extracted sdist | CUDA, merged checkout |
| --- | --- | --- |
| Quick, MONAI helpers, common entry points | 238 passed, 19 skipped | 257 passed |
| nnU-Net | 195 passed, 64 skipped | 259 passed |

CPU skips are accelerator-dependent tests. Counts include migration regressions
for namespace aliases, optional-import isolation, suite dispatch, packaged
MONAI resources, and workspace path handling. They are not dataset counts.

## Compatibility and evidence

- All **25** dataset manifests preserve their settings and resolve to the same
  inputs when `RANKSEG_NNUNET_WORKSPACE` selects the original workspace. Relative
  textual provenance links were adjusted for the new manifest directory depth.
- All **41** files in `evidence/nnunet/` are byte-identical to the original
  evidence, retaining **16 datasets / 2,181 cases**.
- `verify-evidence` passed through both the unified and legacy CLIs; rebuilding
  the Full-16 aggregate produced byte-identical outputs.
- Seven nnU-Net numerical/evaluation modules (metrics, aggregate, evaluation,
  decoders, postprocessing, OOF, IO) retain identical ASTs. Quick metrics, runner,
  datasets, and timing retain identical ASTs after normalizing import relocation.
- Both distributions build and pass strict Twine checks. An isolated wheel
  installation outside the checkout loads all four console entry points and
  the MONAI resource; nnU-Net evidence and external-checkout workspace resolution
  also pass. Sdist tests ran outside both original repositories.
- Package inspection confirms no checkpoints, images, probability arrays,
  local outputs, environments, or `.git` directories are included. Both license
  texts and the migration attribution ship with the wheel.
- Documentation relative links, workflow YAML, shell syntax, undefined-name
  checks for package/tests, and `git diff --check` pass. Remote CI was not run.

## Scope and preserved work

No inference, checkpoint download, full dataset benchmark, commit, push, or
repository archival was performed. Large workspace directories remain where
they were. The old nnU-Net repository, including its untracked additions, was
checked against a pre-migration snapshot and remains unchanged. The RankSEG
core worktree remains clean.

Small-file backups and validation artifacts are in the local temporary directory
`/tmp/rankseg-benchmark-reorg.UDabd0/`; selected build/test logs are also retained
under the Git-ignored `.cache/reorganization-2026-09-25/` directory.
