# MONAI external evaluation


The workflow has two stages: **generate probabilities once**, then **benchmark
decoders on the same cache**. A complete 3D volume remains one optimization
unit; it is not split into independent 2D predictions.

### 1. Prepare the environment and data

From the cloned repository:

```bash
python -m pip install -e ".[monai]"
rankseg-monai-cache --list
```

Obtain MSD Task07 Pancreas or Task09 Spleen under its original license. Pass
an extracted directory containing `imagesTr/` and `labelsTr/`; the generator
does not download or redistribute the medical datasets.

**Checkpoints, downloaded Bundles, datasets, and probability caches stay
local and are excluded by `.gitignore`.** Git contains code, documentation, and the
[pinned metadata and evaluation configuration](../../rankseg_benchmark/monai/monai_specs.json).
`--download-bundle` obtains the specified Bundle and checkpoint from the MONAI
Model Zoo.

### 2. Generate the Pancreas cache

```bash
rankseg-monai-cache \
    --benchmark monai_btcv_swin_v058_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --bundle-dir ./monai_bundles \
    --output-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --download-bundle --device cuda
```

This generates the fixed **20-volume** cohort. Add `--limit 1` for a one-case
smoke run; a partial cache does not reproduce the full-cohort result or demo.

### 3. Evaluate the cached volumes

```bash
rankseg-bench \
    --dataset monai_btcv_swin_v058_msd_pancreas \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --solver RMA --metric dice --device cuda --batch-size 1 --warmup 1 \
    --json results_monai_pancreas.json
```

Use `--metric iou` to run the IoU objective. The report scores only foreground
and measures decoder-only latency.

<details>
<summary>MSD Spleen commands (9 volumes)</summary>

```bash
rankseg-monai-cache \
    --benchmark monai_btcv_swin_v058_msd_spleen \
    --dataset-root /data/Task09_Spleen \
    --bundle-dir ./monai_bundles \
    --output-dir ./artifacts/monai_btcv_swin_v058_msd_spleen \
    --download-bundle --device cuda

rankseg-bench \
    --dataset monai_btcv_swin_v058_msd_spleen \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_spleen \
    --solver RMA --metric dice --device cuda --batch-size 1 --warmup 1 \
    --json results_monai_spleen.json
```

</details>

### Checkpoint provenance

Both targets use **BTCV Swin UNETR 0.5.8**, not a checkpoint trained on the
same MSD task. The official
[Bundle documentation](https://github.com/Project-MONAI/model-zoo/blob/dev/models/swin_unetr_btcv_segmentation/docs/README.md)
identifies BTCV as the supervised task. MONAI's
[pretraining data loader](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/Pretrain/utils/data_utils.py)
lists LUNA16, TCIA COVID-19, HNSCC, TCIA Colonography, and LIDC-IDRI as
self-supervised sources; MSD Task07 and Task09 are not listed.

Under this declared-dataset provenance, the fixed MSD cases are external
tests: they are not used for checkpoint training or selection, post-processing
parameter selection, or threshold fitting. Each built-in target validates
`checkpoint_exposure=none_by_declared_dataset_provenance`, false test-case
usage flags, and a test dataset ID distinct from every declared supervised
and self-supervised source ID. These declarations are copied into each manifest.

This is an auditable **dataset-level safeguard**, not an independent
patient-identity audit beyond upstream metadata.

<details>
<summary>Preprocessing, foreground mapping, and artifact checks</summary>

The 14-channel BTCV softmax is collapsed to
`[1 - p(target), p(target)]`, preserving the target-versus-rest distribution.

| Target | Model channel | MSD foreground labels |
| :--- | ---: | :--- |
| Pancreas | 11 | 1 and 2: pancreas parenchyma + tumor |
| Spleen | 1 | 1: spleen |

The generator uses RAS orientation, 1.5 × 1.5 × 2.0 mm spacing, the Bundle's
intensity window, a 96³ sliding window, 0.5 overlap, and AMP inference.
`sw_batch_size=1` replaces the deployment setting of 4 to fit 24 GiB GPUs;
this can affect inference throughput and small floating-point reduction
details, but inference is outside decoder timing.

Probabilities are stored as float32, with labels aligned in the deterministic
resampled model space used by the official MONAI RankSEG tutorial. Each case
is stored separately and its SHA-256 is verified when read. Manifests record:

- Bundle name, version, official checkpoint checksum, and generated SHA-256;
- the fixed case list and checkpoint-exposure declarations;
- preprocessing, inference, activation, and coordinate-space settings;
- software versions and source image/label hashes.

</details>

<details>
<summary>Reproduce the README medical demo</summary>

The figure has three large panels: ground truth, Argmax, and RankSEG. Its
Dice scores describe the selected slice; the CLI separately prints the
20-volume mean for the caption. Both themes use transparent backgrounds.

Use the **complete Pancreas cache**:

```bash
python -m pip install -e ".[monai,demo]"

rankseg-monai-demo \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --output ./monai_pancreas_rankseg.png \
    --theme light --device cuda

rankseg-monai-demo \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --output ./monai_pancreas_rankseg_dark.png \
    --theme dark --device cuda
```

After aggregate evaluation, the renderer selects the axial slice that
maximizes corrected minus introduced error pixels among slices with at least
400 foreground pixels, using deterministic tie-breakers. This selection is
for presentation only; it does not change the checkpoint, post-processing
parameters, or the reported cohort result.

</details>
