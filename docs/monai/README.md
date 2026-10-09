# Exploratory MONAI cross-dataset workflow

This page preserves RankSEG-run cross-dataset experiments using checkpoints
published in the MONAI Model Zoo. They are **not MONAI-published benchmark
results** and are not part of the showcased matched-dataset comparison. For
that comparison, see the [matched Bundle results](MATCHED_MODEL_ZOO_RESULTS_2026-10-09.md).

The external-cohort workflow has two stages: **generate probabilities once**, then **benchmark
decoders on the same cache**. A complete 3D volume remains one optimization
unit; it is not split into independent 2D predictions.

The [expanded MONAI results](EXPANDED_RESULTS_2026-10-07.md) cover a second CT
checkpoint and a multiclass cardiac MRI checkpoint. For the original BTCV
checkpoint's paired full-sort versus safe-screening experiment, see the
[BTCV screening results](SCREENING_RESULTS_2026-10-07.md).
The [AMOS22 small-organ study](SMALL_ORGANS_AMOS22.md) evaluates adrenal
glands and gallbladder alongside larger organs on the same frozen BTCV
checkpoint and labeled CT validation cases.
The [FLARE22 replication](FLARE22_BTCV_RESULTS_2026-10-08.md) applies the
same frozen checkpoint to all 50 public labeled CT cases, with 46 images
lacking a byte-identical prior MSD match as its primary analysis. Together,
these studies show
why low baseline Dice is a more useful descriptive signal than organ size
alone, while retaining classes and cases that worsen.

### 1. Prepare the environment and data

From the cloned repository:

```bash
python -m pip install -e ".[monai]" "rankseg==0.0.7"
rankseg-monai-cache --list
```

Obtain MSD Task07 Pancreas, MSD Task09 Spleen, or the ACDC official training
database under their original terms. Pass an extracted directory containing
`imagesTr/` and `labelsTr/`; the generator does not download or redistribute
the medical datasets. For ACDC, use the nnU-Net-style Task027 directory with
`patientNNN_frameNN_0000.nii.gz` images and matching unsuffixed labels.

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

For a controlled safe-screening comparison on the **same cache**, repeat the
RMA Dice command with `--safe-screening`. This explicitly forces screening;
it is not the RankSEG default and may be slower on some shapes. The runner
rejects this flag for solvers or objectives other than RMA Dice.

For comparable decoder latency, use the paired timer instead of comparing
single-pass CLI times: it warms each volume shape separately, then rotates
Argmax, full-sort RMA, and screened RMA over repeated CUDA calls. It also
measures incremental peak allocated GPU memory and counts pixels changed by
screening relative to full sort:

```bash
python -m rankseg_benchmark.monai.paired \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --output ./artifacts/monai_btcv_swin_v058_msd_pancreas/paired.json \
    --warmup 2 --repeats 9
```

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

<details>
<summary>Additional MONAI checkpoints: Whole Body CT and cardiac MRI</summary>

The 3 mm Whole Body model is evaluated on the **same fixed MSD cases** as BTCV:

```bash
rankseg-monai-cache \
    --benchmark monai_wholebody_lowres_v027_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --bundle-dir ./monai_bundles \
    --output-dir ./artifacts/monai_wholebody_lowres_v027_msd_pancreas \
    --download-bundle --device cuda

rankseg-bench \
    --dataset monai_wholebody_lowres_v027_msd_pancreas \
    --artifact-dir ./artifacts/monai_wholebody_lowres_v027_msd_pancreas \
    --solver RMA --metric dice --device cuda --batch-size 1 \
    --json results_wholebody_pancreas.json
```

Replace `pancreas` with `spleen` and use `/data/Task09_Spleen` to run the
9-volume Whole Body Spleen target. For the multiclass cardiac target:

```bash
rankseg-monai-cache \
    --benchmark monai_ventricular_v035_acdc \
    --dataset-root /data/Task027_ACDC \
    --bundle-dir ./monai_bundles \
    --output-dir ./artifacts/monai_ventricular_v035_acdc \
    --download-bundle --device cuda

rankseg-bench \
    --dataset monai_ventricular_v035_acdc \
    --artifact-dir ./artifacts/monai_ventricular_v035_acdc \
    --solver RMA --metric dice --device cuda --batch-size 1 \
    --per-class --json results_acdc.json
```

`--limit 1` is available for inference smoke checks; only a complete manifest
reproduces the full-cohort results. The cardiac Bundle segments slices in 2D,
but RankSEG evaluates each assembled 3D frame as one volume; both labeled
frames from every one of the 100 ACDC patients are included.

</details>

### Checkpoint provenance

The original two targets use **BTCV Swin UNETR 0.5.8**, not a checkpoint
trained on the same MSD task. The official
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
patient-identity audit beyond upstream metadata. The added Whole Body 0.2.7
checkpoint declares TotalSegmentator training, while the Ventricular 0.3.5
checkpoint declares private King's College London data; these are also
distinct by declared dataset from the fixed MSD and ACDC targets,
respectively. See the [expanded results](EXPANDED_RESULTS_2026-10-07.md)
for limitations and the actual paired scores. We do not claim that the
different checkpoints are directly comparable as models.

<details>
<summary>BTCV preprocessing, foreground mapping, and artifact checks</summary>

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
<summary>Additional Bundle mappings and preprocessing</summary>

Whole Body low resolution uses 3 mm isotropic spacing, RAS orientation,
nonzero intensity normalization, and scaling to [−1, 1]. Its 105-channel
softmax is collapsed to `[1-p(target), p(target)]` using model channel 10
for Pancreas or 1 for Spleen. The MSD Pancreas foreground combines source
labels 1 and 2, as in the BTCV comparison.

The Ventricular Bundle uses 2D `SliceInferer` with a 256 × 256 ROI and scales
each native-grid ACDC volume's image intensities to [0, 1]. Its model labels
are 0 background, 1 LV pool, 2 myocardium, and 3 RV pool. ACDC's source
labels are 0 background, 1 RV, 2 myocardium, and 3 LV cavity, so the generator
maps source labels `0→0, 1→3, 2→2, 3→1` before evaluation. It retains all
four probability channels and scores the three foreground classes.

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
