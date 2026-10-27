# TIAToolbox paired RankSEG benchmarks

Compare standard decoding with **RankSEG 0.0.8** using four pretrained
TIAToolbox 1.6.0 checkpoints on CoNSeP, Kumar, and BCSS. Both methods receive
the same saved model probabilities, with no retraining, probability scaling,
or threshold tuning. Official model inference and subsequent instance
postprocessing are shared by both methods.

This folder contains the inference/evaluation code, small per-case CSVs, and
a [generated results report](RESULTS.md). Rebuilding the report requires only
NumPy; rerunning model inference requires the datasets, weights, TIAToolbox,
and a CUDA GPU.

**TIAToolbox 2.x follow-up:** see the [five-cohort results](RESULTS_V2.md)
and [migration protocol](V2_MIGRATION.md). The detailed
[MicroNet / CoNSeP audit](V2_MICRONET_CONSEP.md) explains the padding difference
behind its lower historical baseline. The table below preserves the 1.6
results; use the 2.x report when discussing the current inference pipeline.

## Results at a glance

Dice is computed for each image or ROI and then averaged. Scores below are
on a 0–1 scale; changes and confidence intervals are in percentage points (pp),
computed from unrounded per-case scores.

| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | Paired 95% bootstrap interval (pp) | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `micronet-consep` / CoNSeP Test | Nucleus Dice | 14 | 0.7376 | 0.7915 | +5.392 | +4.739 to +6.068 | 14/14 |
| `micronet-consep` / Kumar test | Nucleus Dice | 14 | 0.5924 | 0.6472 | +5.485 | +4.493 to +6.535 | 14/14 |
| `hovernet_original-consep` / CoNSeP Test | Nucleus Dice | 14 | 0.8388 | 0.8394 | +0.055 | -0.016 to +0.120 | 9/14 |
| `hovernet_original-kumar` / Kumar test | Nucleus Dice | 14 | 0.8276 | 0.8295 | +0.196 | +0.044 to +0.352 | 11/14 |
| `fcn_resnet50_unet-bcss` / BCSS test | Five-class ROI macro Dice | 45 | 0.5928 | 0.5918 | -0.097 | -0.664 to +0.448 | 23/45 |

MicroNet improved on both tested cohorts. HoVer-Net changed little, and BCSS
showed no overall multiclass gain under the ROI-wise metric. The two CoNSeP
rows share the same 14 images, as do the two Kumar rows; MicroNet/Kumar is a
cross-dataset evaluation. These are small exploratory cohorts, and the
historical baseline scores do not fully reproduce TIAToolbox's published
reference scores. The MicroNet/CoNSeP discrepancy is investigated in the
2.x follow-up above. See [RESULTS.md](RESULTS.md) for per-class BCSS results, secondary
AJI/PQ and multilabel comparisons, and interpretation limits.

## Comparison protocol

- RankSEG uses `metric="dice"`, `solver="RMA"`, `safe_screening=True`, and
  `pruning_prob=0.5` throughout.
- MicroNet and HoVer-Net decode the nucleus probability independently with
  `output_mode="multilabel"`, yielding one foreground mask per image. Both
  methods then use the same official instance postprocessor; HoVer-Net's HV
  and type maps are retained. The endpoint is binary Dice on `inst_map > 0`.
- BCSS uses `output_mode="multiclass"` to produce one five-class label map.
  Raw ground-truth codes 0 and 7 are excluded when scoring both methods.
  Dice is averaged over the five classes within each ROI, then over all
  45 ROIs; `Other` is included. Both-empty masks score 1, and a predicted
  class with no ground-truth pixels scores 0. Ignore masks are used only in
  evaluation, after decoding.
- Confidence intervals resample paired images/ROIs 10,000 times. All cases
  remain in the comparison, including losses. BCSS multilabel results are a
  separate diagnostic because their masks may overlap.

## TIAToolbox 2.x migration check

All five checkpoint/cohort combinations have dedicated 2.1.3 outputs. The
[combined report](RESULTS_V2.md) includes comparisons with the historical
results, nucleus AJI/PQ, all BCSS classes and multilabel diagnostics. The
[migration guide](V2_MIGRATION.md) gives reproduction commands for all cohorts.
Regenerate the full table with NumPy only:

```bash
python -m rankseg_benchmark.tiatoolbox.table_v2 --output rankseg_benchmark/tiatoolbox/RESULTS_V2.md
```

`micronet_consep_v2.py` reruns the 14 CoNSeP test images with **TIAToolbox
2.1.3** and its public `MultiTaskSegmentor` engine. It uses the same registered
MicroNet checkpoint and input files as the 1.6 experiment. The new environment,
probabilities, official instance maps, and scores are kept separately under
`artifacts/tiatoolbox_v2/`.

The [completed report](V2_MICRONET_CONSEP.md) includes all 14 per-image
results, secondary multiclass decoding, and the old/new baseline audit.
Rebuild its main table from tracked evidence without inference:

```bash
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 report --from-evidence
```

This command needs NumPy, SciPy and Pillow, but no TIAToolbox, RankSEG or GPU.

Create an isolated Python 3.12 environment and install
`environments/tiatoolbox-v2.txt`. For development, an editable installation of
the local RankSEG checkout can replace the PyPI installation; the runner
verifies the decoder source hash. Then run from the repository root:

```bash
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 infer --limit 1
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 evaluate --limit 1
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 infer
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 evaluate
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 report
```

The engine uses the registered 252×252 input/output patches and 150-pixel
stride, with full-image coverage and no tissue mask. PNGs receive `mpp=0.25`
metadata to match the registered resolution and keep inference at their
original pixel scale. No physical rescaling or probability normalization is
introduced. Torch compilation is disabled, as in the previous run.

For every image, the evaluator checks that direct MicroNet postprocessing of
the saved probabilities exactly reproduces the engine's instance array, and
that passing the argmax binary mask through the adapter gives that same
array. Both RankSEG variants use this adapter with the unchanged official
postprocessor. Primary Dice uses `tiatoolbox.utils.metrics.dice` on instance
foreground, evaluated per image; all 14 ground-truth masks are nonempty.
AJI/PQ use the existing instance scorer. An `audit` command accepts the
HoVer-Net authors' `metrics/stats_utils.py` through `--reference-scorer` and
compares all three metrics against that implementation.

Migration diagnostics compare the new and old probability maps and replay
the 1.6 baseline. In 2.1.3, `MicroNet.postproc` returns the connected-component
map before hole filling in `predictions`, while its contour information uses
the filled map. The primary evaluation follows the returned instance array.
Hole-filled scores are recorded separately to measure this version change.
The `micronet_padding_audit` module additionally replays both black and white
edge padding with the same model. This optional diagnostic needs the old and
new saved probabilities; the primary 2.x evaluation does not require a 1.6
installation or its inference artifacts.

## Layout

| File | Purpose |
| --- | --- |
| `micronet_consep.py`, `micronet_kumar.py` | MicroNet tile inference, binary decoding, official instance postprocessing, per-image Dice; Kumar also scores AJI/PQ as a secondary check. |
| `micronet_consep_v2.py`, `micronet_padding_audit.py` | TIAToolbox 2.1.3 MicroNet inference, paired evaluation, scorer audit, and controlled padding replay. |
| `cohorts_v2.py`, `table_v2.py` | Remaining 2.1.3 cohorts, optional padding replay, and the combined table from per-image CSVs. |
| `hovernet_consep.py`, `hovernet_kumar.py` | HoVer-Net tile inference, nucleus decoding with unchanged HV/type maps, official instance postprocessing, per-image Dice. |
| `bcss.py` | Download BCSS test ROIs, run official semantic model, score one five-class mask per ROI. |
| `download_weights.py` | Fetch the four TIACentre checkpoints into ignored `artifacts/` paths and verify SHA-256. |
| `provenance.py` | Verify RankSEG 0.0.8 decoder source and record the installed package and runtime versions. |
| `bcss_multilabel.py`, `bcss_other_audit.py` | Optional independent-class and `Other` diagnostics; multilabel output is not a valid single-label BCSS segmentation. |
| `table.py` | Recompute means, changes, win counts, and seeded paired bootstrap intervals from case-level CSVs; generate `RESULTS.md`. |
| `evidence/*.csv` | Small, tracked per-case scores. No images, model weights, or probability maps. |
| `metadata/kumar_test_index.json` | Frozen public index of the 14 Kumar test TIFF/MAT pairs used here. |
| `metadata/*_v2.json` | 2.1.3 runtime, input hashes, checks and migration diagnostics. |

The original images, annotations, checkpoints, probability maps, inference
outputs, and full run JSONs remain under the repository's ignored `artifacts/`
tree. The repository `.gitignore` also excludes checkpoint and image formats.
Do not place raw inputs in `evidence/`.

## Rebuild the table without model inference

From the repository root, in an environment with NumPy installed:

```bash
python -m rankseg_benchmark.tiatoolbox.table --output rankseg_benchmark/tiatoolbox/RESULTS.md
```

This reads the tracked `evidence/*.csv` files and recomputes the paired
statistics. The bootstrap RNG calls match the original evaluators, including
their call order, so the reported intervals are reproducible. To compare the
tracked evidence with the original local run outputs and rebuild directly from
those outputs:

```bash
python -m rankseg_benchmark.tiatoolbox.table --from-artifacts --check-artifacts
```

After a fresh full run, use `--from-artifacts` to inspect its numbers. The
strict `--check-artifacts` option checks byte-for-byte equality with the
frozen evidence and is intended for the original run. If intentionally
updating the evidence, run `--refresh-evidence`, review the CSV diff, and
regenerate `RESULTS.md`.

## Full inference and evaluation

Use a CUDA-capable environment with TIAToolbox 1.6.0, PyTorch, SciPy, Pillow,
NumPy, `requests`, and `gdown`. The recorded runtime used Python 3.10.12,
PyTorch 2.11.0+cu130, torchvision 0.26.0, NumPy 1.26.4, SciPy 1.15.3, and
Pillow 11.3.0. In that environment, install RankSEG and this repository from
the benchmark root:

```bash
python -m pip install -e . "rankseg==0.0.8" gdown
```

The runners accept the installed RankSEG package and check the decoder source
hash before evaluation; a sibling source checkout is not required. They record
the package location, versions, source hash, checkpoint hash, and data manifest
or archive hash. Partial results can only resume with matching provenance.

The measured environment was assembled with newer Torch, SciPy, and Pillow
than TIAToolbox 1.6.0's declared upper bounds. A default dependency-resolved
TIAToolbox installation therefore differs from the recorded runtime; these
runs do not validate all such dependency combinations. Rebuilding the frozen
table from CSVs does not require this inference environment.

These are the expected ignored input paths. Downloaded checkpoints must match
the stated SHA-256 values; the evaluators verify them before inference.

| Input | Ignored path under repository root | SHA-256 / source |
| --- | --- | --- |
| MicroNet CoNSeP weights | `artifacts/tiatoolbox_micronet_consep/micronet-consep.pth` | `f085d591f74cf8de3a7a5a75f08fab3e47859c49bfc7415bc567113dba778279` |
| HoVer-Net CoNSeP weights | `artifacts/tiatoolbox_hovernet_consep/hovernet_original-consep.pth` | `77e5baf98b1a2e4c5991c82bcf08a7c1b955e146f0ba9e655b292e1686cf646e` |
| HoVer-Net Kumar weights | `artifacts/tiatoolbox_hovernet_kumar/hovernet_original-kumar.pth` | `d6f0510a99216e7295a19470c4430cb9840e2ef544b40e5f89a921e5117d674e` |
| FCN BCSS weights | `artifacts/tiatoolbox_bcss/fcn_resnet50_unet-bcss.pth` | `92d6c78e00ca18debfa1cfbb7c976facdce05fb9c0293117fbf4229e94a1aabf` |
| CoNSeP mirror archive | `artifacts/tiatoolbox_micronet_consep/consep_kaggle.zip` | `7f62e3fed9435d5e91568944bca9edd112ff0a67d0940a5300fbd848e5c255dd` |
| BCSS metadata repository | `artifacts/tiatoolbox_bcss/BCSS-source/` | [PathologyDataScience/BCSS](https://github.com/PathologyDataScience/BCSS), commit `3c49d554497dd896cd02ffe382503ef4ce28b1c1`; `meta/roiBounds.csv` SHA-256 `564bc31fdd5808e5d57de40cc65541aad2c1dbc9415e2e0448f25c00ed144d28` |

The four weights come from the [official TIACentre release](https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/tree/main).
Fetch them to their ignored paths and verify their hashes with:

```bash
python -m rankseg_benchmark.tiatoolbox.download_weights
# Or check already downloaded files without network access:
python -m rankseg_benchmark.tiatoolbox.download_weights --verify-only
```

The CoNSeP archive was downloaded from a [Kaggle mirror](https://www.kaggle.com/datasets/karthikperupogu/consep), then extracted so that
`artifacts/tiatoolbox_micronet_consep/CoNSeP/Test/{Images,Labels}` exists.
Its byte identity with the authors' archive has not been verified. The Kumar
test files come from the [HoVer-Net authors' public folder](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK);
`hovernet_kumar.py download` retrieves exactly the 8 `test_same` and 6
`test_diff` image/annotation pairs listed in the frozen index. These split
names are from the Kumar authors, not an in-domain designation for MicroNet.
The compact index SHA-256 is `7338c918d68ed4e91c3b1f2f88282ecda1df7318ad8946984f5b90e39a215026`.
Existing local manifests record the original full author-folder index SHA-256
`d812ccda25a4cf04e961f9fe7a4a6b70e435cb43cb7df6602bfa23395b26ea03`;
both index versions identify the same 28 selected files and URLs.

After placing the Kaggle ZIP at the path in the table, unpack it and obtain
the pinned BCSS metadata source:

```bash
unzip -q artifacts/tiatoolbox_micronet_consep/consep_kaggle.zip -d artifacts/tiatoolbox_micronet_consep
git clone https://github.com/PathologyDataScience/BCSS.git artifacts/tiatoolbox_bcss/BCSS-source
git -C artifacts/tiatoolbox_bcss/BCSS-source checkout 3c49d554497dd896cd02ffe382503ef4ce28b1c1
```

Once inputs are placed as above, run the modules in this order:

```bash
python -m rankseg_benchmark.tiatoolbox.micronet_consep infer
python -m rankseg_benchmark.tiatoolbox.micronet_consep evaluate
python -m rankseg_benchmark.tiatoolbox.hovernet_consep infer
python -m rankseg_benchmark.tiatoolbox.hovernet_consep evaluate

python -m rankseg_benchmark.tiatoolbox.hovernet_kumar download
python -m rankseg_benchmark.tiatoolbox.hovernet_kumar infer
python -m rankseg_benchmark.tiatoolbox.hovernet_kumar evaluate
python -m rankseg_benchmark.tiatoolbox.micronet_kumar infer
python -m rankseg_benchmark.tiatoolbox.micronet_kumar evaluate
python -m rankseg_benchmark.tiatoolbox.micronet_kumar instances

python -m rankseg_benchmark.tiatoolbox.bcss download
python -m rankseg_benchmark.tiatoolbox.bcss infer
python -m rankseg_benchmark.tiatoolbox.bcss evaluate
python -m rankseg_benchmark.tiatoolbox.bcss_multilabel
python -m rankseg_benchmark.tiatoolbox.bcss_other_audit
python -m rankseg_benchmark.tiatoolbox.table --from-artifacts
```

The `infer` steps save model outputs once; both decoders read the same maps.
The `evaluate` steps write `results.json` and `per_case.csv` within their
respective ignored artifact folders. Nucleus Dice is calculated on each image
after identical official MicroNet or HoVer-Net postprocessing, then averaged.
BCSS maps the annotated classes into five categories, ignores raw codes 0 and
7 for both decoders, computes each class Dice per ROI, then averages classes
and ROIs. The exact evaluation protocol behind TIAToolbox's published BCSS
score has not been established; see [RESULTS.md](RESULTS.md) for this and other
interpretation limits.
