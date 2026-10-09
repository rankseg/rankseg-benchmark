# RankSEG × nnU-Net benchmark

This repository evaluates one focused question: **what changes when
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)'s usual voxel-wise argmax is replaced by
[RankSEG](https://github.com/rankseg/rankseg) at inference time?** The comparison uses the same exported model outputs
for both decoders and does not change training, checkpoints, preprocessing, or test data.

The public evidence package contains 16 datasets and 2,181 labeled cases. It is small enough to clone, checksummed,
and independently rebuildable without downloading images, model weights, or probability arrays.

> Commands below run from the **rankseg-benchmark repository root**. This page
> distinguishes the 0.0.7 screened rerun from the unchanged historical
> checksummed Full-16 evidence.
> See [migration and workspace setup](../MIGRATION.md) before reproducing inference.

A separate [TotalSegmentator v2 / AMOS22 fixed 10-case pilot](AMOS22_TOTAL_SEGMENTATOR_V2_PILOT.md)
uses RankSEG 0.0.7 and six binary organ endpoints. It is **not** part of the
Full-16 evidence or its dataset-level aggregate.

## Experiment design

### Comparison

- **Baseline:** voxel-wise argmax.
- **Alternative:** RankSEG with its public default multiclass Dice configuration.
- **Input:** the identical post-softmax nnU-Net array for both decoders.
- **Postprocessing:** the configuration selected by the official model release, applied identically to both outputs.
- **Tuning:** no dataset-specific RankSEG tuning and no calibration fitted on evaluation cases.

Most datasets use official nnU-Net v1 checkpoints because they provide a broad, checksummed model release. A
TotalSegmentator checkpoint supplies an additional nnU-Net v2 inference-path replication.

### Cohort

Full-16 contains:

- a Core-12 cohort evaluated with strict five-fold out-of-fold (OOF) prediction from official nnU-Net v1 models;
- PROMISE12, evaluated on 30 independent former-hidden-test cases;
- CHAOS MRI, evaluated out of fold with uncertainty clustered by patient;
- Fluo-C3DH-A549, evaluated out of fold with uncertainty clustered by sequence;
- CHAOS CT, evaluated on 20 external cases with a TotalSegmentator nnU-Net v2 checkpoint.

The datasets span CT, MRI, cardiac cine MRI, tumour and organ segmentation, vessels, nested BraTS evaluation regions,
and microscopy. Exactly one model configuration was selected per dataset before the cross-dataset analysis. Every
eligible result was retained, including regressions.

### Outcomes

The primary outcome is the change in foreground macro mean Dice:

```text
Δ Dice = Dice(RankSEG) - Dice(argmax)
```

Each dataset is one unit in the Full-16 analysis, regardless of its number of cases. The benchmark also reports
paired bootstrap intervals, case-level changes, tail behavior, decoder runtime, and recorded peak memory. Repeated
observations from the same patient or sequence are clustered where required.

The complete rules are in the [fixed benchmark protocol](BENCHMARK_PROTOCOL.md).

## RankSEG 0.0.7 + safe screening rerun

The complete Full-16 cohort was rerun with screening **explicitly enabled** on
every case. Dataset-macro Dice was **83.63646% Argmax → 83.84726% RankSEG**
(**+0.21080 pp**), and dataset-macro IoU improved by **+0.19943 pp**. Thirteen
datasets improved in Dice and three declined. For the full 16-row table,
case-level safety counts, numerical differences, and timing interpretation,
see the [0.0.7 screened rerun report](FULL16_RANKSEG_0_0_7_SCREENED_2026-10-09.md).

The low-baseline endpoints are easier to see at the class level than in a
dataset average: Colon tumour starts at **48.92%** Argmax Dice and Pancreas
tumour at **54.65%**. An exploratory grouping of the 31 class/region records by
Argmax Dice finds that all 5 high-difficulty records (<70%) improve, whereas
the medium group (70% to <90%) has 7 gains and 3 declines and a slightly
negative mean change. The [rerun report](FULL16_RANKSEG_0_0_7_SCREENED_2026-10-09.md)
shows every record; this grouping is not a significance test or a new selected
aggregate.

The detailed 0.0.7 outputs are currently local, Git-ignored artifacts. The
public checksummed evidence below remains the historical 0.0.5 run; its
numbers are not a controlled estimate of screening's effect relative to 0.0.7.

## Historical published evidence (RankSEG 0.0.5, no screening)

- **16 datasets and 2,181 cases**
- **13 positive and 3 negative** dataset-level Dice changes
- Macro-average Dice: **83.63646% argmax** vs **83.84318% RankSEG**
- Dataset-level mean change: **+0.20671 pp**; median change: **+0.00979 pp**
- Exact two-sided sign test: **p=0.02127**; Wilcoxon signed-rank test: **p=0.01099**
- At a ±1 Dice pp case threshold: **246 improved, 1,802 stable, 133 worsened**

The table reports **each foreground class or evaluated region separately**;
`Valid/total` is the number of cases with defined Dice for that row. There is
no within-dataset average in the table. These class-level rows come from the
retained original local outputs; each output's dataset macro matches the
checksummed public evidence. The evidence package itself does not contain the
per-class CSVs.

| Dataset | Class / evaluated region | Valid/total | Argmax Dice (%) | 0.0.5 RankSEG Dice (%) | Δ Dice (pp) | Δ IoU (pp) |
|---|---|---:|---:|---:|---:|---:|
| MSD BrainTumour | whole tumour | 484/484 | 91.00015 | 90.91972 | -0.08043 | -0.13839 |
| MSD BrainTumour | tumour core | 484/484 | 83.37486 | 83.41289 | +0.03803 | -0.02225 |
| MSD BrainTumour | enhancing tumour | 478/484 | 79.34149 | 79.40703 | +0.06554 | +0.04073 |
| MSD Heart | left atrium | 20/20 | 93.29433 | 93.29623 | +0.00191 | +0.00293 |
| MSD Liver | liver | 131/131 | 96.21178 | 96.21580 | +0.00403 | +0.00595 |
| MSD Liver | liver tumour | 120/131 | 65.78628 | 67.43048 | +1.64420 | +1.57421 |
| MSD Hippocampus | Anterior | 260/260 | 89.74944 | 89.75282 | +0.00338 | +0.00583 |
| MSD Hippocampus | Posterior | 260/260 | 88.06751 | 88.07223 | +0.00473 | +0.00756 |
| MSD Prostate | peripheral zone | 32/32 | 66.13404 | 67.89496 | +1.76092 | +2.11025 |
| MSD Prostate | transition zone | 32/32 | 85.78014 | 84.06512 | -1.71502 | -1.02269 |
| MSD Lung | lung tumour | 63/63 | 72.28822 | 72.74899 | +0.46077 | +0.56961 |
| MSD Pancreas | pancreas | 281/281 | 82.14158 | 81.91599 | -0.22559 | -0.32636 |
| MSD Pancreas | pancreas tumour | 281/281 | 54.65037 | 57.20974 | +2.55938 | +1.98728 |
| MSD HepaticVessel | hepatic vessel | 303/303 | 64.72835 | 65.04924 | +0.32089 | +0.33037 |
| MSD HepaticVessel | liver tumour | 303/303 | 72.53273 | 73.28147 | +0.74874 | +0.60884 |
| MSD Spleen | spleen | 41/41 | 97.23659 | 97.24320 | +0.00661 | +0.01095 |
| MSD Colon | colon cancer primary | 126/126 | 48.91759 | 49.15961 | +0.24203 | -0.12520 |
| PROMISE12 | prostate | 30/30 | 91.93567 | 91.85350 | -0.08217 | -0.13970 |
| ACDC | right ventricle | 200/200 | 91.62896 | 91.37808 | -0.25088 | -0.33276 |
| ACDC | myocardium | 200/200 | 90.82449 | 90.85268 | +0.02819 | +0.04441 |
| ACDC | left ventricle cavity | 200/200 | 94.90798 | 94.90948 | +0.00150 | -0.01437 |
| CHAOS MRI | liver | 60/60 | 93.45059 | 93.52053 | +0.06994 | +0.11630 |
| CHAOS MRI | right kidney | 60/60 | 92.91439 | 93.01772 | +0.10333 | +0.10175 |
| CHAOS MRI | left kidney | 60/60 | 92.38598 | 92.74302 | +0.35704 | +0.52661 |
| CHAOS MRI | spleen | 60/60 | 89.42461 | 89.59833 | +0.17372 | +0.27372 |
| SegTHOR | esophagus | 40/40 | 85.49781 | 85.47971 | -0.01811 | -0.01905 |
| SegTHOR | heart | 40/40 | 95.33305 | 95.33070 | -0.00235 | -0.00392 |
| SegTHOR | trachea | 40/40 | 90.87343 | 90.88136 | +0.00793 | +0.00929 |
| SegTHOR | aorta | 40/40 | 94.37565 | 94.38067 | +0.00502 | +0.00908 |
| Fluo-C3DH-A549 | cell | 90/90 | 93.94401 | 93.95587 | +0.01186 | +0.01898 |
| CHAOS CT / TotalSegmentator v2 | liver | 20/20 | 97.08608 | 97.09150 | +0.00542 | +0.01023 |

The three negative dataset-level results are retained: PROMISE12 (-0.08217 pp), ACDC (-0.07373 pp), and SegTHOR
(-0.00188 pp). Three datasets have paired intervals entirely above zero: Liver, Pancreas, and HepaticVessel.

See the [generated historical Full-16 report](../../evidence/nnunet/full16/RESULTS.md) for model configurations and
per-dataset case outcome counts, or [browse the machine-readable evidence](../../evidence/nnunet). Its original
dataset-macro table remains part of the checksummed archive.

## Conclusion

On standard official nnU-Net outputs, RankSEG produced a positive average Dice change and more dataset-level gains
than losses. The result supports RankSEG as an **optional inference decoder** that can be evaluated without retraining
the underlying model.

It does not support replacing argmax unconditionally. The median gain is small, three datasets decline, and some
individual cases have material regressions. The defensible claim is therefore positive average benefit with explicit
downside reporting—not guaranteed improvement for every dataset or patient.

## Notes

### Probability calibration

The official v1 checkpoints used in most of Full-16 were trained with nnU-Net's default cross-entropy plus soft-Dice
objective. Their exported arrays are valid softmax-normalized scores, but normalization alone does not establish that
they are calibrated estimates of conditional class probabilities. Because RankSEG treats its inputs as probabilities,
calibration may affect the observed decoder difference.

Full-16 measures the practical plug-in effect on standard official outputs. It does not claim that those outputs are
calibrated or that the result is independent of the training loss. Overconfidence is a hypothesis for a separate,
matched CE-only versus CE + soft-Dice study; it is not inferred from the loss alone. See the
[calibration protocol](BENCHMARK_PROTOCOL.md#probability-interpretation-and-calibration).

### Scope and limitations

- Most evidence comes from official nnU-Net v1 models; the v2 result is an implementation-path replication, not an
  equally broad v2 benchmark.
- BrainTumour uses mutually exclusive softmax channels followed by nested-region metric evaluation. Full-16 does not
  validate RankSEG on models trained with overlapping sigmoid region channels.
- No Full-16 dataset uses an ignore label.
- Missing legacy timing or memory measurements are stored as `null`; the schema migration does not invent them.
- Rejected post-hoc variants and exclusions remain visible in
  [Rejected experiments](REJECTED_EXPERIMENTS.md).

### Verify the published evidence

```bash
python -m venv env
env/bin/python -m pip install -r environments/nnunet/requirements.txt
env/bin/python -m pip install --no-build-isolation -e ".[nnunet]"
env/bin/rankseg-nnunet-bench verify-evidence evidence/nnunet
```

Verification checks every published SHA256 and schema, validates all case counts, rebuilds the Full-16 aggregate in a
temporary directory, and requires byte-identical aggregate outputs. `requirements-lock.txt` records the complete
Linux/Python 3.10/CUDA 12.8 reference environment; `requirements.txt` is the portable installation input.

### Reproduce or extend the benchmark

- [Dataset manifests](../../configs/nnunet) define inputs, decoder settings, evaluation design, and provenance.
- [Model and dataset registries](../../registry/nnunet) record publisher URLs, sizes, and checksums.
- [Script guide](../../scripts/nnunet/README.md) documents the supported download, conversion, OOF inference, audit, and aggregate
  entry points.
- [Dataset plan](DATASET_PLAN.md) records cohort construction and completion status.
- [Results to date](../../results/nnunet/RESULTS_TO_DATE.md) provides the detailed scientific narrative and regression audit.

Large images, checkpoints, probability arrays, and local outputs are intentionally excluded from Git. The compact
`evidence/` directory contains the published summaries, paired case deltas, available audits, and rebuilt aggregate.
