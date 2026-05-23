# RankSEG Benchmark

Reproducible benchmark for [RankSEG](https://github.com/rankseg/rankseg): measures
**segmentation performance (Dice / IoU) and runtime** against the `argmax` baseline
on **pre-computed probability masks** from frozen segmentation models.

No training. No segmentation model setup. Just one command.

## Quick start

```bash
pip install rankseg-benchmark

# Run RankSEG vs. argmax on ADE20K (downloads pre-computed probs from Hugging Face)
rankseg-bench --dataset ade20k --solver RMA --metric dice
```

You'll get a table like:

```
## Performance

|       | argmax | rankseg-RMA | Improvement |
|-------|--------|-------------|-------------|
| mDice |  51.21 |       52.17 |       +0.96 |
| mIoU  |  40.00 |       40.82 |       +0.82 |

## Runtime

|                 | argmax | rankseg-RMA | Overhead |
|-----------------|--------|-------------|----------|
| mean ms / img   |   1.20 |        8.40 |    +7.20 |
| median ms / img |   1.10 |        7.90 |    +6.80 |
```

## Benchmark results

Performance numbers are reported as percentages. Improvement is RankSEG minus
the argmax baseline.

### Pascal VOC

|       | argmax | rankseg-RMA | Improvement |
|-------|-------:|------------:|------------:|
| mDice |  91.02 |       91.46 |       +0.44 |
| mIoU  |  87.80 |       88.23 |       +0.43 |

### Cityscapes

|       | argmax | rankseg-RMA | Improvement |
|-------|-------:|------------:|------------:|
| mDice |  82.62 |       83.23 |       +0.61 |
| mIoU  |  75.67 |       76.19 |       +0.52 |

### ADE20K

|       | argmax | rankseg-RMA | Improvement |
|-------|-------:|------------:|------------:|
| mDice |  63.98 |       64.92 |       +0.93 |
| mIoU  |  56.95 |       57.67 |       +0.72 |

### KiTS

|       |   argmax |   rankseg-RMA |   Improvement |
|-------|----------|---------------|---------------|
| mDice |    61.16 |         63.53 |          2.37 |
| mIoU  |    54.19 |         56.21 |          2.02 |

## Available datasets

| Name         | Classes | Mode        | Source model        |
|--------------|--------:|-------------|---------------------|
| `pascal_voc` |      21 | multiclass  | Pre-computed probs   |
| `ade20k`     |     150 | multiclass  | Pre-computed probs   |
| `cityscapes` |      19 | multiclass  | Pre-computed probs   |
| `kits`       |       2 | multiclass  | Pre-computed probs   |

List datasets at runtime:

```bash
rankseg-bench --list-datasets
```

## Common flags

```bash
# Quick smoke run (5 samples) on CPU
rankseg-bench --dataset pascal_voc --limit 5 --device cpu

# IoU metric instead of Dice
rankseg-bench --dataset pascal_voc --metric iou --solver RMA

# Per-class Dice/IoU breakdown (optional — large classes count = long table)
rankseg-bench --dataset ade20k --solver RMA --per-class

# Save full results to JSON (includes per-class arrays)
rankseg-bench --dataset pascal_voc --solver RMA --json results_voc.json

# Test a local RankSEG checkout instead of the installed package
rankseg-bench --dataset pascal_voc --solver RMA --rankseg-path ~/code/rankseg
```

You can also set `RANKSEG_PATH=~/code/rankseg`. The `--rankseg-path` flag takes
precedence when both are provided. If neither is set, the benchmark imports the
normal installed `rankseg` package from the active Python environment.

## Bring your own model

The bundled pre-computed probability dumps are loaded from
[`ZixunWang/rankseg-benchmark`](https://huggingface.co/datasets/ZixunWang/rankseg-benchmark),
using its `pascal_voc`, `ade20k`, and `cityscapes` data directories. If you
want to benchmark **your own** model's probabilities, generate a dump with the
provided script and point the registry at your repo:

```bash
python scripts/generate_probs.py \
    --model deeplabv3plus_voc \
    --images-dir /path/to/voc/JPEGImages \
    --labels-dir /path/to/voc/SegmentationClass \
    --split-file /path/to/voc/ImageSets/Segmentation/val.txt \
    --num-classes 21 \
    --output ./probs_voc_mine
```

Then either upload the directory to a Hugging Face dataset repo and add an
entry to `rankseg_benchmark/datasets.py`, or call the benchmark internals
directly:

```python
from rankseg_benchmark.datasets import DatasetSpec
from rankseg_benchmark.runner import run_benchmark, format_report

spec = DatasetSpec(
    name="voc-mine",
    hf_repo="your-username/your-probs-repo",
    hf_split="validation",
    hf_data_dir=None,
    num_classes=21,
    output_mode="multiclass",
    ignore_index=255,
)
baseline, rs = run_benchmark(spec, solver="RMA")
print(format_report(baseline, rs, per_class=True))
```

## What gets measured

- **Performance (`mDice` / `mIoU`)**: the per-image metric used in
  the [RankSEG-RMA paper](https://openreview.net/pdf?id=4tRMm1JJhw). For each
  image we compute per-class Dice/IoU, average over the classes that are
  actually active in that image, then average across the dataset. This matches
  the [reference implementation](https://github.com/ZixunWang/RankSEG-RMA/blob/master/exp/metrics/accuracy_metric.py).
- **Runtime**: per-image latency for `RankSEG.predict` vs. `torch.argmax`,
  with CUDA synchronization and a configurable warmup (default 3 samples).
- **Per-class gain breakdown** (with `--per-class`): for each class, its
  Dice/IoU averaged over the images where the class is active. Useful for
  showing RankSEG's larger lift on rare classes.

## License

BSD 3-Clause.
