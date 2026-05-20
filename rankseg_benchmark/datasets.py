"""Dataset registry for pre-computed probability masks.

Each entry maps a benchmark name to a Hugging Face Datasets repo/subdirectory
that stores:
- ``probs`` : (num_classes, H, W) fp16 probability tensor (softmax/sigmoid output of a frozen model)
- ``label`` : (H, W) int64 ground-truth class indices (or (num_classes, H, W) bool for multilabel)

For binary medical tasks we store probabilities as (2, H, W) or (1, H, W) and rely on
``output_mode`` to control how RankSEG handles them.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import numpy as np
import torch

HF_BENCHMARK_REPO = "ZixunWang/rankseg-benchmark"
HF_DATASETS_SIZE_URL = "https://datasets-server.huggingface.co/size"
LOGGER = logging.getLogger(__name__)


def _display_cache_dir(cache_dir: str | None) -> str:
    if cache_dir is not None:
        return str(Path(cache_dir).expanduser())
    if os.environ.get("HF_DATASETS_CACHE"):
        return str(Path(os.environ["HF_DATASETS_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        return str(Path(os.environ["HF_HOME"]).expanduser() / "datasets")
    return str(Path.home() / ".cache" / "huggingface" / "datasets")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_repo: str            # e.g. "ZixunWang/rankseg-benchmark"
    hf_split: str           # e.g. "test"
    num_classes: int
    output_mode: str        # "multiclass" or "multilabel"
    hf_data_dir: str | None = None
    ignore_index: int | None = None
    description: str = ""


REGISTRY: dict[str, DatasetSpec] = {
    "pascal_voc": DatasetSpec(
        name="pascal_voc",
        hf_repo=HF_BENCHMARK_REPO,
        hf_data_dir="pascal_voc",
        hf_split="test",
        num_classes=21,
        output_mode="multiclass",
        ignore_index=21,
        description="PASCAL VOC 2012, precomputed probabilities.",
    ),
    "ade20k": DatasetSpec(
        name="ade20k",
        hf_repo=HF_BENCHMARK_REPO,
        hf_data_dir="ade20k",
        hf_split="test",
        num_classes=150,
        output_mode="multiclass",
        ignore_index=255,
        description="ADE20K, precomputed probabilities.",
    ),
    "cityscapes": DatasetSpec(
        name="cityscapes",
        hf_repo=HF_BENCHMARK_REPO,
        hf_data_dir="cityscapes",
        hf_split="test",
        num_classes=19,
        output_mode="multiclass",
        ignore_index=255,
        description="Cityscapes, precomputed probabilities.",
    ),
}


def list_datasets() -> list[str]:
    return sorted(REGISTRY.keys())


def get_spec(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list_datasets()}")
    return REGISTRY[name]


def get_dataset_size_from_metadata(spec: DatasetSpec) -> int | None:
    """Return row count from the Hugging Face Dataset Viewer metadata API."""
    params = urlencode({"dataset": spec.hf_repo})
    request = Request(f"{HF_DATASETS_SIZE_URL}?{params}", headers={"Accept": "application/json"})
    LOGGER.info(
        "Resolving dataset size from Hugging Face metadata: repo=%s split=%s config=%s",
        spec.hf_repo,
        spec.hf_split,
        spec.hf_data_dir or "<default>",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not resolve dataset size from Hugging Face metadata: %s", exc)
        return None

    size = payload.get("size", {})
    target_config = spec.hf_data_dir
    split_rows = size.get("splits") or []

    candidates = []
    for item in split_rows:
        if item.get("split") != spec.hf_split:
            continue
        if target_config is None or item.get("config") == target_config:
            candidates.append(item)

    if len(candidates) == 1 and candidates[0].get("num_rows") is not None:
        rows = int(candidates[0]["num_rows"])
        LOGGER.info("Dataset size resolved from metadata: rows=%d", rows)
        return rows

    if target_config is None:
        dataset_rows = size.get("dataset", {}).get("num_rows")
        if dataset_rows is not None:
            rows = int(dataset_rows)
            LOGGER.info("Dataset size resolved from metadata: rows=%d", rows)
            return rows

    LOGGER.warning(
        "Dataset metadata did not contain an unambiguous row count for split=%s config=%s",
        spec.hf_split,
        target_config or "<default>",
    )
    return None


def get_dataset_size_from_cache(spec: DatasetSpec, *, cache_dir: str | None = None) -> int:
    """Return dataset row count, downloading/building the local HF cache if needed."""
    from datasets import load_dataset  # imported lazily so import-time stays cheap

    load_kwargs = {"split": spec.hf_split, "streaming": False}
    if spec.hf_data_dir is not None:
        load_kwargs["data_dir"] = spec.hf_data_dir
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir

    LOGGER.info(
        "Resolving dataset size from local cache: repo=%s split=%s data_dir=%s cache_dir=%s",
        spec.hf_repo,
        spec.hf_split,
        spec.hf_data_dir or "<repo root>",
        _display_cache_dir(cache_dir),
    )
    ds = load_dataset(spec.hf_repo, **load_kwargs)
    size = len(ds)
    LOGGER.info("Dataset size resolved: rows=%d", size)
    return size


def iter_samples(
    spec: DatasetSpec,
    *,
    limit: int | None = None,
    device: torch.device | str = "cpu",
    cache_dataset: bool = False,
    cache_dir: str | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (probs, label) pairs.

    probs : (1, C, H, W) float32 on ``device``, decoded from stored fp16.
    label : (1, H, W) int64 on ``device`` (multiclass) or (1, C, H, W) bool (multilabel).
    """
    yield from iter_batches(
        spec,
        limit=limit,
        device=device,
        batch_size=1,
        cache_dataset=cache_dataset,
        cache_dir=cache_dir,
    )


def iter_batches(
    spec: DatasetSpec,
    *,
    limit: int | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
    cache_dataset: bool = False,
    cache_dir: str | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield batched (probs, label) pairs.

    Consecutive samples are batched only when their decoded shapes match. This
    keeps variable-resolution segmentation datasets valid without padding labels.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    from datasets import load_dataset  # imported lazily so import-time stays cheap

    load_kwargs = {"split": spec.hf_split, "streaming": not cache_dataset}
    if spec.hf_data_dir is not None:
        load_kwargs["data_dir"] = spec.hf_data_dir
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    LOGGER.info(
        "Opening Hugging Face dataset: repo=%s split=%s data_dir=%s mode=%s cache_dir=%s limit=%s device=%s batch_size=%d",
        spec.hf_repo,
        spec.hf_split,
        spec.hf_data_dir or "<repo root>",
        "cache-first" if cache_dataset else "streaming",
        _display_cache_dir(cache_dir),
        limit if limit is not None else "all",
        device,
        batch_size,
    )
    if cache_dataset:
        LOGGER.info("Caching dataset locally before benchmark; this may take a while on first run")
    ds = load_dataset(spec.hf_repo, **load_kwargs)
    if cache_dataset:
        LOGGER.info("Dataset cache ready: rows=%s", len(ds) if hasattr(ds, "__len__") else "unknown")

    probs_batch: list[torch.Tensor] = []
    label_batch: list[torch.Tensor] = []
    current_shape: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    n_seen = 0
    n_batches = 0

    def flush() -> tuple[torch.Tensor, torch.Tensor] | None:
        nonlocal probs_batch, label_batch, current_shape, n_batches
        if not probs_batch:
            return None
        probs = torch.stack(probs_batch).to(device)
        labels = torch.stack(label_batch).to(device)
        n_batches += 1
        LOGGER.debug(
            "Yielding batch %d: batch_size=%d probs_shape=%s label_shape=%s",
            n_batches,
            probs.size(0),
            tuple(probs.shape),
            tuple(labels.shape),
        )
        probs_batch = []
        label_batch = []
        current_shape = None
        return probs, labels

    for row in ds:
        if limit is not None and n_seen >= limit:
            LOGGER.info("Reached sample limit: %d", limit)
            break
        probs = _decode_probs(row["probs"], num_classes=spec.num_classes)
        label = _decode_label(row["label"], output_mode=spec.output_mode)
        n_seen += 1
        shape_key = (tuple(probs.shape), tuple(label.shape))

        if n_seen == 1:
            LOGGER.info(
                "First sample decoded: probs_shape=%s probs_dtype=%s label_shape=%s label_dtype=%s",
                tuple(probs.shape),
                probs.dtype,
                tuple(label.shape),
                label.dtype,
            )

        if current_shape is not None and (shape_key != current_shape or len(probs_batch) >= batch_size):
            batch = flush()
            if batch is not None:
                yield batch

        current_shape = shape_key
        probs_batch.append(probs)
        label_batch.append(label)

    batch = flush()
    if batch is not None:
        yield batch
    LOGGER.info("Dataset exhausted: samples=%d batches=%d", n_seen, n_batches)


def _decode_probs(raw, *, num_classes: int) -> torch.Tensor:
    arr = _as_numpy(raw)
    if arr.ndim != 3 or arr.shape[0] != num_classes:
        raise ValueError(
            f"Expected probs shape (num_classes={num_classes}, H, W), got {arr.shape}"
        )
    return torch.from_numpy(arr.astype(np.float32))


def _decode_label(raw, *, output_mode: str) -> torch.Tensor:
    arr = _as_numpy(raw)
    if output_mode == "multilabel":
        return torch.from_numpy(arr.astype(np.bool_))
    return torch.from_numpy(arr.astype(np.int64))


def _as_numpy(raw) -> np.ndarray:
    if isinstance(raw, bytes):
        return np.load(io.BytesIO(raw))
    return np.asarray(raw)
