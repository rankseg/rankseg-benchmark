"""Dataset registry for pre-computed probability masks.

Each entry maps a benchmark name to a Hugging Face Datasets repo/subdirectory
that stores:
- ``probs`` : (num_classes, *spatial) probability tensor (softmax/sigmoid output of a frozen model)
- ``label`` : (*spatial) integer class indices (or (num_classes, *spatial) bool for multilabel)

For foreground-only binary medical tasks, probabilities may remain stored as
two-channel background/foreground softmax outputs while ``rankseg_channels``
explicitly tells the runner to pass only the foreground channel to RankSEG.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
    eval_unit: str = "image"    # "image" or "case"
    num_folds: int | None = None
    case_id_key: str | None = None
    description: str = ""
    spatial_dims: int = 2
    rankseg_channels: tuple[int, ...] | None = None
    evaluation_class_ids: tuple[int, ...] | None = None
    background_class_id: int = 0
    source_type: str = "huggingface"
    class_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if self.output_mode not in {"multiclass", "multilabel"}:
            raise ValueError("output_mode must be 'multiclass' or 'multilabel'")
        if self.spatial_dims not in {2, 3}:
            raise ValueError("spatial_dims must be 2 or 3")
        if self.eval_unit not in {"image", "case", "volume"}:
            raise ValueError("eval_unit must be 'image', 'case', or 'volume'")
        if self.source_type not in {"huggingface", "local_artifacts"}:
            raise ValueError("source_type must be 'huggingface' or 'local_artifacts'")
        if not 0 <= self.background_class_id < self.num_classes:
            raise ValueError("background_class_id must be a valid class index")

        for field_name in ("rankseg_channels", "evaluation_class_ids"):
            values = getattr(self, field_name)
            if values is None:
                continue
            values = tuple(values)
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must not contain duplicates")
            if any(not isinstance(value, int) or not 0 <= value < self.num_classes for value in values):
                raise ValueError(f"{field_name} must contain valid class indices")
            object.__setattr__(self, field_name, values)
        if self.output_mode == "multiclass" and self.rankseg_channels is not None:
            full_channel_order = tuple(range(self.num_classes))
            if len(self.rankseg_channels) != 1 and self.rankseg_channels != full_channel_order:
                raise ValueError(
                    "multiclass rankseg_channels must select one foreground class or preserve the full class order"
                )
            if len(self.rankseg_channels) == 1 and self.rankseg_channels[0] == self.background_class_id:
                raise ValueError("single-channel multiclass routing must select a non-background class")
        if self.class_names is not None:
            class_names = tuple(self.class_names)
            if len(class_names) != self.num_classes:
                raise ValueError("class_names length must equal num_classes")
            object.__setattr__(self, "class_names", class_names)

    @property
    def display_mode(self) -> str:
        if self.output_mode == "multiclass" and self.rankseg_channels is not None:
            if len(self.rankseg_channels) == 1 and self.rankseg_channels[0] != self.background_class_id:
                return "foreground"
        return self.output_mode


REGISTRY: dict[str, DatasetSpec] = {
    "pascal_voc": DatasetSpec(
        name="pascal_voc",
        hf_repo=HF_BENCHMARK_REPO,
        hf_data_dir="pascal_voc",
        hf_split="test",
        num_classes=21,
        output_mode="multiclass",
        ignore_index=255,
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
    "kits": DatasetSpec(
        name="kits",
        hf_repo=HF_BENCHMARK_REPO,
        hf_data_dir="kits",
        hf_split="test",
        num_classes=2,
        output_mode="multiclass",
        ignore_index=255,
        eval_unit="case",
        num_folds=5,
        case_id_key="case_id",
        rankseg_channels=(1,),
        evaluation_class_ids=(1,),
        description="KiTS, 5-fold case-level evaluation from slice predictions.",
    ),
    "monai_btcv_swin_v058_msd_pancreas": DatasetSpec(
        name="monai_btcv_swin_v058_msd_pancreas",
        hf_repo="",
        hf_split="external_test",
        num_classes=2,
        output_mode="multiclass",
        eval_unit="volume",
        spatial_dims=3,
        rankseg_channels=(1,),
        evaluation_class_ids=(1,),
        source_type="local_artifacts",
        class_names=("background", "pancreas"),
        description="MONAI BTCV Swin UNETR 0.5.8 on unseen MSD Task07 Pancreas volumes.",
    ),
    "monai_btcv_swin_v058_msd_spleen": DatasetSpec(
        name="monai_btcv_swin_v058_msd_spleen",
        hf_repo="",
        hf_split="external_test",
        num_classes=2,
        output_mode="multiclass",
        eval_unit="volume",
        spatial_dims=3,
        rankseg_channels=(1,),
        evaluation_class_ids=(1,),
        source_type="local_artifacts",
        class_names=("background", "spleen"),
        description="MONAI BTCV Swin UNETR 0.5.8 on unseen MSD Task09 Spleen volumes.",
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


def _load_local_manifest(spec: DatasetSpec, artifact_dir: str | os.PathLike[str] | None) -> tuple[Path, dict]:
    if artifact_dir is None:
        raise ValueError(
            f"Dataset {spec.name!r} uses local MONAI artifacts; pass --artifact-dir with its generated cache directory"
        )
    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing local artifact manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read local artifact manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported local artifact schema in {manifest_path}")
    if manifest.get("benchmark_id") != spec.name:
        raise ValueError(
            f"Artifact benchmark mismatch: expected {spec.name!r}, got {manifest.get('benchmark_id')!r}"
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Artifact manifest {manifest_path} must contain a cases list")
    if not cases:
        raise ValueError(f"Artifact manifest {manifest_path} contains no cases")
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    case_files = [case.get("file") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError(f"Every case in artifact manifest {manifest_path} must contain a non-empty string case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"Artifact manifest {manifest_path} contains duplicate case IDs")
    if len(case_files) != len(cases) or any(not isinstance(case_file, str) or not case_file for case_file in case_files):
        raise ValueError(f"Every case in artifact manifest {manifest_path} must contain a non-empty string file path")
    if len(set(case_files)) != len(case_files):
        raise ValueError(f"Artifact manifest {manifest_path} contains duplicate case files")
    return root, manifest


def get_local_artifact_size(spec: DatasetSpec, *, artifact_dir: str | os.PathLike[str] | None) -> int:
    """Return the number of cases listed in a generated local artifact manifest."""
    _, manifest = _load_local_manifest(spec, artifact_dir)
    return len(manifest["cases"])


def _iter_local_rows(spec: DatasetSpec, artifact_dir: str | os.PathLike[str] | None) -> Iterator[dict]:
    root, manifest = _load_local_manifest(spec, artifact_dir)
    for case in manifest["cases"]:
        relative_path = Path(case["file"])
        if relative_path.is_absolute():
            raise ValueError(f"Artifact case path must be relative: {case['file']!r}")
        path = (root / relative_path).resolve()
        if root not in path.parents:
            raise ValueError(f"Artifact case path escapes its cache directory: {case['file']!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing local artifact case: {path}")
        expected_sha256 = case.get("sha256")
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ValueError(f"Invalid SHA-256 entry for local artifact case: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Artifact checksum mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
                )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Artifact case must contain a dictionary: {path}")
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported artifact case schema in {path}")
        case_id = payload.get("case_id", case.get("case_id"))
        if case_id != case.get("case_id"):
            raise ValueError(f"Artifact case ID mismatch in {path}")
        from rankseg_benchmark.monai_cache import validate_probability_payload

        validate_probability_payload(payload, spec.num_classes)
        yield {
            "probs": payload.get("probabilities"),
            "label": payload.get("label"),
            "case_id": case_id,
            "artifact_metadata": payload.get("metadata", {}),
        }


def iter_samples(
    spec: DatasetSpec,
    *,
    limit: int | None = None,
    device: torch.device | str = "cpu",
    cache_dataset: bool = False,
    cache_dir: str | None = None,
    artifact_dir: str | os.PathLike[str] | None = None,
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
        artifact_dir=artifact_dir,
    )


def iter_batches(
    spec: DatasetSpec,
    *,
    limit: int | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
    cache_dataset: bool = False,
    cache_dir: str | None = None,
    artifact_dir: str | os.PathLike[str] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield batched (probs, label) pairs.

    Consecutive samples are batched only when their decoded shapes match. This
    keeps variable-resolution segmentation datasets valid without padding labels.
    """
    for probs, labels, _metadata in _iter_batches(
        spec,
        limit=limit,
        device=device,
        batch_size=batch_size,
        cache_dataset=cache_dataset,
        cache_dir=cache_dir,
        artifact_dir=artifact_dir,
        include_metadata=False,
    ):
        yield probs, labels


def iter_batches_with_metadata(
    spec: DatasetSpec,
    *,
    limit: int | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
    cache_dataset: bool = False,
    cache_dir: str | None = None,
    artifact_dir: str | os.PathLike[str] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]]:
    """Yield batched (probs, label, metadata) triples.

    Metadata contains the row fields other than ``probs`` and ``label``. As in
    ``iter_batches``, consecutive samples are batched only when their decoded
    shapes match.
    """
    yield from _iter_batches(
        spec,
        limit=limit,
        device=device,
        batch_size=batch_size,
        cache_dataset=cache_dataset,
        cache_dir=cache_dir,
        artifact_dir=artifact_dir,
        include_metadata=True,
    )


def _iter_batches(
    spec: DatasetSpec,
    *,
    limit: int | None,
    device: torch.device | str,
    batch_size: int,
    cache_dataset: bool,
    cache_dir: str | None,
    artifact_dir: str | os.PathLike[str] | None,
    include_metadata: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if spec.source_type == "local_artifacts":
        LOGGER.info("Opening local artifact dataset: benchmark=%s artifact_dir=%s", spec.name, artifact_dir)
        rows = _iter_local_rows(spec, artifact_dir)
    else:
        from datasets import load_dataset  # imported lazily so import-time stays cheap

        load_kwargs = {"split": spec.hf_split, "streaming": not cache_dataset}
        if spec.hf_data_dir is not None:
            load_kwargs["data_dir"] = spec.hf_data_dir
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir
        LOGGER.info(
            "Opening Hugging Face dataset: repo=%s split=%s data_dir=%s mode=%s cache_dir=%s limit=%s "
            "device=%s batch_size=%d metadata=%s",
            spec.hf_repo,
            spec.hf_split,
            spec.hf_data_dir or "<repo root>",
            "cache-first" if cache_dataset else "streaming",
            _display_cache_dir(cache_dir),
            limit if limit is not None else "all",
            device,
            batch_size,
            include_metadata,
        )
        if cache_dataset:
            LOGGER.info("Caching dataset locally before benchmark; this may take a while on first run")
        rows = load_dataset(spec.hf_repo, **load_kwargs)
        if cache_dataset:
            LOGGER.info("Dataset cache ready: rows=%s", len(rows) if hasattr(rows, "__len__") else "unknown")

    probs_batch: list[torch.Tensor] = []
    label_batch: list[torch.Tensor] = []
    metadata_batch: list[dict[str, object]] = []
    current_shape: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    n_seen = 0
    n_batches = 0

    def flush() -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]] | None:
        nonlocal probs_batch, label_batch, metadata_batch, current_shape, n_batches
        if not probs_batch:
            return None
        if len(probs_batch) == 1:
            probs = probs_batch[0].unsqueeze(0).to(device)
            labels = label_batch[0].unsqueeze(0)
        else:
            probs = torch.stack(probs_batch).to(device)
            labels = torch.stack(label_batch)
        metadata = metadata_batch
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
        metadata_batch = []
        current_shape = None
        return probs, labels, metadata

    for row in rows:
        if limit is not None and n_seen >= limit:
            LOGGER.info("Reached sample limit: %d", limit)
            break
        probs = _decode_probs(
            row["probs"],
            num_classes=spec.num_classes,
            spatial_dims=spec.spatial_dims,
        )
        label = _decode_label(
            row["label"],
            output_mode=spec.output_mode,
            num_classes=spec.num_classes,
            spatial_dims=spec.spatial_dims,
        )
        if tuple(probs.shape[1:]) != tuple(label.shape[-spec.spatial_dims :]):
            raise ValueError(
                f"Probability/label spatial shape mismatch: probs={tuple(probs.shape)}, label={tuple(label.shape)}"
            )
        metadata = {k: v for k, v in row.items() if k not in {"probs", "label"}} if include_metadata else {}
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
        metadata_batch.append(metadata)

    batch = flush()
    if batch is not None:
        yield batch
    LOGGER.info("Dataset exhausted: samples=%d batches=%d", n_seen, n_batches)


def _decode_probs(raw, *, num_classes: int, spatial_dims: int = 2) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        if raw.ndim != spatial_dims + 1 or raw.shape[0] != num_classes:
            raise ValueError(
                f"Expected probs with shape (num_classes={num_classes}, {spatial_dims} spatial dims), "
                f"got {tuple(raw.shape)}"
            )
        return raw.detach().to(device="cpu", dtype=torch.float32).contiguous()
    arr = _as_numpy(raw)
    expected_ndim = spatial_dims + 1
    if arr.ndim != expected_ndim or arr.shape[0] != num_classes:
        raise ValueError(
            f"Expected probs with shape (num_classes={num_classes}, {spatial_dims} spatial dims), got {arr.shape}"
        )
    return torch.from_numpy(arr.astype(np.float32))


def _decode_label(
    raw,
    *,
    output_mode: str,
    num_classes: int | None = None,
    spatial_dims: int | None = None,
) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        if output_mode == "multilabel":
            if spatial_dims is not None and raw.ndim != spatial_dims + 1:
                raise ValueError(f"Expected multilabel label with {spatial_dims} spatial dims, got {tuple(raw.shape)}")
            if num_classes is not None and (raw.ndim < 1 or raw.shape[0] != num_classes):
                raise ValueError(f"Expected {num_classes} multilabel channels, got {tuple(raw.shape)}")
            return raw.detach().to(device="cpu", dtype=torch.bool).contiguous()
        if output_mode != "multiclass":
            raise ValueError(f"Unknown output_mode: {output_mode!r}")
        if spatial_dims is not None and raw.ndim != spatial_dims:
            raise ValueError(f"Expected multiclass label with {spatial_dims} spatial dims, got {tuple(raw.shape)}")
        return raw.detach().to(device="cpu", dtype=torch.int64).contiguous()
    arr = _as_numpy(raw)
    if output_mode == "multilabel":
        if spatial_dims is not None and arr.ndim != spatial_dims + 1:
            raise ValueError(f"Expected multilabel label with {spatial_dims} spatial dims, got {arr.shape}")
        if num_classes is not None and (arr.ndim < 1 or arr.shape[0] != num_classes):
            raise ValueError(f"Expected {num_classes} multilabel channels, got {arr.shape}")
        return torch.from_numpy(arr.astype(np.bool_))
    if output_mode != "multiclass":
        raise ValueError(f"Unknown output_mode: {output_mode!r}")
    if spatial_dims is not None and arr.ndim != spatial_dims:
        raise ValueError(f"Expected multiclass label with {spatial_dims} spatial dims, got {arr.shape}")
    return torch.from_numpy(arr.astype(np.int64))


def _as_numpy(raw) -> np.ndarray:
    if isinstance(raw, bytes):
        return np.load(io.BytesIO(raw))
    return np.asarray(raw)
