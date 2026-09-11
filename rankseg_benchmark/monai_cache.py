"""Generate reproducible 3D probability artifacts from pinned MONAI Bundles."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import tempfile
import time
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import torch

GENERATOR_REVISION = 1
EXTERNAL_EXPOSURE_DECLARATION = "none_by_declared_dataset_provenance"


def _validate_external_test_provenance(benchmark_id: str, spec: dict[str, Any]) -> None:
    """Reject built-in targets that do not declare an independent test design."""
    dataset = spec.get("dataset", {})
    bundle = spec.get("bundle", {})
    if dataset.get("evaluation_design") != "external_test":
        raise ValueError(f"MONAI benchmark {benchmark_id!r} must use an external_test evaluation design")
    if dataset.get("checkpoint_exposure") != EXTERNAL_EXPOSURE_DECLARATION:
        raise ValueError(
            f"MONAI benchmark {benchmark_id!r} must declare checkpoint exposure as "
            f"{EXTERNAL_EXPOSURE_DECLARATION!r}"
        )

    test_case_usage = dataset.get("test_case_usage")
    required_false_flags = (
        "checkpoint_training",
        "checkpoint_selection",
        "postprocessing_parameter_selection",
    )
    if not isinstance(test_case_usage, dict) or any(test_case_usage.get(key) is not False for key in required_false_flags):
        raise ValueError(f"MONAI benchmark {benchmark_id!r} may not use test cases for training or parameter selection")

    source_dataset_id = dataset.get("source_dataset_id")
    training_dataset_ids = {
        bundle.get("supervised_training_dataset_id"),
        *bundle.get("self_supervised_pretraining_dataset_ids", []),
    }
    training_dataset_ids.discard(None)
    if not source_dataset_id or source_dataset_id in training_dataset_ids:
        raise ValueError(
            f"MONAI benchmark {benchmark_id!r} test dataset must be declared separately from every checkpoint source"
        )


def load_monai_specs() -> dict[str, dict[str, Any]]:
    """Load and validate the built-in MONAI benchmark manifests."""
    path = resources.files("rankseg_benchmark").joinpath("monai_specs.json")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("benchmarks"), dict):
        raise ValueError("Unsupported built-in MONAI benchmark manifest schema")
    benchmarks = payload["benchmarks"]
    for benchmark_id, spec in benchmarks.items():
        _validate_external_test_provenance(benchmark_id, spec)
    return benchmarks


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _case_id(path: Path) -> str:
    return path.name[: -len(".nii.gz")] if path.name.endswith(".nii.gz") else path.stem


def resolve_cases(spec: dict[str, Any], dataset_root: Path) -> list[tuple[str, Path, Path]]:
    """Resolve the exact labeled cases selected by a built-in split definition."""
    images_dir = dataset_root / "imagesTr"
    labels_dir = dataset_root / "labelsTr"
    split = spec["dataset"]["split"]
    strategy = split["strategy"]
    if strategy == "fixed_case_ids":
        case_ids = list(split["case_ids"])
    elif strategy == "sorted_last_n":
        labels = sorted(labels_dir.glob("*.nii.gz"))
        count = int(split["count"])
        if len(labels) < count:
            raise FileNotFoundError(f"Expected at least {count} labels in {labels_dir}, found {len(labels)}")
        case_ids = [_case_id(path) for path in labels[-count:]]
    else:
        raise ValueError(f"Unknown MONAI split strategy: {strategy!r}")

    cases = []
    for name in case_ids:
        image = images_dir / f"{name}.nii.gz"
        label = labels_dir / f"{name}.nii.gz"
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing image/label pair for {name}: image={image}, label={label}")
        cases.append((name, image, label))
    return cases


def validate_probability_payload(payload: dict[str, Any], num_classes: int) -> None:
    probabilities = payload.get("probabilities")
    label = payload.get("label")
    if not isinstance(probabilities, torch.Tensor) or not isinstance(label, torch.Tensor):
        raise TypeError("Artifact probabilities and label must be tensors")
    if probabilities.ndim != 4 or label.ndim != 3:
        raise ValueError(
            f"Expected 3D probabilities (C,D,H,W) and label (D,H,W), got {probabilities.shape}, {label.shape}"
        )
    if probabilities.shape[0] != num_classes or tuple(probabilities.shape[1:]) != tuple(label.shape):
        raise ValueError(f"Probability/label shape mismatch: {probabilities.shape}, {label.shape}")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Probability artifact contains non-finite values")
    if float(probabilities.min()) < 0.0 or float(probabilities.max()) > 1.0:
        raise ValueError("Probability artifact is outside [0, 1]")
    sums = probabilities.float().sum(dim=0)
    if not torch.allclose(sums, torch.ones_like(sums), atol=2e-5, rtol=2e-5):
        raise ValueError("Softmax probability channels do not sum to one")
    labels = set(int(value) for value in torch.unique(label))
    if not labels.issubset(set(range(num_classes))):
        raise ValueError(f"Artifact contains invalid labels: {sorted(labels)}")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_monai():
    try:
        import monai
        from monai.bundle import ConfigParser
        from monai.inferers import SlidingWindowInferer
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            EnsureTyped,
            LoadImaged,
            Orientationd,
            ScaleIntensityRanged,
            Spacingd,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("MONAI cache generation requires `pip install rankseg-benchmark[monai]`") from exc
    return {
        "monai": monai,
        "ConfigParser": ConfigParser,
        "SlidingWindowInferer": SlidingWindowInferer,
        "Compose": Compose,
        "LoadImaged": LoadImaged,
        "EnsureChannelFirstd": EnsureChannelFirstd,
        "Orientationd": Orientationd,
        "Spacingd": Spacingd,
        "ScaleIntensityRanged": ScaleIntensityRanged,
        "EnsureTyped": EnsureTyped,
    }


def _verify_bundle_file(bundle_root: Path, descriptor: dict[str, str]) -> Path:
    path = bundle_root / descriptor["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing Bundle file: {path}")
    actual = _hash_file(path, descriptor["hash_type"])
    if actual != descriptor["hash_value"]:
        raise ValueError(
            f"Bundle checksum mismatch for {path}: expected {descriptor['hash_value']}, got {actual}"
        )
    return path


def _build_model(spec: dict[str, Any], bundle_root: Path, device: torch.device, monai_api: dict[str, Any]):
    bundle = spec["bundle"]
    config_path = bundle_root / bundle["config"]
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Bundle inference config: {config_path}")
    checkpoint_path = _verify_bundle_file(bundle_root, bundle["checkpoint"])

    parser = monai_api["ConfigParser"]()
    parser.read_config(config_path)
    parser["bundle_root"] = str(bundle_root)
    parser["device"] = device
    if "architecture" in bundle:
        architecture_path = _verify_bundle_file(bundle_root, bundle["architecture"])
        architecture = torch.load(architecture_path, map_location=device, weights_only=False)
        parser["arch_ckpt"] = architecture
        parser["dints_space#device"] = str(device)

    network_definition = parser["network_def"]
    if not isinstance(network_definition, dict):
        raise ValueError("Bundle network_def must be a dictionary")
    target_name = str(network_definition.get("_target_", "")).rsplit(".", 1)[-1]
    target = getattr(monai_api["monai"].networks.nets, target_name, None)
    if target is None:
        raise ValueError(f"Could not resolve Bundle network target {target_name!r}")
    accepted_parameters = inspect.signature(target).parameters
    for optional_argument in bundle.get("network_optional_args", []):
        if optional_argument not in accepted_parameters:
            network_definition.pop(optional_argument, None)

    model = parser.get_parsed_content("network_def").to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    return model, config_path, checkpoint_path


def _build_preprocessing(spec: dict[str, Any], monai_api: dict[str, Any]):
    config = spec["preprocessing"]
    a_min, a_max, b_min, b_max = config["intensity"]
    return monai_api["Compose"](
        [
            monai_api["LoadImaged"](keys=("image", "label")),
            monai_api["EnsureChannelFirstd"](keys=("image", "label")),
            monai_api["Orientationd"](keys=("image", "label"), axcodes=config["orientation"]),
            monai_api["Spacingd"](
                keys=("image", "label"),
                pixdim=tuple(config["pixdim"]),
                mode=("bilinear", "nearest"),
            ),
            monai_api["ScaleIntensityRanged"](
                keys="image",
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            ),
            monai_api["EnsureTyped"](keys=("image", "label")),
        ]
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _adapt_probabilities(probabilities: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    transform = spec.get("output_transform")
    if transform is None:
        return probabilities
    if transform.get("type") != "binary_one_vs_rest":
        raise ValueError(f"Unsupported MONAI output transform: {transform.get('type')!r}")
    expected_classes = int(transform["model_num_classes"])
    foreground_channel = int(transform["foreground_channel"])
    if probabilities.ndim != 4 or probabilities.shape[0] != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} model probability channels, got {tuple(probabilities.shape)}"
        )
    if not 0 <= foreground_channel < expected_classes:
        raise ValueError(f"Invalid foreground model channel: {foreground_channel}")
    foreground = probabilities[foreground_channel]
    return torch.stack((1.0 - foreground, foreground))


def _adapt_label(label: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    foreground_ids = spec["dataset"].get("source_label_foreground_ids")
    if foreground_ids is None:
        return label.long()
    allowed_ids = {0, *(int(class_id) for class_id in foreground_ids)}
    observed_ids = {int(class_id) for class_id in torch.unique(label)}
    if not observed_ids.issubset(allowed_ids):
        raise ValueError(f"Source label contains values outside {sorted(allowed_ids)}: {sorted(observed_ids)}")
    foreground = torch.zeros_like(label, dtype=torch.bool)
    for class_id in foreground_ids:
        foreground |= label == int(class_id)
    return foreground.long()


@torch.inference_mode()
def _infer_case(model, image: torch.Tensor, spec: dict[str, Any], device: torch.device, monai_api):
    config = spec["inference"]
    inferer = monai_api["SlidingWindowInferer"](
        roi_size=tuple(config["roi_size"]),
        sw_batch_size=int(config["sw_batch_size"]),
        overlap=float(config["overlap"]),
    )
    _synchronize(device)
    started = time.perf_counter()
    with torch.autocast(device_type=device.type, enabled=bool(config["amp"])):
        logits = inferer(image.unsqueeze(0).to(device), model)
    probabilities = _adapt_probabilities(torch.softmax(logits, dim=1)[0], spec)
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return probabilities.detach().cpu().float().contiguous(), elapsed_ms


def _software_info(monai_module) -> dict[str, Any]:
    info = {
        "python": os.sys.version.split()[0],
        # Some PyTorch releases expose TorchVersion, a str subclass that the
        # weights-only unpickler intentionally rejects. Store plain strings so
        # generated artifacts remain safely loadable.
        "torch": str(torch.__version__),
        "monai": str(monai_module.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def generate_artifacts(
    benchmark_id: str,
    *,
    dataset_root: Path,
    bundle_parent: Path,
    output_dir: Path,
    device: torch.device,
    limit: int | None = None,
    overwrite: bool = False,
    download_bundle: bool = False,
) -> Path:
    specs = load_monai_specs()
    if benchmark_id not in specs:
        raise KeyError(f"Unknown MONAI benchmark {benchmark_id!r}; available: {sorted(specs)}")
    spec = specs[benchmark_id]
    if spec.get("activation") != "softmax":
        raise ValueError(f"Unsupported activation for local MONAI artifacts: {spec.get('activation')!r}")
    monai_api = _require_monai()
    bundle = spec["bundle"]
    bundle_root = bundle_parent / bundle["name"]
    if not bundle_root.is_dir():
        if not download_bundle:
            raise FileNotFoundError(
                f"Missing Bundle {bundle_root}; pass --download-bundle or provide the downloaded Bundle"
            )
        monai_api["monai"].bundle.download(
            name=bundle["name"],
            version=bundle["version"],
            bundle_dir=bundle_parent,
        )

    all_cases = resolve_cases(spec, dataset_root)
    cases = all_cases
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        cases = cases[:limit]

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, config_path, checkpoint_path = _build_model(spec, bundle_root, device, monai_api)
    preprocessing = _build_preprocessing(spec, monai_api)
    class_names = spec["dataset"]["class_names"]
    software = _software_info(monai_api["monai"])
    bundle_hashes = {
        "config_sha256": _hash_file(config_path),
        "checkpoint_sha256": _hash_file(checkpoint_path),
    }
    if "architecture" in bundle:
        architecture_path = _verify_bundle_file(bundle_root, bundle["architecture"])
        bundle_hashes["architecture_sha256"] = _hash_file(architecture_path)
    generation_fingerprint = _hash_json(
        {
            "generator_revision": GENERATOR_REVISION,
            "benchmark_spec": spec,
            "bundle_hashes": bundle_hashes,
            "device": str(device),
            "software": software,
        }
    )
    case_records = []
    for index, (case_id, image_path, label_path) in enumerate(cases, start=1):
        source_image_sha256 = _hash_file(image_path)
        source_label_sha256 = _hash_file(label_path)
        destination = output_dir / "cases" / f"{case_id}.pt"
        if destination.exists() and not overwrite:
            payload = torch.load(destination, map_location="cpu", weights_only=True)
            validate_probability_payload(payload, len(class_names))
            metadata = payload.get("metadata")
            expected_metadata = {
                "coordinate_space": spec["coordinate_space"],
                "source_image_sha256": source_image_sha256,
                "source_label_sha256": source_label_sha256,
                "generation_fingerprint": generation_fingerprint,
            }
            if payload.get("schema_version") != 1 or payload.get("case_id") != case_id:
                raise ValueError(f"Existing artifact identity does not match {case_id}: {destination}")
            if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected_metadata.items()):
                raise ValueError(
                    f"Existing artifact provenance does not match this generation run: {destination}; "
                    "pass --overwrite to regenerate it"
                )
            print(f"[{index}/{len(cases)}] Reusing {destination}", flush=True)
        else:
            print(f"[{index}/{len(cases)}] Inferring {case_id} on {device}", flush=True)
            transformed = preprocessing({"image": str(image_path), "label": str(label_path)})
            image = transformed["image"].as_tensor()
            label = _adapt_label(transformed["label"].as_tensor().squeeze(0), spec).cpu().contiguous()
            probabilities, inference_ms = _infer_case(model, image, spec, device, monai_api)
            payload = {
                "schema_version": 1,
                "case_id": case_id,
                "probabilities": probabilities,
                "label": label,
                "metadata": {
                    "coordinate_space": spec["coordinate_space"],
                    "inference_ms": inference_ms,
                    "source_image_sha256": source_image_sha256,
                    "source_label_sha256": source_label_sha256,
                    "generation_fingerprint": generation_fingerprint,
                    "software": software,
                    "output_transform": spec.get("output_transform"),
                },
            }
            validate_probability_payload(payload, len(class_names))
            _atomic_torch_save(payload, destination)
            print(f"Saved {destination} ({destination.stat().st_size / 1024**3:.2f} GiB)", flush=True)
        case_records.append(
            {
                "case_id": case_id,
                "file": str(destination.relative_to(output_dir)),
                "sha256": _hash_file(destination),
                "spatial_shape": list(payload["label"].shape),
            }
        )

    manifest = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "complete": len(case_records) == len(all_cases),
        "dataset": spec["dataset"],
        "evaluation_design": {
            "role": spec["dataset"]["evaluation_design"],
            "checkpoint_exposure": spec["dataset"]["checkpoint_exposure"],
            "test_case_usage": spec["dataset"]["test_case_usage"],
            "supervised_training_dataset": bundle["supervised_training_dataset"],
            "supervised_training_dataset_id": bundle["supervised_training_dataset_id"],
            "self_supervised_pretraining_datasets": bundle["self_supervised_pretraining_datasets"],
            "self_supervised_pretraining_dataset_ids": bundle["self_supervised_pretraining_dataset_ids"],
        },
        "bundle": {
            "name": bundle["name"],
            "version": bundle["version"],
            "config": bundle["config"],
            "config_sha256": bundle_hashes["config_sha256"],
            "checkpoint": bundle["checkpoint"],
            "checkpoint_sha256": bundle_hashes["checkpoint_sha256"],
        },
        "generation": {
            "activation": spec["activation"],
            "amp": bool(spec["inference"]["amp"]),
            "coordinate_space": spec["coordinate_space"],
            "probability_dtype": "float32",
            "device": str(device),
            "generator_revision": GENERATOR_REVISION,
            "fingerprint": generation_fingerprint,
            "preprocessing": spec["preprocessing"],
            "inference": spec["inference"],
            "software": software,
        },
        "cases": case_records,
    }
    if "architecture" in bundle:
        manifest["bundle"]["architecture"] = bundle["architecture"]
        manifest["bundle"]["architecture_sha256"] = bundle_hashes["architecture_sha256"]
    manifest_path = output_dir / "manifest.json"
    _atomic_json_save(manifest, manifest_path)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List built-in MONAI benchmark targets and exit.")
    parser.add_argument("--benchmark", choices=sorted(load_monai_specs()))
    parser.add_argument("--dataset-root", type=Path, help="Extracted MSD task directory containing imagesTr/labelsTr.")
    parser.add_argument("--bundle-dir", type=Path, default=Path("monai_bundles"), help="Parent directory for Bundles.")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated case artifacts and manifest.json.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-bundle", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    specs = load_monai_specs()
    if args.list:
        for benchmark_id, spec in specs.items():
            print(
                f"{benchmark_id:<30} {spec['bundle']['name']}=={spec['bundle']['version']} | "
                f"{spec['dataset']['name']}"
            )
        return 0
    if not args.benchmark or args.dataset_root is None or args.output_dir is None:
        parser.error("--benchmark, --dataset-root, and --output-dir are required unless --list is used")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu for a smoke run")
    manifest_path = generate_artifacts(
        args.benchmark,
        dataset_root=args.dataset_root.expanduser().resolve(),
        bundle_parent=args.bundle_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        device=device,
        limit=args.limit,
        overwrite=args.overwrite,
        download_bundle=args.download_bundle,
    )
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
