"""Separate versioned experiment definitions from the large local workspace."""
from pathlib import Path
import os


def repository_root():
    return Path(__file__).resolve().parents[2]


def manifest_repository_root(base):
    """Find the checkout containing built-in manifests, also from an installed wheel."""
    base = Path(base).resolve()
    if base.name == "nnunet" and base.parent.name == "configs":
        root = base.parent.parent
        if (root / "pyproject.toml").is_file() and (base / "full16_evidence.yaml").is_file():
            return root
    return None


def workspace_root(default=None):
    """Reuse an existing nnU-Net work/outputs/artifacts tree without copying it."""
    override = os.environ.get("RANKSEG_NNUNET_WORKSPACE")
    return Path(override).expanduser().resolve() if override else Path(default or repository_root()).resolve()


def resolve_manifest_path(base, value):
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    resolved = (base / path).resolve()
    root = manifest_repository_root(base)
    # Only relocated built-in manifests opt into workspace redirection.
    # User manifests and absolute paths keep their original semantics.
    if root is not None:
        for folder in ("work", "outputs", "artifacts", "data"):
            try:
                suffix = resolved.relative_to(root / folder)
            except ValueError:
                continue
            return workspace_root(root) / folder / suffix
    return resolved
