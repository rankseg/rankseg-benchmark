"""Download the four official TIAToolbox checkpoints into ignored artifacts/."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from . import bcss, hovernet_consep, hovernet_kumar, micronet_consep


BASE_URL = "https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/resolve/main"
WEIGHTS = {
    "micronet-consep": (micronet_consep.CHECKPOINT, micronet_consep.CHECKPOINT_SHA256),
    "hovernet_original-consep": (hovernet_consep.CHECKPOINT, hovernet_consep.CHECKPOINT_SHA256),
    "hovernet_original-kumar": (hovernet_kumar.CHECKPOINT, hovernet_kumar.CHECKPOINT_SHA256),
    "fcn_resnet50_unet-bcss": (bcss.CHECKPOINT, bcss.CHECKPOINT_SHA256),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, *, verify_only: bool = False) -> None:
    import requests

    path, expected = WEIGHTS[name]
    if path.exists():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{path}: SHA-256 {actual}, expected {expected}")
        print(f"verified {path}")
        return
    if verify_only:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        raise FileExistsError(f"Inspect or remove incomplete download: {partial}")
    url = f"{BASE_URL}/{path.name}"
    try:
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        actual = sha256(partial)
        if actual != expected:
            raise ValueError(f"{url}: SHA-256 {actual}, expected {expected}")
        os.replace(partial, path)
        print(f"downloaded and verified {path}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(WEIGHTS), help="Fetch or verify one checkpoint")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing files without downloading")
    args = parser.parse_args()
    for name in (args.only,) if args.only else WEIGHTS:
        download(name, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
