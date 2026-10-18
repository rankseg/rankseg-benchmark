"""Validate the decoder implementation and record its installed provenance."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


VERSION = "0.0.8"
SOURCE_FILES = ("functional.py", "_rankseg_algo.py", "_screening.py", "_screening_cuda.py")
SOURCE_SHA256 = "f0ff60cff120f8c6e1b663bd29b42351a62df735f23335a06852ced4e66b3262"


def decoder_source() -> dict:
    import rankseg

    assert rankseg.__version__ == VERSION, f"Install rankseg=={VERSION}; found {rankseg.__version__}"
    source = Path(rankseg.__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in SOURCE_FILES:
        digest.update(filename.encode())
        digest.update((source / filename).read_bytes())
    assert digest.hexdigest() == SOURCE_SHA256, "RankSEG decoder source differs from the benchmark implementation"
    revision = None
    if (source.parent / ".git").exists():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source.parent, text=True).strip()
    return {
        "rankseg_version": rankseg.__version__,
        "package_path": str(source),
        "git_revision": revision,
        "source_sha256": digest.hexdigest(),
        "files_hashed": list(SOURCE_FILES),
    }


def runtime_source() -> dict:
    import tiatoolbox
    import torch

    assert tiatoolbox.__version__ == "1.6.0", tiatoolbox.__version__
    source = decoder_source()
    return {
        "rankseg_version": source["rankseg_version"],
        "rankseg_path": source["package_path"],
        "rankseg_git_revision": source["git_revision"],
        "rankseg_source_sha256": source["source_sha256"],
        "rankseg_files_hashed": source["files_hashed"],
        "tiatoolbox_version": tiatoolbox.__version__,
        "torch_version": torch.__version__,
    }
