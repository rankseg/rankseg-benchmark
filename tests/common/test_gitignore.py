"""Published evidence must survive git add without admitting local data."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ignored_paths(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required to check repository ignore rules")
    subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    shutil.copyfile(ROOT / ".gitignore", tmp_path / ".gitignore")

    def check(paths):
        result = subprocess.run(
            [git, "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index", "--stdin"],
            cwd=tmp_path,
            input="\n".join(paths) + "\n",
            text=True,
            capture_output=True,
        )
        assert result.returncode in (0, 1), result.stderr
        return set(result.stdout.splitlines())

    return check


def test_all_manifest_evidence_files_can_be_added_to_git(ignored_paths):
    manifest = json.loads((ROOT / "evidence/nnunet/MANIFEST.json").read_text())
    paths = ["evidence/nnunet/MANIFEST.json"]
    paths.extend(f"evidence/nnunet/{name}" for name in manifest["files"])
    assert not ignored_paths(paths)


def test_evidence_exception_keeps_downloads_and_large_artifacts_ignored(ignored_paths):
    paths = [
        "datasets/example/summary.json",
        "data/example/summary.json",
        "artifacts/example/summary.json",
        "checkpoints/model.pth",
        "other/datasets/example/summary.json",
        "evidence/nnunet/datasets/example/model.pth",
        "evidence/nnunet/datasets/example/volume.nii.gz",
        "evidence/nnunet/datasets/example/probabilities.npz",
        "evidence/nnunet/datasets/example/cache.parquet",
        "evidence/nnunet/datasets/example/unpublished.json",
        "evidence/nnunet/datasets/example/nested/summary.json",
        "evidence/nnunet/datasets/summary.json",
    ]
    assert ignored_paths(paths) == set(paths)
