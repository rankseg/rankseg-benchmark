"""Offline checks for the fixed, outcome-independent cache selection script."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def cache_script(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts" / "cache_screening_examples.py"
    spec = importlib.util.spec_from_file_location("cache_screening_examples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = io.BytesIO()
    pq.write_table(pa.table({"id": list(range(7))}), buffer, row_group_size=3)
    payload = buffer.getvalue()
    calls = []

    class OfflineFilesystem:
        def open(self, source, mode, **kwargs):
            calls.append(source)
            assert mode == "rb"
            assert f"@{module.REVISION}/" in source
            return io.BytesIO(payload)

    monkeypatch.setattr(module, "HfFileSystem", OfflineFilesystem)
    return module, calls


def invoke(module, monkeypatch, directory, count):
    monkeypatch.setattr(sys, "argv", ["cache", "--output-dir", str(directory),
                                    "--count", str(count), "--datasets", "pascal_voc"])
    module.main()


def test_fixed_prefix_spanning_row_groups(cache_script, monkeypatch, tmp_path):
    module, calls = cache_script
    invoke(module, monkeypatch, tmp_path, 5)
    path = tmp_path / "pascal_voc-first5.parquet"
    assert pq.read_table(path)["id"].to_pylist() == [0, 1, 2, 3, 4]
    metadata = json.loads(path.with_suffix(".source.json").read_text())
    assert metadata["rows"] == 5 and metadata["revision"] == module.REVISION
    assert metadata["source"] == calls[0]
    assert not path.with_suffix(".partial.parquet").exists()


@pytest.mark.parametrize("suffix", [".parquet", ".partial.parquet", ".source.json"])
def test_refuses_any_existing_cache_artifact(cache_script, monkeypatch, tmp_path, suffix):
    module, calls = cache_script
    path = tmp_path / f"pascal_voc-first5{suffix}"
    path.write_bytes(b"existing user artifact")
    with pytest.raises(FileExistsError):
        invoke(module, monkeypatch, tmp_path, 5)
    assert path.read_bytes() == b"existing user artifact"
    assert not calls


def test_short_source_is_not_published_as_complete(cache_script, monkeypatch, tmp_path):
    module, _ = cache_script
    with pytest.raises(ValueError, match="Only 7 rows available"):
        invoke(module, monkeypatch, tmp_path, 8)
    path = tmp_path / "pascal_voc-first8.parquet"
    assert not path.exists() and not path.with_suffix(".source.json").exists()
    assert pq.read_table(path.with_suffix(".partial.parquet"))["id"].to_pylist() == list(range(7))
