"""Workspace relocation must not alter custom manifests or scientific inputs."""
from importlib import import_module
from pathlib import Path

import pytest

from rankseg_benchmark.nnunet.config import load_dataset_config
from rankseg_benchmark.nnunet.paths import repository_root, resolve_manifest_path, workspace_root


@pytest.mark.parametrize("module", ["config", "metrics", "aggregate", "evidence", "screening_benchmark"])
def test_old_namespace_aliases_canonical_module(module):
    assert import_module('rankseg_nnunet_bench.' + module) is import_module('rankseg_benchmark.nnunet.' + module)


def test_workspace_override_only_redirects_builtin_relative_data(tmp_path, monkeypatch):
    root = repository_root()
    monkeypatch.setenv('RANKSEG_NNUNET_WORKSPACE', str(tmp_path))
    base = root / 'configs/nnunet'
    for folder in ('work', 'artifacts', 'outputs', 'data'):
        assert resolve_manifest_path(base, f'../../{folder}/fixture') == tmp_path / folder / 'fixture'
    assert resolve_manifest_path(base, '../../evidence/nnunet') == root / 'evidence/nnunet'
    assert resolve_manifest_path(base, str(root / 'work/absolute')) == root / 'work/absolute'
    assert resolve_manifest_path(tmp_path / 'custom', '../work/data') == tmp_path / 'work/data'
    assert resolve_manifest_path(base, None) is None


def test_workspace_default_is_the_merged_checkout(monkeypatch):
    monkeypatch.delenv('RANKSEG_NNUNET_WORKSPACE', raising=False)
    assert workspace_root() == repository_root()


def test_manifest_workspace_resolution_from_an_installed_wheel(tmp_path, monkeypatch):
    from rankseg_benchmark.nnunet import paths
    checkout = repository_root()
    monkeypatch.setattr(paths, 'repository_root', lambda: tmp_path / 'site-packages')
    monkeypatch.setenv('RANKSEG_NNUNET_WORKSPACE', str(tmp_path / 'existing-workspace'))
    assert paths.resolve_manifest_path(checkout / 'configs/nnunet', '../../work/case') == (
        tmp_path / 'existing-workspace/work/case')


def test_all_dataset_manifests_load_with_workspace_override(tmp_path, monkeypatch):
    monkeypatch.setenv('RANKSEG_NNUNET_WORKSPACE', str(tmp_path))
    for manifest in (repository_root() / 'configs/nnunet').glob('*.yaml'):
        if manifest.name == 'full16_evidence.yaml':
            continue
        config = load_dataset_config(manifest)
        assert config.probabilities_dir.is_relative_to(tmp_path)
        assert config.labels_dir.is_relative_to(tmp_path)
        assert config.output_dir.is_relative_to(tmp_path)


def test_evidence_is_a_closed_byte_preserved_package():
    from rankseg_benchmark.nnunet.evidence import _evidence_readme
    import hashlib
    import json
    assert 'verify-evidence evidence/nnunet' in _evidence_readme(16, 2181)
    assert 'configs/nnunet/full16_evidence.yaml' in _evidence_readme(16, 2181)
    root = repository_root() / 'evidence/nnunet'
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    assert manifest['dataset_count'] == 16 and manifest['case_count'] == 2181
    for name, expected in manifest['files'].items():
        payload = (root / name).read_bytes()
        assert len(payload) == expected['size_bytes']
        assert hashlib.sha256(payload).hexdigest() == expected['sha256']
