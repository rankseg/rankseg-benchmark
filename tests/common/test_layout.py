"""Public entry points, import isolation, and packaged resources after migration."""
from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest

from rankseg_benchmark.cli import main


@pytest.mark.parametrize("legacy,canonical", [
    ("datasets", "quick.datasets"), ("metrics", "quick.metrics"),
    ("runner", "quick.runner"), ("timing", "common.timing"),
    ("monai_cache", "monai.cache"), ("demo", "monai.demo"),
])
def test_legacy_imports_are_the_same_module(legacy, canonical):
    assert import_module('rankseg_benchmark.' + legacy) is import_module('rankseg_benchmark.' + canonical)


@pytest.mark.parametrize("args,expected,absent", [
    (["quick", "--list-datasets"], "pascal_voc", "monai_btcv"),
    (["monai", "evaluate", "--list-datasets"], "monai_btcv", "pascal_voc"),
    (["--list-datasets"], "pascal_voc", "not-a-real-dataset"),
    (["monai", "cache", "--list"], "monai_btcv", "pascal_voc"),
])
def test_suite_routing_and_legacy_flags(args, expected, absent, capsys):
    assert main(args) == 0
    output = capsys.readouterr().out
    assert expected in output and absent not in output


@pytest.mark.parametrize("args", [
    ["quick", "--dataset", "monai_btcv_swin_v058_msd_pancreas"],
    ["monai", "evaluate", "--dataset", "pascal_voc"],
    ["unknown-suite"],
])
def test_wrong_suite_rejected_before_loading_data(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_help_and_quick_registry_do_not_import_optional_backends(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'monai', 'nnunet', 'nnunetv2', 'SimpleITK', 'nibabel', 'PIL'}:
            raise AssertionError('Eager optional import: ' + fullname)
sys.meta_path.insert(0, Block())
from rankseg_benchmark.cli import main
assert main(['--help']) == 0
assert main(['quick', '--list-datasets']) == 0
assert main(['monai', '--help']) == 0
try:
    main(['nnunet', '--help'])
except SystemExit as exc:
    assert exc.code == 0
'''
    result = subprocess.run([sys.executable, '-I', '-c', f'import sys; sys.path.insert(0, {str(root)!r});\n' + code],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_monai_resource_loads_without_checkout_working_directory(tmp_path, monkeypatch):
    from rankseg_benchmark.monai.cache import load_monai_specs
    monkeypatch.chdir(tmp_path)
    assert len(load_monai_specs()) == 2
