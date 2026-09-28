import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "auto_subset", Path(__file__).resolve().parents[2] / "scripts/nnunet/run_auto_acceptance_subset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("task", ["Task003_Liver", "Task007_Pancreas"])
@pytest.mark.parametrize("already_in_subset", [False, True])
def test_selection_is_predeclared_and_never_duplicates(task, already_in_subset):
    first = SimpleNamespace(case_id="first")
    extra = SimpleNamespace(case_id="liver_43")
    selected = [first, extra] if already_in_subset else [first]
    calls = []
    def prepare(config, count):
        calls.append(count)
        return (selected if count else [first, extra]), {"oof": True}, 131
    result, audit, available = module.select_with_regression(prepare, SimpleNamespace(dataset_id=task), 2)
    assert available == 131 and audit == {"oof": True}
    assert result == ([first, extra] if task == "Task003_Liver" else selected)
    assert calls == ([2, 0] if task == "Task003_Liver" else [2])
    assert selected == ([first, extra] if already_in_subset else [first])


def test_missing_large_regression_is_not_silently_skipped():
    with pytest.raises(ValueError, match="liver_43"):
        module.select_with_regression(lambda config, count: ([], {}, 131),
                                      SimpleNamespace(dataset_id="Task003_Liver"), 2)
