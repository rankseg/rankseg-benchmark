"""Locate the selected RankSEG package without requiring a sibling checkout."""
from importlib.util import find_spec
from pathlib import Path


def installed_rankseg_root():
    spec = find_spec("rankseg")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install RankSEG or pass an explicit --rankseg-path checkout")
    return Path(spec.origin).resolve().parent.parent
