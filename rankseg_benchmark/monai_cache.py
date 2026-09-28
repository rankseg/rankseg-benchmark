"""Compatibility import; implementation lives in rankseg_benchmark.monai.cache."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("rankseg_benchmark.monai.cache")
_sys.modules[__name__] = _module

if __name__ == "__main__":
    raise SystemExit(_module.main())
