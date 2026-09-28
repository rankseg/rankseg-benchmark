"""Compatibility import; implementation lives in rankseg_benchmark.quick.runner."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("rankseg_benchmark.quick.runner")
_sys.modules[__name__] = _module
