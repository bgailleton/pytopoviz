"""Adapter loading.

Each adapter module is a registration unit that hard-imports its science library
and registers types, converters and processes into the default core registries.
A missing library is skipped with a warning; a process it would have provided then
surfaces as an error at workflow validation (DESIGN.md §8).

Importing this package loads every available adapter.
"""

from __future__ import annotations

import importlib
import warnings

# Shared pytopoviz-conceptual types, registered before any library adapter so
# each adapter can reference them (e.g. field2d). Not guarded: a failure here is
# a real bug, not a missing optional dependency.
from . import _conceptual  # noqa: F401,E402

# Library adapter modules to load, in order. A missing library is skipped.
_ADAPTER_MODULES = [
    "geopandas",
    "topotoolbox",
    "pyfastflow",
    "lsdtt3",
    "dem_sources",
]

_loaded = []
_skipped = {}

for _name in _ADAPTER_MODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
        _loaded.append(_name)
    except ImportError as exc:  # missing science library
        _skipped[_name] = str(exc)
        warnings.warn(
            f"pytopoviz adapter {_name!r} skipped: {exc}", RuntimeWarning, stacklevel=2
        )


def loaded_adapters():
    return list(_loaded)


def skipped_adapters():
    return dict(_skipped)
