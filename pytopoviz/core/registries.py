"""Global default registries.

The framework exposes registry *classes* for isolated instances, plus one shared
default set that decorators and adapters use out of the box (DESIGN.md §2).

The default ProcessRegistry is owned by :mod:`pytopoviz.core.process` (so the
``@process`` decorator can default to it without an import cycle) and re-exported
here as ``PROCESSES``.

Author: B.G.
"""

from __future__ import annotations

from .converter import ConverterRegistry
from .process import DEFAULT_PROCESSES as PROCESSES
from .types import TypeRegistry

# The default triplet. Adapters register into these on import.
TYPES = TypeRegistry()
CONVERTERS = ConverterRegistry(TYPES)


def register_type(type_id, kind, check, doc="", codec=None):
    """Register a type in the default TypeRegistry."""
    return TYPES.register_type(type_id, kind, check, doc, codec)


def register_converter(from_type, to_type, fn):
    """Register a converter in the default ConverterRegistry."""
    return CONVERTERS.register_converter(from_type, to_type, fn)


def reset_defaults() -> None:
    """Empty the default registries in place. Test helper; not for normal use."""
    PROCESSES.clear()
    CONVERTERS.clear()
    TYPES.clear()
