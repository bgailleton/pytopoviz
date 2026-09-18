"""pytopoviz framework core.

Imports nothing from the rest of pytopoviz and no science library. Everything the
machinery needs lives here; adapters/routines/figmaker depend on this package.

Author: B.G.
"""

from __future__ import annotations

from .kinds import DATA_KINDS, KINDS, PARAM_KINDS, is_data_kind, is_kind, is_param_kind
from .errors import (
    ConversionError,
    PytopovizError,
    RegistrationError,
    SessionError,
    ValidationError,
    WorkflowError,
)
from .types import TypeRegistry, TypeSpec
from .ports import Output, Param, Port
from .process import (
    Process,
    ProcessRegistry,
    ProcessSpec,
    make_spec,
    process,
    validate_spec_types,
)
from .converter import ConverterRegistry, ConverterSpec
from .session import DataHandle, Session
from .contract import SCHEMA_VERSION, build_contract, process_contract
from .workflow import Workflow

from . import registries
from .registries import (
    CONVERTERS,
    PROCESSES,
    TYPES,
    register_converter,
    register_type,
    reset_defaults,
)

# Register built-in primitive param types into the default registry.
from . import primitives  # noqa: F401  (import for side effect)

__all__ = [
    # kinds
    "PARAM_KINDS", "DATA_KINDS", "KINDS", "is_param_kind", "is_data_kind", "is_kind",
    # errors
    "PytopovizError", "RegistrationError", "ConversionError", "ValidationError",
    "SessionError", "WorkflowError",
    # types
    "TypeRegistry", "TypeSpec",
    # interface
    "Port", "Param", "Output",
    # process
    "process", "Process", "ProcessSpec", "ProcessRegistry", "make_spec",
    "validate_spec_types",
    # converter
    "ConverterRegistry", "ConverterSpec",
    # session
    "Session", "DataHandle",
    # contract
    "build_contract", "process_contract", "SCHEMA_VERSION",
    # workflow
    "Workflow",
    # default registries
    "TYPES", "CONVERTERS", "PROCESSES", "registries",
    "register_type", "register_converter", "reset_defaults",
]
