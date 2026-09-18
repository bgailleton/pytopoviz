"""Contract generation.

The contract is *generated* from the live registries, never authored (DESIGN.md §4).
It is the JSON a frontend reads to know every kind, type, converter and process,
with each interface member's ``kind`` injected from the TypeRegistry.

Envelope: ``{schema_version, kinds, types, converters, processes}``.

Author: B.G.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .converter import ConverterRegistry
from .kinds import DATA_KINDS, PARAM_KINDS
from .ports import Output, Param, Port
from .process import ProcessRegistry
from .types import TypeRegistry

SCHEMA_VERSION = "0.1.0"


def _type_field(types: TypeRegistry, type_ref: Sequence[str]) -> Dict:
    """Emit ``{type: [...], kind: ...}`` for an interface member's TypeRef."""
    ids = list(type_ref)
    kinds = {types.kind_of(t) for t in ids}
    kind = next(iter(kinds)) if len(kinds) == 1 else None
    field: Dict = {"type": ids, "kind": kind}
    return field


def _port_dict(types: TypeRegistry, port: Port) -> Dict:
    d = _type_field(types, port.types)
    d.update(name=port.name, optional=port.optional)
    if port.arg is not None:
        d["arg"] = port.arg
    return d


def _param_dict(types: TypeRegistry, param: Param) -> Dict:
    d = _type_field(types, param.types)
    d.update(name=param.name, optional=param.optional)
    if param.has_default:
        d["default"] = param.default_value
    if param.choices is not None:
        d["choices"] = list(param.choices)
    if param.min is not None:
        d["min"] = param.min
    if param.max is not None:
        d["max"] = param.max
    if param.arg is not None:
        d["arg"] = param.arg
    return d


def _output_dict(types: TypeRegistry, out: Output) -> Dict:
    d = _type_field(types, out.types)
    d["name"] = out.name
    return d


def process_contract(types: TypeRegistry, proc) -> Dict:
    spec = proc.spec
    return {
        "id": spec.id,
        "label": spec.label,
        "callable": spec.impl,
        "inputs": [_port_dict(types, p) for p in spec.inputs],
        "params": [_param_dict(types, p) for p in spec.params],
        "outputs": [_output_dict(types, o) for o in spec.outputs],
    }


def build_contract(
    types: TypeRegistry,
    converters: ConverterRegistry,
    processes: ProcessRegistry,
) -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kinds": {
            "param": sorted(PARAM_KINDS),
            "data": sorted(DATA_KINDS),
        },
        "types": [
            {"type_id": t.type_id, "kind": t.kind} for t in types
        ],
        "converters": [
            {"from": c.from_type, "to": c.to_type} for c in converters
        ],
        "processes": [process_contract(types, p) for p in processes],
    }
