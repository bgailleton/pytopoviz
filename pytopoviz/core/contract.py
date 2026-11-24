"""Contract generation.

The contract is *generated* from the live registries, never authored (DESIGN.md §4).
It is the JSON a frontend reads to know every kind, type, converter and process,
with each interface member's ``kind`` injected from the TypeRegistry (and its
``doc`` too when the member has none of its own).

Envelope: ``{schema_version, kinds, types, converters, processes}``.

Author: B.G.
"""

from __future__ import annotations

import inspect

from typing import Dict, List, Optional, Sequence

from .converter import ConverterRegistry
from .kinds import DATA_KINDS, PARAM_KINDS
from .ports import Output, Param, Port
from .process import ProcessRegistry
from .runner import RUNNERS
from .types import TypeRegistry

SCHEMA_VERSION = "0.1.0"


def _type_field(types: TypeRegistry, type_ref: Sequence[str]) -> Dict:
    """Emit ``{type: [...], kind: ...}`` for an interface member's TypeRef."""
    ids = list(type_ref)
    kinds = {types.kind_of(t) for t in ids}
    kind = next(iter(kinds)) if len(kinds) == 1 else None
    field: Dict = {"type": ids, "kind": kind}
    return field


def _with_doc(types: TypeRegistry, d: Dict, member) -> Dict:
    """The member's doc, else its type's (single-type members only); and
    its label and unit when set."""
    for key in ("label", "unit", "show_if", "group"):
        if getattr(member, key, None):
            d[key] = getattr(member, key)
    doc = member.doc
    if not doc and len(member.types) == 1 and types.has(member.types[0]):
        doc = types.get(member.types[0]).doc
    d["doc"] = doc
    return d


def _port_dict(types: TypeRegistry, port: Port) -> Dict:
    d = _type_field(types, port.types)
    d.update(name=port.name, optional=port.optional)
    if port.arg is not None:
        d["arg"] = port.arg
    return _with_doc(types, d, port)


def _param_dict(types: TypeRegistry, param: Param) -> Dict:
    d = _type_field(types, param.types)
    d.update(name=param.name, optional=param.optional)
    if param.has_default:
        d["default"] = param.default_value
    if param.choices is not None:
        d["choices"] = list(param.choices)
    if param.choice_labels is not None:
        d["choice_labels"] = list(param.choice_labels)
    if param.min is not None:
        d["min"] = param.min
    if param.max is not None:
        d["max"] = param.max
    if param.arg is not None:
        d["arg"] = param.arg
    return _with_doc(types, d, param)


def _output_dict(types: TypeRegistry, out: Output) -> Dict:
    d = _type_field(types, out.types)
    d.update(name=out.name, optional=out.optional)
    if out.switch is not None:
        d["switch"] = out.switch
    return _with_doc(types, d, out)


def process_contract(types: TypeRegistry, proc) -> Dict:
    spec = proc.spec
    entry = {
        "id": spec.id,
        "label": spec.label,
        "description": spec.description,
        "category": spec.category,
        "callable": spec.impl,
        "describe": proc.describable,
        "inputs": [_port_dict(types, p) for p in spec.inputs],
        "params": [_param_dict(types, p) for p in spec.params],
        "outputs": [_output_dict(types, o) for o in spec.outputs],
    }
    if spec.id in RUNNERS:
        # A live runner exists (see core.runner): its params fixed at opening
        # and the axis of its coordinate.
        cls = RUNNERS[spec.id]
        entry["runner"] = {"config": list(cls.config), "axis": cls.axis, "actions": [
            {"name": name, "label": label,
             "point": "x" in inspect.signature(getattr(cls, name)).parameters,
             "doc": (inspect.getdoc(getattr(cls, name)) or "").split("\n\n")[0].replace("\n", " ")}
            for name, label in cls.actions.items()]}
    return entry


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
            {"type_id": t.type_id, "kind": t.kind, "doc": t.doc} for t in types
        ],
        "converters": [
            {"from": c.from_type, "to": c.to_type} for c in converters
        ],
        "processes": [process_contract(types, p) for p in processes
                      if _types_known(types, p.spec)],
    }


def _types_known(types: TypeRegistry, spec) -> bool:
    """Whether every type of ``spec``'s inputs, params and outputs is
    registered. A routine over a library whose adapter was skipped (not
    installed) names types nobody registered: it cannot run, and is left out
    of the contract."""
    members = list(spec.inputs) + list(spec.params) + list(spec.outputs)
    return all(types.has(t) for m in members for t in m.types)
