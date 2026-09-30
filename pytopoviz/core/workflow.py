"""Workflow DAG: JSON document, validation, and headless runner.

Document (DESIGN.md §7)::

    {
      "version": 1,
      "inputs":  { name: {type, default?, prompt?, required?} },
      "nodes":   [ {id, process, params, inputs} ],
      "outputs": { name: "node.port" }
    }

- ``params`` value = literal or ``{"$ref": input_name}``.
- ``inputs`` (per port) = ``"node.port"`` (edge) or ``{"$ref": input_name}``.
  A port has exactly one source; there is no separate ``edges`` list.
- Loaders and figures are plain processes — no special sections.

Runner: ``Workflow.from_json(doc).run(args, session=None) -> {name: handle}``:
validate -> topo-sort -> sequential execution through a Session.

Author: B.G.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .converter import ConverterRegistry
from .errors import ValidationError, WorkflowError
from .process import ProcessRegistry
from .session import DataHandle, Session
from .types import TypeRegistry


def _split_ref(ref: str) -> Tuple[str, str]:
    parts = ref.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValidationError(f"edge {ref!r} must be 'node.port'")
    return parts[0], parts[1]


def _is_ref(value) -> bool:
    return isinstance(value, dict) and set(value.keys()) == {"$ref"}


@dataclass
class InputDecl:
    name: str
    type: str
    default: object = None
    has_default: bool = False
    prompt: Optional[str] = None
    required: bool = False


@dataclass
class Node:
    id: str
    process: str
    params: Dict[str, object] = field(default_factory=dict)
    inputs: Dict[str, object] = field(default_factory=dict)


class Workflow:
    def __init__(
        self,
        version: int,
        inputs: Dict[str, InputDecl],
        nodes: List[Node],
        outputs: Dict[str, str],
        types: TypeRegistry,
        converters: ConverterRegistry,
        processes: ProcessRegistry,
    ) -> None:
        self.version = version
        self.inputs = inputs
        self.nodes = nodes
        self.outputs = outputs
        self._types = types
        self._converters = converters
        self._processes = processes
        self._by_id: Dict[str, Node] = {n.id: n for n in nodes}
        self.validate()

    # ---- construction -------------------------------------------------

    @classmethod
    def from_json(
        cls,
        doc: Dict,
        types: Optional[TypeRegistry] = None,
        converters: Optional[ConverterRegistry] = None,
        processes: Optional[ProcessRegistry] = None,
    ) -> "Workflow":
        from . import registries as reg

        types = types or reg.TYPES
        converters = converters or reg.CONVERTERS
        processes = processes or reg.PROCESSES

        inputs: Dict[str, InputDecl] = {}
        for name, spec in (doc.get("inputs") or {}).items():
            if "type" not in spec:
                raise ValidationError(f"input {name!r} missing 'type'")
            inputs[name] = InputDecl(
                name=name,
                type=spec["type"],
                default=spec.get("default"),
                has_default="default" in spec,
                prompt=spec.get("prompt"),
                required=bool(spec.get("required", False)),
            )

        nodes: List[Node] = []
        seen = set()
        for raw in doc.get("nodes") or []:
            if "id" not in raw or "process" not in raw:
                raise ValidationError("each node needs 'id' and 'process'")
            nid = raw["id"]
            if nid in seen:
                raise ValidationError(f"duplicate node id {nid!r}")
            seen.add(nid)
            nodes.append(
                Node(
                    id=nid,
                    process=raw["process"],
                    params=dict(raw.get("params") or {}),
                    inputs=dict(raw.get("inputs") or {}),
                )
            )

        outputs = dict(doc.get("outputs") or {})
        return cls(
            version=int(doc.get("version", 1)),
            inputs=inputs,
            nodes=nodes,
            outputs=outputs,
            types=types,
            converters=converters,
            processes=processes,
        )

    # ---- validation ---------------------------------------------------

    def _accepted_or_convertible(
        self, src_types: Sequence[str], accepted: Sequence[str]
    ) -> bool:
        """True if every src type is accepted by, or convertible into, ``accepted``."""
        for s in src_types:
            if s in accepted:
                continue
            if any(self._converters.has(s, a) for a in accepted):
                continue
            return False
        return True

    def validate(self) -> None:
        for node in self.nodes:
            # (1) process ids resolve
            if not self._processes.has(node.process):
                raise ValidationError(
                    f"node {node.id!r}: unknown process {node.process!r}"
                )
            spec = self._processes.get(node.process).spec

            declared_params = {p.name for p in spec.params}
            declared_ports = {p.name for p in spec.inputs}

            # (2a) node param/port names exist
            for pname in node.params:
                if pname not in declared_params:
                    raise ValidationError(
                        f"node {node.id!r}: unknown param {pname!r} for {node.process!r}"
                    )
            for pname in node.inputs:
                if pname not in declared_ports:
                    raise ValidationError(
                        f"node {node.id!r}: unknown input port {pname!r} for {node.process!r}"
                    )

            # (2b) required ports bound
            for port in spec.inputs:
                if not port.optional and port.name not in node.inputs:
                    raise ValidationError(
                        f"node {node.id!r}: required input port {port.name!r} not bound"
                    )
            # (2c) required params present or defaulted
            for param in spec.params:
                if param.name in node.params:
                    continue
                if param.optional or param.has_default:
                    continue
                raise ValidationError(
                    f"node {node.id!r}: required param {param.name!r} missing"
                )

            # (6) param literal values satisfy type/constraints; (4) param $ref
            for param in spec.params:
                if param.name not in node.params:
                    continue
                value = node.params[param.name]
                if _is_ref(value):
                    self._check_param_ref(node, param, value["$ref"])
                else:
                    self._check_param_literal(node, param, value)

            # (3)/(4) input port sources
            for port in spec.inputs:
                if port.name not in node.inputs:
                    continue
                source = node.inputs[port.name]
                if _is_ref(source):
                    self._check_port_ref(node, port, source["$ref"])
                elif isinstance(source, str):
                    self._check_edge(node, port, source)
                else:
                    raise ValidationError(
                        f"node {node.id!r} input {port.name!r}: source must be "
                        f"'node.port' or {{'$ref': ...}}"
                    )

        # outputs reference existing node.port
        for oname, ref in self.outputs.items():
            up_id, up_port = _split_ref(ref)
            self._check_upstream_port(f"output {oname!r}", up_id, up_port)

        # (5) acyclic
        self._topo_order()

    def _check_param_literal(self, node, param, value) -> None:
        ok = False
        for tid in param.types:
            spec = self._types.get(tid)
            if spec.matches(value):
                ok = True
                break
        if not ok:
            raise ValidationError(
                f"node {node.id!r} param {param.name!r}: value {value!r} does not "
                f"match type(s) {list(param.types)}"
            )
        if param.choices is not None and value not in param.choices:
            raise ValidationError(
                f"node {node.id!r} param {param.name!r}: {value!r} not in {list(param.choices)}"
            )
        if param.min is not None and value < param.min:
            raise ValidationError(
                f"node {node.id!r} param {param.name!r}: {value!r} < min {param.min}"
            )
        if param.max is not None and value > param.max:
            raise ValidationError(
                f"node {node.id!r} param {param.name!r}: {value!r} > max {param.max}"
            )

    def _check_param_ref(self, node, param, input_name) -> None:
        if input_name not in self.inputs:
            raise ValidationError(
                f"node {node.id!r} param {param.name!r}: unknown $ref {input_name!r}"
            )
        decl = self.inputs[input_name]
        if decl.type not in param.types:
            raise ValidationError(
                f"node {node.id!r} param {param.name!r}: input {input_name!r} type "
                f"{decl.type!r} not in {list(param.types)}"
            )

    def _check_port_ref(self, node, port, input_name) -> None:
        if input_name not in self.inputs:
            raise ValidationError(
                f"node {node.id!r} input {port.name!r}: unknown $ref {input_name!r}"
            )
        decl = self.inputs[input_name]
        if not self._types.has(decl.type):
            raise ValidationError(
                f"input {input_name!r}: unknown type {decl.type!r}"
            )
        if not self._accepted_or_convertible((decl.type,), port.types):
            raise ValidationError(
                f"node {node.id!r} input {port.name!r}: input {input_name!r} type "
                f"{decl.type!r} is not accepted by or convertible into {list(port.types)}"
            )

    def _check_edge(self, node, port, source) -> None:
        up_id, up_port = _split_ref(source)
        out = self._check_upstream_port(
            f"node {node.id!r} input {port.name!r}", up_id, up_port
        )
        if not self._accepted_or_convertible(out.types, port.types):
            raise ValidationError(
                f"node {node.id!r} input {port.name!r}: {source} type(s) {list(out.types)} "
                f"not accepted by or convertible into {list(port.types)}"
            )

    def _check_upstream_port(self, where, up_id, up_port):
        if up_id not in self._by_id:
            raise ValidationError(f"{where}: unknown node {up_id!r}")
        up_spec = self._processes.get(self._by_id[up_id].process).spec
        try:
            return up_spec.output(up_port)
        except KeyError:
            raise ValidationError(
                f"{where}: node {up_id!r} has no output {up_port!r}"
            )

    def _topo_order(self) -> List[Node]:
        # edges: node -> set of upstream node ids
        deps: Dict[str, set] = {n.id: set() for n in self.nodes}
        for node in self.nodes:
            for source in node.inputs.values():
                if isinstance(source, str):
                    up_id, _ = _split_ref(source)
                    deps[node.id].add(up_id)
        order: List[Node] = []
        resolved: set = set()
        pending = list(self.nodes)
        while pending:
            progressed = False
            for node in list(pending):
                if deps[node.id] <= resolved:
                    order.append(node)
                    resolved.add(node.id)
                    pending.remove(node)
                    progressed = True
            if not progressed:
                stuck = [n.id for n in pending]
                raise ValidationError(f"workflow has a cycle among nodes {stuck}")
        return order

    # ---- execution ----------------------------------------------------

    def run(
        self, args: Optional[Dict[str, object]] = None, session: Optional[Session] = None
    ) -> Dict[str, DataHandle]:
        args = args or {}
        if session is None:
            session = Session(self._types, self._converters)

        input_values = self._resolve_inputs(args)
        produced: Dict[Tuple[str, str], DataHandle] = {}

        for node in self._topo_order():
            proc = self._processes.get(node.process)
            spec = proc.spec
            kwargs: Dict[str, object] = {}

            for param in spec.params:
                if param.name in node.params:
                    raw = node.params[param.name]
                    value = input_values[raw["$ref"]] if _is_ref(raw) else raw
                elif param.has_default:
                    value = param.default_value
                else:
                    continue
                kwargs[param.target] = value

            for port in spec.inputs:
                if port.name not in node.inputs:
                    continue
                source = node.inputs[port.name]
                if _is_ref(source):
                    value = input_values[source["$ref"]]
                else:
                    up_id, up_port = _split_ref(source)
                    if (up_id, up_port) not in produced:
                        # An optional upstream output that was not produced.
                        if port.optional:
                            continue
                        raise WorkflowError(
                            f"node {node.id!r} input {port.name!r}: "
                            f"{up_id}.{up_port} was not produced"
                        )
                    handle = produced[(up_id, up_port)]
                    value = session.get(handle)
                coerced, _ = self._converters.resolve(value, port.types)
                kwargs[port.target] = coerced

            result = proc(**kwargs)
            for oname, value in result.items():
                out = spec.output(oname)
                handle = session.put(
                    value,
                    type_id=out.types[0] if len(out.types) == 1 else None,
                    name=f"{node.id}.{oname}",
                    provenance=node.id,
                )
                produced[(node.id, oname)] = handle

        # An optional output a node did not produce is left out of the results.
        results: Dict[str, DataHandle] = {}
        for oname, ref in self.outputs.items():
            up_id, up_port = _split_ref(ref)
            if (up_id, up_port) in produced:
                results[oname] = produced[(up_id, up_port)]
        return results

    def _resolve_inputs(self, args: Dict[str, object]) -> Dict[str, object]:
        values: Dict[str, object] = {}
        for name, decl in self.inputs.items():
            if name in args:
                values[name] = args[name]
            elif decl.has_default:
                values[name] = decl.default
            elif decl.required:
                raise WorkflowError(f"required input {name!r} not provided")
            else:
                values[name] = None
        return values
