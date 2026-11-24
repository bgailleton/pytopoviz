"""Workflow DAG: JSON document, validation, and headless runner.

Document (DESIGN.md §7)::

    {
      "version": 2,
      "requires": [name, ...],
      "inputs":  { name: {type, default?, prompt?, required?} },
      "nodes":   [ {id, process, params, inputs, keep?, runner?} ],
      "zones":   [ {id, kind, out?, ...} ],
      "outputs": { name: "node.port" },
      "ui":      { ... }
    }

- ``params`` value = literal, ``{"$ref": input_name}`` or ``{"$from":
  "node.port"}`` (a node's output, e.g. a number, feeds the param).
- ``inputs`` (per port) = ``"node.port"`` (edge) or ``{"$ref": input_name}``.
  A port has exactly one source; there is no separate ``edges`` list.
- ``keep``: the node's outputs are returned too (``execute``), not only the
  declared ``outputs``.
- ``requires``: what the workflow needs beyond pytopoviz (e.g. ``"topos"``:
  processes only a host provides). A host declares what it provides with
  ``provide(name)``; a workflow whose requirements are not all provided is
  refused, naming them. Empty: runnable with pytopoviz alone.
- ``ui``: a frontend's own state (node positions...), kept as is, never read.
- An input with ``many: true`` is bound to a list of values of its type (a
  For each zone runs over them).
- ``runner`` (a node of a process with a live runner, core.runner):
  ``{advances: N (1), until_stopped: false, record_every: 0, record_fields?}``:
  the runner is opened, advanced N times (fewer when ``until_stopped`` and
  its status says it stopped), closed; the node's outputs are the last
  advance's. ``record_every`` k > 0: every k-th advance is a frame of a
  series (``WorkflowResult.series[node id]``), holding ``record_fields``
  (default: its grid and field outputs).

Zones (loops, as Blender's): a zone is two sockets, its input ``id`` and its
output ``out`` (default ``id + "_out"``), referenced like nodes:
``"z.index"`` inside, ``"z_out.state"`` after. Its body is every node that
reads its input (directly or not) or that its output reads, but what the
zone itself reads from before it; it runs once per iteration. Zones nest.

- ``kind: "repeat"``: ``count`` (int, ``$ref`` or ``$from``) iterations.
  ``state: {name: {type, init}}``: ``init`` a source ("node.port" or
  ``{"$ref"}``); inside, ``"z.name"`` is the value of this iteration (init
  at the first), ``next: {name: "node.port"}`` the value the next one gets;
  ``"z_out.name"`` the last. ``"z.index"`` (int, from 0). A node of a
  runner process in its body keeps its runner across the iterations
  (opened at the first, ``update`` with new inputs, closed after the last).
- ``kind: "sweep"``: ``values: {start, stop, step}`` (stop included when
  hit) or ``{list: [...]}``; ``"z.value"``, ``"z.index"``; ``name``: the
  axis name (default "value").
- ``kind: "foreach"``: ``items``: a source holding several (a vector: each
  feature; a ``many`` input: each value); ``"z.item"``, ``"z.index"``.

``collect: {name: "node.port"}``: those values, after each iteration (a
repeat zone: every ``record_every``-th, default 1), are frames of a series on
the zone's axis (iteration, the swept value, item), in
``WorkflowResult.series[zone id]``. ``label``: the series' name.
- Loaders and figures are plain processes — no special sections.
- Version 1 documents are read as they are (no ``$from``, ``keep``,
  ``requires`` or ``ui``).

Runner: ``Workflow.from_json(doc).run(args, session=None) -> {name: handle}``:
validate -> topo-sort -> sequential execution through a Session.
``execute`` also takes ``until`` (run only what those nodes need) and
``on_status`` (per-node events), and returns the kept outputs as well.

Author: B.G.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import progress
from .converter import ConverterRegistry
from .errors import ValidationError, WorkflowError
from .ports import Output
from .process import ProcessRegistry
from .runner import RUNNERS
from .series import record
from .session import GROUP, DataHandle, Session
from .types import TypeRegistry


def _split_ref(ref: str) -> Tuple[str, str]:
    parts = ref.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValidationError(f"edge {ref!r} must be 'node.port'")
    return parts[0], parts[1]


def _is_ref(value) -> bool:
    return isinstance(value, dict) and set(value.keys()) == {"$ref"}


def _is_from(value) -> bool:
    return isinstance(value, dict) and set(value.keys()) == {"$from"}


#: What the running host provides beyond pytopoviz (``requires`` names).
PROVIDED: set = set()


def provide(name: str) -> None:
    """Declares that this process provides ``name`` (e.g. ``"topos"``), so
    workflows requiring it may run."""
    PROVIDED.add(str(name))


#: Current document version (``to_json`` writes it).
VERSION = 2


@dataclass
class InputDecl:
    name: str
    type: str
    default: object = None
    has_default: bool = False
    prompt: Optional[str] = None
    required: bool = False
    many: bool = False


@dataclass
class Node:
    id: str
    process: str
    params: Dict[str, object] = field(default_factory=dict)
    inputs: Dict[str, object] = field(default_factory=dict)
    keep: bool = False
    runner: Optional[Dict] = None


ZONE_KINDS = ("repeat", "sweep", "foreach")
#: Axis of a zone's series, per kind (a sweep's is its ``name``).
ZONE_AXES = {"repeat": "iteration", "foreach": "item"}


@dataclass
class Zone:
    """A loop (see the module doc); ``raw`` its document entry."""

    id: str
    kind: str
    out: str
    raw: Dict

    @property
    def state(self) -> Dict[str, Dict]:
        return dict(self.raw.get("state") or {}) if self.kind == "repeat" else {}

    @property
    def next(self) -> Dict[str, str]:
        return dict(self.raw.get("next") or {}) if self.kind == "repeat" else {}

    @property
    def collect(self) -> Dict[str, str]:
        return dict(self.raw.get("collect") or {})

    @property
    def axis(self) -> str:
        if self.kind == "sweep":
            return str(self.raw.get("name") or "value")
        return ZONE_AXES[self.kind]

    @property
    def record_every(self) -> int:
        return int(self.raw.get("record_every", 1)) if self.kind == "repeat" else 1


@dataclass
class WorkflowResult:
    """``outputs``: declared output name -> handle (those produced);
    ``kept``: ``"node.port"`` -> handle, every output of the kept nodes and
    of the ``until`` nodes (a node in a zone: its last iteration's);
    ``series``: zone or runner node id -> series (a group with an axis) of
    what it recorded."""

    outputs: Dict[str, DataHandle]
    kept: Dict[str, DataHandle]
    series: Dict[str, DataHandle] = field(default_factory=dict)


class _Socket:
    """A zone socket's port, typed as an Output is."""

    def __init__(self, name: str, types: Sequence[str]) -> None:
        self.name, self.types = name, tuple(types)


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
        requires: Sequence[str] = (),
        ui: Optional[Dict] = None,
        check: bool = True,
        zones: Optional[List[Zone]] = None,
    ) -> None:
        self.version = version
        self.inputs = inputs
        self.nodes = nodes
        self.zones = list(zones or [])
        self.outputs = outputs
        self.requires = [str(r) for r in requires]
        self.ui = dict(ui or {})
        self._types = types
        self._converters = converters
        self._processes = processes
        self._by_id: Dict[str, Node] = {n.id: n for n in nodes}
        self._zones: Dict[str, Zone] = {z.id: z for z in self.zones}
        self._zone_out: Dict[str, Zone] = {z.out: z for z in self.zones}
        self._structure_cache = None
        if check:
            self.validate()

    # ---- construction -------------------------------------------------

    @classmethod
    def from_json(
        cls,
        doc: Dict,
        types: Optional[TypeRegistry] = None,
        converters: Optional[ConverterRegistry] = None,
        processes: Optional[ProcessRegistry] = None,
        check: bool = True,
    ) -> "Workflow":
        """The workflow ``doc`` describes; ``check``: validated (False: built
        as it is, ``problems()`` lists what is wrong)."""
        from . import registries as reg

        types = reg.TYPES if types is None else types
        converters = reg.CONVERTERS if converters is None else converters
        processes = reg.PROCESSES if processes is None else processes

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
                many=bool(spec.get("many", False)),
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
                    keep=bool(raw.get("keep", False)),
                    runner=dict(raw["runner"]) if isinstance(raw.get("runner"), dict) else None,
                )
            )

        zones: List[Zone] = []
        for raw in doc.get("zones") or []:
            if not isinstance(raw, dict) or "id" not in raw or "kind" not in raw:
                raise ValidationError("each zone needs 'id' and 'kind'")
            zid = str(raw["id"])
            out = str(raw.get("out") or zid + "_out")
            for name in (zid, out):
                if name in seen:
                    raise ValidationError(f"duplicate node or zone id {name!r}")
                seen.add(name)
            zones.append(Zone(id=zid, kind=str(raw["kind"]), out=out, raw=dict(raw)))

        outputs = dict(doc.get("outputs") or {})
        version = int(doc.get("version", 1))
        if version > VERSION:
            raise ValidationError(f"workflow version {version} is newer than this pytopoviz reads ({VERSION})")
        requires = doc.get("requires") or []
        if not isinstance(requires, list) or not all(isinstance(r, str) for r in requires):
            raise ValidationError("'requires' must be a list of names")
        return cls(
            version=version,
            inputs=inputs,
            nodes=nodes,
            outputs=outputs,
            types=types,
            converters=converters,
            processes=processes,
            requires=requires,
            ui=doc.get("ui"),
            check=check,
            zones=zones,
        )

    def to_json(self, ui: bool = True) -> Dict:
        """The document (current version); ``ui=False`` leaves the frontend
        state out (an export)."""
        doc: Dict = {"version": VERSION}
        if self.requires:
            doc["requires"] = list(self.requires)
        doc["inputs"] = {}
        for name, d in self.inputs.items():
            entry: Dict = {"type": d.type}
            if d.has_default:
                entry["default"] = d.default
            if d.prompt is not None:
                entry["prompt"] = d.prompt
            if d.required:
                entry["required"] = True
            if d.many:
                entry["many"] = True
            doc["inputs"][name] = entry
        doc["nodes"] = []
        for n in self.nodes:
            entry = {"id": n.id, "process": n.process, "params": dict(n.params),
                     "inputs": dict(n.inputs)}
            if n.keep:
                entry["keep"] = True
            if n.runner is not None:
                entry["runner"] = dict(n.runner)
            doc["nodes"].append(entry)
        if self.zones:
            doc["zones"] = [dict(z.raw, id=z.id, kind=z.kind, out=z.out) for z in self.zones]
        doc["outputs"] = dict(self.outputs)
        if ui and self.ui:
            doc["ui"] = dict(self.ui)
        return doc

    # ---- validation ---------------------------------------------------

    def _accepted_or_convertible(
        self, src_types: Sequence[str], accepted: Sequence[str], widen: bool = False
    ) -> bool:
        """True if every src type is accepted by, or convertible into,
        ``accepted``; ``widen``: an int is accepted where a float is."""
        for s in src_types:
            if s in accepted:
                continue
            if widen and s == "int" and "float" in accepted:
                continue
            if any(self._converters.has(s, a) for a in accepted):
                continue
            return False
        return True

    def validate(self) -> None:
        """Raises ValidationError with the first of ``problems()``."""
        for problem in self.problems():
            raise ValidationError(problem["message"])

    def problems(self) -> List[Dict]:
        """Everything wrong with the document, each ``{node, where, field,
        message}``: ``node`` the node id (None: the workflow as a whole),
        ``where`` "param" / "input" / "output" / None, ``field`` the param,
        port or output name (None: the node as a whole)."""
        out: List[Dict] = []

        def add(node, where, fld, message):
            out.append({"node": node, "where": where, "field": fld, "message": str(message)})

        def check(node, where, fld, fn, *args):
            try:
                fn(*args)
            except ValidationError as exc:
                add(node, where, fld, exc)

        missing = [r for r in self.requires if r not in PROVIDED]
        if missing:
            add(None, None, None,
                f"this workflow needs {', '.join(missing)}, not provided here "
                f"(it does not run with pytopoviz alone)")
        for node in self.nodes:
            # (1) process ids resolve
            if not self._processes.has(node.process):
                add(node.id, None, None, f"node {node.id!r}: unknown process {node.process!r}")
                continue
            spec = self._processes.get(node.process).spec

            declared_params = {p.name for p in spec.params}
            declared_ports = {p.name for p in spec.inputs}

            # (2a) node param/port names exist
            for pname in node.params:
                if pname not in declared_params:
                    add(node.id, "param", pname,
                        f"node {node.id!r}: unknown param {pname!r} for {node.process!r}")
            for pname in node.inputs:
                if pname not in declared_ports:
                    add(node.id, "input", pname,
                        f"node {node.id!r}: unknown input port {pname!r} for {node.process!r}")

            # (2b) required ports bound
            for port in spec.inputs:
                if not port.optional and port.name not in node.inputs:
                    add(node.id, "input", port.name,
                        f"node {node.id!r}: required input port {port.name!r} not bound")
            # (2c) required params present or defaulted
            for param in spec.params:
                if param.name in node.params or param.optional or param.has_default:
                    continue
                add(node.id, "param", param.name,
                    f"node {node.id!r}: required param {param.name!r} missing")

            # (2d) the runner section
            if node.runner is not None:
                check(node.id, "runner", None, self._check_runner, node, spec)

            # (6) param literal values satisfy type/constraints; (4) param $ref / $from
            for param in spec.params:
                if param.name not in node.params:
                    continue
                value = node.params[param.name]
                if _is_ref(value):
                    check(node.id, "param", param.name, self._check_param_ref, node, param, value["$ref"])
                elif _is_from(value):
                    check(node.id, "param", param.name, self._check_param_edge, node, param, value["$from"])
                else:
                    check(node.id, "param", param.name, self._check_param_literal, node, param, value)

            # (3)/(4) input port sources
            for port in spec.inputs:
                if port.name not in node.inputs:
                    continue
                source = node.inputs[port.name]
                if _is_ref(source):
                    check(node.id, "input", port.name, self._check_port_ref, node, port, source["$ref"])
                elif isinstance(source, str):
                    check(node.id, "input", port.name, self._check_edge, node, port, source)
                else:
                    add(node.id, "input", port.name,
                        f"node {node.id!r} input {port.name!r}: source must be "
                        f"'node.port' or {{'$ref': ...}}")

        for zone in self.zones:
            for problem in self._zone_problems(zone):
                add(zone.id, *problem)

        # outputs reference existing node.port
        for oname, ref in self.outputs.items():
            try:
                up_id, up_port = _split_ref(ref)
                self._check_upstream_port(f"output {oname!r}", up_id, up_port)
            except ValidationError as exc:
                add(None, "output", oname, exc)

        # (5) acyclic, zones well formed
        check(None, None, None, self._structure)
        return out

    def _check_runner(self, node, spec) -> None:
        cfg = node.runner
        if node.process not in RUNNERS:
            raise ValidationError(f"node {node.id!r}: {node.process!r} has no live runner")
        unknown = set(cfg) - {"advances", "until_stopped", "record_every", "record_fields"}
        if unknown:
            raise ValidationError(f"node {node.id!r} runner: unknown {sorted(unknown)}")
        for key, low in (("advances", 1), ("record_every", 0)):
            value = cfg.get(key, low)
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValidationError(f"node {node.id!r} runner: {key} must be an int >= {low}")
        names = {o.name for o in spec.outputs}
        fields = cfg.get("record_fields")
        if fields is not None and (not isinstance(fields, list) or not set(fields) <= names):
            raise ValidationError(
                f"node {node.id!r} runner: record_fields must be outputs of {node.process!r}")

    def _zone_problems(self, zone: Zone) -> List[Tuple]:
        """``(where, field, message)`` for each thing wrong with ``zone``."""
        out: List[Tuple] = []
        z = zone.raw

        def bad(where, fld, message):
            out.append((where, fld, f"zone {zone.id!r}: {message}"))

        def source(where, fld, src, types):
            """``src`` (a "node.port" or {"$ref"}) gives one of ``types``."""
            try:
                got = self._source_types(src)
            except ValidationError as exc:
                bad(where, fld, exc)
                return
            if not self._accepted_or_convertible(got, types, widen=True):
                bad(where, fld, f"{src} type(s) {list(got)} not accepted by or convertible "
                                f"into {list(types)}")

        if zone.kind not in ZONE_KINDS:
            bad(None, None, f"kind must be one of {list(ZONE_KINDS)}, got {zone.kind!r}")
            return out
        for name in zone.state:
            if name == "index":
                bad("state", name, "a state cannot be named 'index' (the iteration's socket)")
        if zone.kind == "repeat":
            count = z.get("count")
            if _is_ref(count) or _is_from(count):
                source("param", "count", count if _is_ref(count) else count["$from"], ("int",))
            elif isinstance(count, bool) or not isinstance(count, int) or count < 0:
                bad("param", "count", "count must be an int >= 0, a $ref or a $from")
            every = z.get("record_every", 1)
            if isinstance(every, bool) or not isinstance(every, int) or every < 1:
                bad("param", "record_every", "record_every must be an int >= 1")
            for name, decl in zone.state.items():
                if not isinstance(decl, dict) or "type" not in decl or "init" not in decl:
                    bad("state", name, f"state {name!r} needs 'type' and 'init'")
                    continue
                if not self._types.has(decl["type"]):
                    bad("state", name, f"state {name!r}: unknown type {decl['type']!r}")
                    continue
                source("state", name, decl["init"], (decl["type"],))
                if name not in zone.next:
                    bad("state", name, f"state {name!r} has no 'next'")
                elif not isinstance(zone.next[name], str):
                    bad("state", name, f"state {name!r}: 'next' must be 'node.port'")
                else:
                    source("state", name, zone.next[name], (decl["type"],))
            for name in zone.next:
                if name not in zone.state:
                    bad("state", name, f"'next' {name!r} is not a state")
        elif zone.kind == "sweep":
            try:
                _sweep_values(z.get("values"))
            except ValidationError as exc:
                bad("param", "values", f"values: {exc}")
        else:
            items = z.get("items")
            if items is None:
                bad("input", "items", "'items' is not bound")
            else:
                try:
                    self._item_types(zone)
                except ValidationError as exc:
                    bad("input", "items", exc)
        for name, ref in zone.collect.items():
            if not isinstance(ref, str):
                bad("collect", name, f"collect {name!r} must be 'node.port'")
                continue
            try:
                self._source_types(ref)
            except ValidationError as exc:
                bad("collect", name, exc)
        return out

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

    def _check_param_edge(self, node, param, source) -> None:
        if not isinstance(source, str):
            raise ValidationError(
                f"node {node.id!r} param {param.name!r}: $from must be 'node.port'"
            )
        up_id, up_port = _split_ref(source)
        out = self._check_upstream_port(
            f"node {node.id!r} param {param.name!r}", up_id, up_port
        )
        if not self._accepted_or_convertible(out.types, param.types, widen=True):
            raise ValidationError(
                f"node {node.id!r} param {param.name!r}: {source} type(s) {list(out.types)} "
                f"not accepted by or convertible into {list(param.types)}"
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
        """The output (or zone socket) ``up_id.up_port`` names."""
        if up_id in self._zones or up_id in self._zone_out:
            zone = self._zones.get(up_id) or self._zone_out[up_id]
            ports = self._zone_ports(zone) if up_id == zone.id else self._zone_out_ports(zone)
            if up_port not in ports:
                raise ValidationError(f"{where}: zone {up_id!r} has no socket {up_port!r}")
            return _Socket(up_port, ports[up_port])
        if up_id not in self._by_id:
            raise ValidationError(f"{where}: unknown node {up_id!r}")
        if not self._processes.has(self._by_id[up_id].process):
            raise ValidationError(f"{where}: node {up_id!r} has an unknown process")
        up_spec = self._processes.get(self._by_id[up_id].process).spec
        try:
            return up_spec.output(up_port)
        except KeyError:
            raise ValidationError(
                f"{where}: node {up_id!r} has no output {up_port!r}"
            )

    def _source_types(self, src) -> Tuple[str, ...]:
        """Types of a source: "node.port" (a node output or zone socket) or
        ``{"$ref": input}``."""
        if _is_ref(src):
            name = src["$ref"]
            if name not in self.inputs:
                raise ValidationError(f"unknown $ref {name!r}")
            return (self.inputs[name].type,)
        if not isinstance(src, str):
            raise ValidationError(f"source {src!r} must be 'node.port' or {{'$ref': ...}}")
        up_id, up_port = _split_ref(src)
        return tuple(self._check_upstream_port(src, up_id, up_port).types)

    def _item_types(self, zone: Zone) -> Tuple[str, ...]:
        """Type of one item of a For each zone's ``items``."""
        src = zone.raw.get("items")
        if isinstance(src, str) and src.split(".")[0] in (zone.id, zone.out):
            raise ValidationError("items cannot come from the zone itself")
        types = self._source_types(src)
        if _is_ref(src) and self.inputs[src["$ref"]].many:
            return types
        if len(types) == 1 and self._types.has(types[0]) and self._types.kind_of(types[0]) == "vector":
            return types
        raise ValidationError(f"items {src} ({list(types)}): a vector (each feature) or an "
                              f"input with 'many' (each value) is needed")

    def _zone_ports(self, zone: Zone) -> Dict[str, Tuple[str, ...]]:
        """The zone input's sockets (read inside its body): name -> types."""
        ports: Dict[str, Tuple[str, ...]] = {"index": ("int",)}
        if zone.kind == "repeat":
            for name, decl in zone.state.items():
                if isinstance(decl, dict) and "type" in decl:
                    ports[name] = (str(decl["type"]),)
        elif zone.kind == "sweep":
            try:
                values = _sweep_values(zone.raw.get("values"))
            except ValidationError:
                values = []
            ints = values and all(isinstance(v, int) and not isinstance(v, bool) for v in values)
            ports["value"] = ("int",) if ints else ("float",)
        elif zone.kind == "foreach":
            try:
                ports["item"] = self._item_types(zone)
            except ValidationError:
                pass
        return ports

    def _zone_out_ports(self, zone: Zone) -> Dict[str, Tuple[str, ...]]:
        """The zone output's sockets (read after it): a repeat zone's states."""
        return {name: types for name, types in self._zone_ports(zone).items()
                if name in zone.state}

    # ---- structure: vertices, zones' bodies, scopes ---------------------
    #
    # Vertices: node ids, zone ids (a zone's input), zone outs. A zone
    # input reads what the zone takes from before it (count, state inits,
    # items); a zone out reads its input and its next / collect sources.

    @staticmethod
    def _ref_ids(refs) -> set:
        ids = set()
        for ref in refs:
            parts = ref.split(".") if isinstance(ref, str) else []
            if len(parts) == 2 and parts[0]:
                ids.add(parts[0])
        return ids

    @staticmethod
    def _upstream(node: Node) -> set:
        """Ids of the vertices ``node`` reads (input edges and ``$from`` params)."""
        refs = [v for v in node.inputs.values() if isinstance(v, str)]
        refs += [v["$from"] for v in node.params.values() if _is_from(v)]
        return Workflow._ref_ids(refs)

    def _vertex_deps(self, vid: str) -> set:
        if vid in self._by_id:
            ups = self._upstream(self._by_id[vid])
        elif vid in self._zones:
            z = self._zones[vid]
            refs = [d.get("init") for d in z.state.values() if isinstance(d, dict)]
            count = z.raw.get("count")
            if _is_from(count):
                refs.append(count["$from"])
            refs.append(z.raw.get("items"))
            ups = self._ref_ids(refs)
        else:
            z = self._zone_out[vid]
            ups = self._ref_ids(list(z.next.values()) + list(z.collect.values())) | {z.id}
        return ups & self._vertices()

    def _vertices(self) -> set:
        return set(self._by_id) | set(self._zones) | set(self._zone_out)

    def _structure(self):
        """``(deps, body, owner)``: vertex -> vertices it reads; zone id -> the
        vertices of its body (nested zones' included); vertex -> the zone
        directly holding it (None: the top level). ValidationError on a cycle
        or a zone mixed up with another."""
        if self._structure_cache is not None:
            return self._structure_cache
        verts = self._vertices()
        deps = {v: self._vertex_deps(v) for v in verts}
        readers: Dict[str, set] = {v: set() for v in verts}
        for v, ups in deps.items():
            for u in ups:
                readers[u].add(v)

        # acyclic
        indeg = {v: len(deps[v]) for v in verts}
        ready = [v for v in verts if indeg[v] == 0]
        seen = 0
        while ready:
            v = ready.pop()
            seen += 1
            for r in readers[v]:
                indeg[r] -= 1
                if indeg[r] == 0:
                    ready.append(r)
        if seen != len(verts):
            stuck = sorted(v for v in verts if indeg[v] > 0)
            raise ValidationError(f"workflow has a cycle among nodes {stuck}")

        def reach(start, edges, avoid):
            out, todo = set(), list(edges[start])
            while todo:
                v = todo.pop()
                if v in out or v == avoid:
                    continue
                out.add(v)
                todo.extend(edges[v])
            return out

        # A zone's body: what reads its input (directly or not), and, of what
        # its output reads, the runner nodes (they advance once per
        # iteration) and what reads them; the rest its output reads does not
        # change from one iteration to the next and runs once, before. A zone
        # touching another's body is wholly in it.
        body: Dict[str, set] = {}
        before: Dict[str, set] = {}
        after: Dict[str, set] = {}
        for z in self.zones:
            down = reach(z.id, readers, z.out)
            up = reach(z.out, deps, z.id)
            live = {v for v in up if v in self._by_id and self._by_id[v].process in RUNNERS}
            for v in list(live):
                live |= reach(v, readers, z.out) & up
            before[z.id] = reach(z.id, deps, None)
            after[z.id] = reach(z.out, readers, None)
            body[z.id] = (down | live) - before[z.id] - {z.id, z.out}
        whole = lambda zid: body[zid] | {zid, self._zones[zid].out}  # noqa: E731
        grown = True
        while grown:
            grown = False
            for z in self.zones:
                for w in self.zones:
                    if w is z or not {w.id, w.out} & body[z.id] or whole(w.id) <= body[z.id]:
                        continue
                    if {z.id, z.out} & body[w.id]:
                        raise ValidationError(
                            f"zones {z.id!r} and {w.id!r} each read the other's inside")
                    body[z.id] |= whole(w.id)
                    grown = True
        for z in self.zones:
            for w in self.zones:
                shared = body[z.id] & body[w.id]
                if w is not z and shared and not whole(w.id) <= body[z.id] \
                        and not whole(z.id) <= body[w.id]:
                    raise ValidationError(
                        f"zones {z.id!r} and {w.id!r} overlap without one holding the other "
                        f"({sorted(shared)})")
            mixed = sorted(body[z.id] & after[z.id])
            if mixed:
                raise ValidationError(
                    f"zone {z.id!r}: {mixed} read both inside the zone and its result")
            looped = sorted(body[z.id] & before[z.id])
            if looped:
                raise ValidationError(
                    f"zone {z.id!r}: {looped} are inside the zone and feed what it takes before")
        owner: Dict[str, Optional[str]] = {}
        for v in verts:
            zid_of_v = v if v in self._zones else (self._zone_out[v].id if v in self._zone_out else None)
            holders = [z.id for z in self.zones if v in body[z.id] and z.id != zid_of_v]
            owner[v] = min(holders, key=lambda zid: len(body[zid])) if holders else None
        self._structure_cache = (deps, body, owner)
        return self._structure_cache

    def _unit(self, vid: str) -> str:
        """The unit a vertex runs in: a node itself, a zone socket its zone."""
        return self._zone_out[vid].id if vid in self._zone_out else vid

    def _scope_order(self, scope: Optional[str], only: Optional[set] = None) -> List[str]:
        """Units (node ids, zone ids) run directly in ``scope`` (a zone id,
        None: the top level), in an order where each comes after what it
        reads; ``only``: those vertices' units."""
        deps, body, owner = self._structure()

        def parent(unit):
            return owner[unit]

        def lift(v):
            """The unit of ``scope`` holding vertex ``v``, None when ``v`` is
            not inside ``scope`` (it ran before)."""
            u = self._unit(v)
            while True:
                p = parent(u)
                if p == scope:
                    return u
                if p is None:
                    return None
                u = p

        units = [u for u in dict.fromkeys(self._unit(v) for v in self._doc_order())
                 if parent(u) == scope and (only is None or u in only)]
        needs: Dict[str, set] = {}
        for u in units:
            verts = {u} if u in self._by_id else body[u] | {u, self._zones[u].out}
            got = set()
            for v in verts:
                for d in deps[v]:
                    lu = lift(d)
                    if lu is not None and lu != u:
                        got.add(lu)
            needs[u] = got & set(units)
        order: List[str] = []
        done: set = set()
        pending = list(units)
        while pending:
            for u in list(pending):
                if needs[u] <= done:
                    order.append(u)
                    done.add(u)
                    pending.remove(u)
                    break
            else:
                raise ValidationError(f"workflow has a cycle among {sorted(pending)}")
        return order

    def _doc_order(self) -> List[str]:
        out = [n.id for n in self.nodes]
        for z in self.zones:
            out += [z.id, z.out]
        return out

    def _topo_order(self) -> List[Node]:
        """Every node, each after what it reads (zones' bodies where their
        zone runs)."""
        out: List[Node] = []

        def visit(scope):
            for u in self._scope_order(scope):
                if u in self._by_id:
                    out.append(self._by_id[u])
                else:
                    visit(u)

        visit(None)
        return out

    # ---- execution ----------------------------------------------------

    def run(
        self, args: Optional[Dict[str, object]] = None, session: Optional[Session] = None,
        until: Optional[Iterable[str]] = None,
        on_status: Optional[Callable[[Dict], None]] = None,
    ) -> Dict[str, DataHandle]:
        """The declared outputs produced (see ``execute``)."""
        return self.execute(args, session, until, on_status).outputs

    def needed(self, until: Iterable[str]) -> set:
        """Ids of the ``until`` nodes and of every vertex they read, directly
        or not (a vertex in a zone brings the zone's input and output)."""
        todo = list(until)
        for nid in todo:
            if nid not in self._by_id:
                raise WorkflowError(f"until: unknown node {nid!r}")
        deps, _body, owner = self._structure()
        out: set = set()
        while todo:
            vid = todo.pop()
            if vid in out:
                continue
            out.add(vid)
            todo.extend(deps[vid])
            unit = self._unit(vid)
            if unit in self._zones:
                todo += [unit, self._zones[unit].out]
            if owner[unit] is not None:
                todo += [owner[unit], self._zones[owner[unit]].out]
        return out

    def execute(
        self, args: Optional[Dict[str, object]] = None, session: Optional[Session] = None,
        until: Optional[Iterable[str]] = None,
        on_status: Optional[Callable[[Dict], None]] = None,
    ) -> WorkflowResult:
        """Runs the workflow: every node, or (``until``: node ids) only those
        nodes and what they read. ``on_status(event)`` gets, per node run,
        ``{node, state: "running", index, count}``, then any number of
        ``{node, state: "progress", done, total, phase}`` (the process's
        progress reports), then ``{node, state: "done"}``, or ``{node, state:
        "error" | "cancelled", message}`` before the exception goes on. A
        zone sends ``{node: zone id, state: "iteration", iteration, iterations}``
        before each iteration; the events of the nodes in a zone carry
        ``index`` / ``count`` of the top-level unit and ``zones``: [[zone id,
        iteration, iterations], ...] (outermost first). The cancel check of
        the progress listener around the call still applies inside the
        nodes."""
        args = args or {}
        if session is None:
            session = Session(self._types, self._converters)
        targets = None if until is None else list(until)
        scope = None if targets is None else self.needed(targets)
        self._structure()

        run = _Execution(self, session, self._resolve_inputs(args), on_status,
                         None if scope is None else {self._unit(v) for v in scope})
        run.scope(None, None)
        produced = run.produced

        # An optional output a node did not produce is left out of the results.
        results: Dict[str, DataHandle] = {}
        for oname, ref in self.outputs.items():
            up_id, up_port = _split_ref(ref)
            if (up_id, up_port) in produced:
                results[oname] = produced[(up_id, up_port)]
        keep_ids = {nid for nid in run.ran if self._by_id[nid].keep} | set(targets or ())
        kept = {f"{nid}.{port}": h for (nid, port), h in produced.items() if nid in keep_ids}
        return WorkflowResult(outputs=results, kept=kept, series=run.series)

    def _node_kwargs(self, node, input_values, produced, session) -> Dict[str, object]:
        """The keyword arguments of ``node``'s process: its params (literal,
        workflow input, or another node's output) and input ports."""
        spec = self._processes.get(node.process).spec
        kwargs: Dict[str, object] = {}

        for param in spec.params:
            if param.name in node.params:
                raw = node.params[param.name]
                if _is_ref(raw):
                    value = input_values[raw["$ref"]]
                elif _is_from(raw):
                    up = _split_ref(raw["$from"])
                    if up not in produced:
                        if param.optional or param.has_default:
                            if param.has_default:
                                kwargs[param.target] = param.default_value
                            continue
                        raise WorkflowError(
                            f"node {node.id!r} param {param.name!r}: "
                            f"{raw['$from']} was not produced"
                        )
                    value = self._linked_param(node, param, session.get(produced[up]))
                else:
                    value = raw
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
                value = session.get(produced[(up_id, up_port)])
            coerced, _ = self._converters.resolve(value, port.types)
            kwargs[port.target] = coerced
        return kwargs

    def _linked_param(self, node, param, value):
        """``value`` (a node's output) as ``param`` takes it: an int widened to
        a float param, else converted; its choices / bounds checked."""
        if any(self._types.has(t) and self._types.get(t).matches(value) for t in param.types):
            coerced = value
        elif self._types.identify(value) == "int" and "float" in param.types:
            coerced = float(value)
        else:
            coerced, _ = self._converters.resolve(value, param.types)
        try:
            self._check_param_literal(node, param, coerced)
        except ValidationError as exc:
            raise WorkflowError(str(exc)) from None
        return coerced

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


def _sweep_values(values) -> List:
    """A sweep zone's values: ``{list: [...]}`` or ``{start, stop, step}``
    (stop included when a step lands on it)."""
    if not isinstance(values, dict):
        raise ValidationError("give {list: [...]} or {start, stop, step}")
    if "list" in values:
        out = values["list"]
        if not isinstance(out, list) or not out or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in out):
            raise ValidationError("list must be a non-empty list of numbers")
        return list(out)
    try:
        start, stop, step = (values[k] for k in ("start", "stop", "step"))
    except KeyError as exc:
        raise ValidationError(f"missing {exc.args[0]!r}") from None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (start, stop, step)):
        raise ValidationError("start, stop and step must be numbers")
    if step == 0 or (stop - start) * step < 0:
        raise ValidationError("step must be nonzero and go from start towards stop")
    count = int(abs(stop - start) / abs(step) + 1e-9) + 1
    if count > 100000:
        raise ValidationError(f"{count} values: more than 100000")
    ints = all(isinstance(v, int) for v in (start, stop, step))
    return [start + i * step if ints else float(start + i * step) for i in range(count)]


_FRAME_KINDS = ("grid", "field")


class _Kept(dict):
    """A Repeat zone's runners kept across its iterations: node id -> (runner,
    its input sources, params, series key); ``suffix``: the names' suffix
    where the zone runs."""

    def __init__(self, suffix: str) -> None:
        super().__init__()
        self.suffix = suffix


class _Execution:
    """One ``Workflow.execute``: what the vertices produced (``(vertex,
    port) -> handle``, an iteration's replacing the previous one's, which is
    released), the series recorded, the nodes run."""

    def __init__(self, wf: Workflow, session: Session, values: Dict, on_status, only) -> None:
        self.wf, self.session, self.values = wf, session, values
        self.on_status, self.only = on_status, only
        self.produced: Dict[Tuple[str, str], DataHandle] = {}
        self.series: Dict[str, DataHandle] = {}
        self.ran: set = set()
        self.index = self.count = 0
        self.zones: List[List] = []  # [zone id, iteration, iterations], outermost first
        self.suffix = ""  # names a sweep / for each iteration's runner series
        self.advanced: Dict[str, int] = {}

    # ---- helpers -----------------------------------------------------

    def emit(self, event: Dict) -> None:
        if self.on_status is None:
            return
        if self.zones:
            event = dict(event, zones=[list(z) for z in self.zones])
        self.on_status(event)

    def assign(self, key: Tuple[str, str], handle: DataHandle) -> None:
        old = self.produced.get(key)
        self.produced[key] = handle
        if old is not None and old != handle and old not in self.produced.values() \
                and self.session.has(old):
            self.session.release(old)

    def put(self, key: Tuple[str, str], value, type_id: Optional[str]) -> None:
        handle = self.session.put(value, type_id=type_id, name=f"{key[0]}.{key[1]}",
                                  provenance=key[0])
        self.assign(key, handle)

    def source_value(self, src):
        """The value of a source ("vertex.port" or {"$ref"})."""
        if _is_ref(src):
            return self.values[src["$ref"]]
        key = _split_ref(src)
        if key not in self.produced:
            raise WorkflowError(f"{src} was not produced")
        return self.session.get(self.produced[key])

    # ---- running -----------------------------------------------------

    def scope(self, scope: Optional[str], runners: Optional[Dict]) -> None:
        order = self.wf._scope_order(scope, self.only)
        for i, unit in enumerate(order):
            if scope is None:
                self.index, self.count = i, len(order)
            if unit in self.wf._zones:
                self.zone(self.wf._zones[unit])
            else:
                self.node(self.wf._by_id[unit], runners)

    def node(self, node: Node, runners: Optional[Dict]) -> None:
        wf = self.wf
        self.emit({"node": node.id, "state": "running", "index": self.index, "count": self.count})
        try:
            progress.check_cancelled()
            kwargs = wf._node_kwargs(node, self.values, self.produced, self.session)
            proc = wf._processes.get(node.process)
            if self.on_status is None:
                result = self.call(node, proc, kwargs, runners)
            else:
                with progress.relayed(
                    lambda d, t, ph, nid=node.id: self.emit(
                        {"node": nid, "state": "progress", "done": d, "total": t, "phase": ph})
                ):
                    result = self.call(node, proc, kwargs, runners)
        except progress.Cancelled as exc:
            self.emit({"node": node.id, "state": "cancelled", "message": str(exc)})
            raise
        except Exception as exc:
            self.emit({"node": node.id, "state": "error", "message": str(exc)})
            raise
        spec = proc.spec
        for oname, value in result.items():
            out = spec.output(oname)
            self.put((node.id, oname), value, out.types[0] if len(out.types) == 1 else None)
        self.ran.add(node.id)
        self.emit({"node": node.id, "state": "done"})

    def call(self, node: Node, proc, kwargs: Dict, runners: Optional[Dict]) -> Dict:
        """The node's outputs: the process called, or its live runner
        advanced (a ``runner`` section, or a runner process in a Repeat zone:
        ``runners`` keeps its runner from one iteration to the next)."""
        if node.process not in RUNNERS or (node.runner is None and runners is None):
            return proc(**kwargs)
        cls = RUNNERS[node.process]
        spec = proc.spec
        ports = {p.target for p in spec.inputs}
        inputs = {k: v for k, v in kwargs.items() if k in ports}
        params = {k: v for k, v in kwargs.items() if k not in ports}
        cfg = node.runner or {}
        # Where each input comes from this time: an input whose source
        # handle is another one than at the last iteration changed.
        sources = {}
        for port in spec.inputs:
            src = node.inputs.get(port.name)
            if isinstance(src, str):
                sources[port.target] = self.produced.get(_split_ref(src))
        held = runners.get(node.id) if runners is not None else None
        if held is None:
            live = cls(inputs, params)
            # a kept runner records one series across the iterations
            key = node.id + (runners.suffix if runners is not None else self.suffix)
            self.advanced[key] = 0
        else:
            live, last_sources, last_params, key = held
            fixed = [k for k in cls.config if params.get(k) != last_params.get(k)]
            if fixed:
                raise WorkflowError(
                    f"node {node.id!r}: {', '.join(fixed)} changed between iterations; "
                    f"the runner fixes them when it opens")
            changed = {k: inputs[k] for k, h in sources.items()
                       if k in inputs and h != last_sources.get(k)}
            if changed:
                if not cls.takes_updates():
                    raise WorkflowError(
                        f"node {node.id!r}: {node.process} cannot take new inputs "
                        f"({', '.join(changed)}) between advances")
                live.update(changed)
        if runners is not None:
            runners[node.id] = (live, sources, dict(params), key)
        try:
            result = None
            for _ in range(int(cfg.get("advances", 1))):
                progress.check_cancelled()
                result = live.advance(dict(params))
                self.advanced[key] = self.advanced.get(key, 0) + 1
                every = int(cfg.get("record_every", 0))
                if every > 0 and self.advanced[key] % every == 0:
                    self.record_runner(node, spec, live, result, cfg, key)
                if cfg.get("until_stopped") and live.status().get("stopped"):
                    break
        finally:
            if runners is None:
                live.close()
        return proc._bind_outputs(result)

    def record_runner(self, node, spec, live, result, cfg, key) -> None:
        types = {o.name: o.types[0] for o in spec.outputs}
        coord = live.coordinate() or {"advance": self.advanced[key]}
        axis, at = next(iter(coord.items()))
        series = self.series.get(key)
        if series is None or not self.session.has(series):
            series = self.session.group(f"{node.id} frames{key[len(node.id):]}", axis=axis,
                                        provenance=node.id)
            self.series[key] = series
        fields = cfg.get("record_fields")
        if fields is None:
            fields = [n for n, t in types.items()
                      if self.wf._types.has(t) and self.wf._types.kind_of(t) in _FRAME_KINDS]
        values = {n: v for n, v in (result or {}).items() if n in types}
        record(self.session, series, values, at, type_ids=types, fields=fields,
               provenance=node.id)

    def zone(self, z: Zone) -> None:
        wf = self.wf
        self.emit({"node": z.id, "state": "running", "index": self.index, "count": self.count})
        try:
            if z.kind == "repeat":
                count = z.raw.get("count", 0)
                if _is_ref(count) or _is_from(count):
                    count = self.source_value(count if _is_ref(count) else count["$from"])
                iters = list(range(int(count)))
            elif z.kind == "sweep":
                iters = _sweep_values(z.raw.get("values"))
            else:
                items = self.source_value(z.raw["items"])
                src = z.raw["items"]
                if _is_ref(src) and wf.inputs[src["$ref"]].many:
                    iters = list(items or [])
                else:
                    iters = [items.feature(i) for i in range(items.n_features)]
            ports = wf._zone_ports(z)
            for name, decl in z.state.items():
                init = decl["init"]
                if isinstance(init, str) and _split_ref(init) in self.produced:
                    value = self.session.get(self.produced[_split_ref(init)])
                else:
                    value = self.source_value(init)
                coerced, _ = wf._converters.resolve(value, (decl["type"],))
                self.put((z.id, name), coerced, decl["type"])
            runners = _Kept(self.suffix) if z.kind == "repeat" else None
            types = {}
            for name, ref in z.collect.items():
                t = wf._source_types(ref)
                types[name] = t[0] if len(t) == 1 else None
            series_key = z.id + self.suffix
            outer_suffix = self.suffix
            self.zones.append([z.id, 0, len(iters)])
            try:
                for k, it in enumerate(iters):
                    progress.check_cancelled()
                    self.zones[-1][1] = k + 1
                    self.emit({"node": z.id, "state": "iteration", "iteration": k + 1,
                               "iterations": len(iters), "index": self.index, "count": self.count})
                    self.put((z.id, "index"), k, "int")
                    if z.kind == "sweep":
                        self.put((z.id, "value"), it, ports["value"][0])
                    elif z.kind == "foreach":
                        self.put((z.id, "item"), it, ports.get("item", (None,))[0])
                    at = k + 1 if z.kind == "repeat" else (it if z.kind == "sweep" else k)
                    self.suffix = f"{outer_suffix} ({z.axis} = {at})"
                    self.scope(z.id, runners)
                    self.suffix = outer_suffix
                    nexts = {}
                    for name, ref in z.next.items():
                        key = _split_ref(ref)
                        if key not in self.produced:
                            raise WorkflowError(f"zone {z.id!r}: next {name!r}: {ref} was not produced")
                        nexts[name] = self.produced[key]
                    for name, handle in nexts.items():
                        stype = z.state[name]["type"]
                        value = self.session.get(handle)
                        coerced, _ = wf._converters.resolve(value, (stype,))
                        if coerced is value:
                            self.assign((z.id, name), handle)
                        else:
                            self.put((z.id, name), coerced, stype)
                    if z.collect and (k + 1) % max(1, z.record_every) == 0:
                        values = {}
                        for name, ref in z.collect.items():
                            key = _split_ref(ref)
                            if key in self.produced:
                                values[name] = self.session.get(self.produced[key])
                        series = self.series.get(series_key)
                        if series is None:
                            series = self.session.group(str(z.raw.get("label") or z.id) + outer_suffix,
                                                        axis=z.axis, provenance=z.id)
                            self.series[series_key] = series
                        record(self.session, series, values, at, type_ids=types, provenance=z.id)
            finally:
                self.zones.pop()
                self.suffix = outer_suffix
                for live, *_rest in (runners or {}).values():
                    live.close()
            for name in z.state:
                self.assign((z.out, name), self.produced[(z.id, name)])
        except progress.Cancelled as exc:
            self.emit({"node": z.id, "state": "cancelled", "message": str(exc)})
            raise
        except Exception as exc:
            self.emit({"node": z.id, "state": "error", "message": str(exc)})
            raise
        self.emit({"node": z.id, "state": "done"})
