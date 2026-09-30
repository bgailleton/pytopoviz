"""Process declaration and registry.

A *process* is a unit of work with a typed interface. Declared with ``@process``:

    @process(id="topotoolbox.gaussian_smooth", label="Gaussian smooth",
             inputs=[Port("dem", "topotoolbox.GridObject")],
             params=[Param("sigma", "float", default=2.0)],
             outputs=[Output("smoothed", "topotoolbox.GridObject")])
    def gaussian_smooth(dem, sigma):
        ...

The decorated object stays directly callable (kwargs in) and carries ``.spec``.
Output binding: one Output -> bare return; many -> a dict keyed by output name,
or a tuple/list in declared order.

``impl`` records how the process is implemented for the contract:
"library" (adapter wrapper over a lib fn) or "composite" (body calls processes).

``description`` defaults to the decorated function's docstring.

``describe`` is an optional callable taking the same keyword arguments as the
process: it says, cheaply and without running it, what a run would produce (grid
shape, download size...), as a JSON-serialisable dict. Frontends call it to
preview a run.

Author: B.G.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .errors import RegistrationError, ValidationError
from .kinds import is_data_kind, is_param_kind
from .ports import Output, Param, Port
from .types import TypeRegistry


@dataclass(frozen=True)
class ProcessSpec:
    """The typed interface of a process."""

    id: str
    label: str
    inputs: Tuple[Port, ...]
    params: Tuple[Param, ...]
    outputs: Tuple[Output, ...]
    impl: str  # "library" | "composite"
    description: str = ""

    def input(self, name: str) -> Port:
        for p in self.inputs:
            if p.name == name:
                return p
        raise KeyError(name)

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def output(self, name: str) -> Output:
        for o in self.outputs:
            if o.name == name:
                return o
        raise KeyError(name)


class Process:
    """Callable wrapper carrying a ProcessSpec.

    Calling it runs the underlying function with keyword arguments and normalises
    the return into ``{output_name: value}``.
    """

    def __init__(self, fn: Callable, spec: ProcessSpec,
                 describe: Optional[Callable] = None) -> None:
        self._fn = fn
        self._describe = describe
        self.spec = spec
        functools.update_wrapper(self, fn)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def describable(self) -> bool:
        return self._describe is not None

    def describe(self, **kwargs) -> Dict:
        """What a run with these arguments would produce, without running it."""
        if self._describe is None:
            raise ValidationError(f"process {self.spec.id!r} has no describe")
        return self._describe(**kwargs)

    def __call__(self, **kwargs):
        result = self._fn(**kwargs)
        return self._bind_outputs(result)

    def call_raw(self, **kwargs):
        """Call the underlying function, returning its raw (un-normalised) result."""
        return self._fn(**kwargs)

    def _bind_outputs(self, result) -> Dict[str, object]:
        """Map the raw result to ``{output name: value}``. Optional outputs
        that are absent or ``None`` are left out of the mapping."""
        outs = self.spec.outputs
        if len(outs) == 0:
            return {}
        if len(outs) == 1:
            if isinstance(result, dict) and outs[0].name in result:
                return {outs[0].name: result[outs[0].name]}
            return {outs[0].name: result}
        # multiple outputs
        if isinstance(result, dict):
            missing = [o.name for o in outs if o.name not in result and not o.optional]
            if missing:
                raise ValidationError(
                    f"process {self.spec.id!r} returned dict missing outputs: {missing}"
                )
            return {
                o.name: result[o.name]
                for o in outs
                if not (o.optional and result.get(o.name) is None)
            }
        if isinstance(result, (tuple, list)):
            if len(result) != len(outs):
                raise ValidationError(
                    f"process {self.spec.id!r} returned {len(result)} values, "
                    f"expected {len(outs)}"
                )
            return {o.name: v for o, v in zip(outs, result)}
        raise ValidationError(
            f"process {self.spec.id!r} has {len(outs)} outputs but returned a "
            f"{type(result).__name__}; return a dict or tuple"
        )

    def __repr__(self) -> str:
        return f"<Process {self.spec.id!r}>"


class ProcessRegistry:
    """Holds Process objects keyed by id."""

    def __init__(self) -> None:
        self._procs: Dict[str, Process] = {}

    def add(self, proc: Process) -> Process:
        pid = proc.spec.id
        if pid in self._procs:
            raise RegistrationError(f"process id already registered: {pid!r}")
        self._procs[pid] = proc
        return proc

    def get(self, process_id: str) -> Process:
        try:
            return self._procs[process_id]
        except KeyError:
            raise RegistrationError(f"unknown process id: {process_id!r}")

    def has(self, process_id: str) -> bool:
        return process_id in self._procs

    def clear(self) -> None:
        self._procs.clear()

    def __iter__(self) -> Iterator[Process]:
        return iter(self._procs.values())

    def __len__(self) -> int:
        return len(self._procs)

    def __contains__(self, process_id: object) -> bool:
        return process_id in self._procs


def make_spec(
    id: str,
    label: Optional[str] = None,
    inputs: Optional[Sequence[Port]] = None,
    params: Optional[Sequence[Param]] = None,
    outputs: Optional[Sequence[Output]] = None,
    impl: str = "composite",
    description: str = "",
) -> ProcessSpec:
    if not id or not isinstance(id, str):
        raise RegistrationError("process id must be a non-empty string")
    if impl not in ("library", "composite"):
        raise RegistrationError(f"impl must be 'library' or 'composite', got {impl!r}")
    inputs = tuple(inputs or ())
    params = tuple(params or ())
    outputs = tuple(outputs or ())

    for p in params:
        if p.choice_labels is not None and (
            p.choices is None or len(p.choice_labels) != len(p.choices)
        ):
            raise RegistrationError(
                f"process {id!r}: param {p.name!r} needs one choice label per choice"
            )

    seen = set()
    for item in (*inputs, *params, *outputs):
        if item.name in seen:
            raise RegistrationError(
                f"process {id!r}: duplicate interface name {item.name!r}"
            )
        seen.add(item.name)

    return ProcessSpec(
        id=id,
        label=label or id,
        inputs=inputs,
        params=params,
        outputs=outputs,
        impl=impl,
        description=description or "",
    )


def process(
    id: str,
    label: Optional[str] = None,
    inputs: Optional[Sequence[Port]] = None,
    params: Optional[Sequence[Param]] = None,
    outputs: Optional[Sequence[Output]] = None,
    impl: str = "composite",
    registry: Optional[ProcessRegistry] = None,
    description: Optional[str] = None,
    describe: Optional[Callable] = None,
) -> Callable[[Callable], Process]:
    """Decorator turning a function into a registered Process.

    ``registry`` defaults to the global default registry (bound in
    ``core.registries`` and passed by that module to avoid an import cycle).
    """

    def decorate(fn: Callable) -> Process:
        desc = description if description is not None else (inspect.getdoc(fn) or "")
        spec = make_spec(id, label, inputs, params, outputs, impl, desc)
        proc = Process(fn, spec, describe)
        target = registry if registry is not None else DEFAULT_PROCESSES
        target.add(proc)
        return proc

    return decorate


# The default process registry. Owned here (no import cycle): registries.py
# re-exports it as PROCESSES. @process with no explicit registry lands here.
DEFAULT_PROCESSES = ProcessRegistry()


def validate_spec_types(spec: ProcessSpec, types: TypeRegistry) -> None:
    """Check every type_id referenced by ``spec`` exists and unions share a kind.

    Params must reference PARAM kinds, inputs/outputs DATA kinds.
    """

    def check_union(where: str, type_ids: Tuple[str, ...], want_param: bool) -> None:
        kinds = set()
        for tid in type_ids:
            if not types.has(tid):
                raise ValidationError(
                    f"process {spec.id!r} {where}: unknown type_id {tid!r}"
                )
            k = types.kind_of(tid)
            kinds.add(k)
            if want_param and not is_param_kind(k):
                raise ValidationError(
                    f"process {spec.id!r} {where}: {tid!r} has data kind {k!r}, "
                    f"expected a param kind"
                )
            if not want_param and not is_data_kind(k):
                raise ValidationError(
                    f"process {spec.id!r} {where}: {tid!r} has param kind {k!r}, "
                    f"expected a data kind"
                )
        if len(kinds) > 1:
            raise ValidationError(
                f"process {spec.id!r} {where}: union spans multiple kinds {sorted(kinds)}"
            )

    for p in spec.inputs:
        check_union(f"input {p.name!r}", p.types, want_param=False)
    for p in spec.params:
        check_union(f"param {p.name!r}", p.types, want_param=True)
    for o in spec.outputs:
        check_union(f"output {o.name!r}", o.types, want_param=False)
