"""Series: values along one axis (time steps, model time, a parameter).

A series is a Session group with an ``axis``; each frame is a group under it
with ``coord = {axis: value}``, holding that frame's fields (e.g. depth and
discharge at one model time). ``record`` adds a frame (a live runner's outputs
after an advance); ``sweep`` runs a process once per value of one param and
records each run as a frame, so any process gives a parameter-space series.

Author: B.G.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import progress
from .errors import SessionError
from .process import Process
from .session import GROUP, DataHandle, Session


def record(
    session: Session,
    series: DataHandle,
    outputs: Dict[str, object],
    coord: Any,
    type_ids: Optional[Dict[str, str]] = None,
    fields: Optional[Iterable[str]] = None,
    name: Optional[str] = None,
    provenance: Optional[str] = None,
) -> DataHandle:
    """Adds a frame at ``coord`` (on the series' axis) holding ``outputs``
    (name -> value; ``None`` values skipped), only the names in ``fields``
    when given. ``type_ids`` names a value's type when it can't be identified."""
    axis = session.item(series).get("axis")
    if series.type_id != GROUP or not axis:
        raise SessionError(f"{series!r} is not a series (a group with an axis)")
    keep = None if fields is None else set(fields)
    frame = session.group(name or f"{axis} = {_fmt(coord)}", parent=series,
                          coord={axis: coord}, provenance=provenance)
    types = type_ids or {}
    for oname, value in outputs.items():
        if value is None or (keep is not None and oname not in keep):
            continue
        session.put(value, type_id=types.get(oname), name=oname, parent=frame,
                    provenance=provenance)
    return frame


def frames(session: Session, series: DataHandle,
           field: Optional[str] = None) -> List[Tuple[Any, DataHandle]]:
    """``(coord, frame)`` in axis order; with ``field``, ``(coord, that
    field's handle)`` for the frames that hold it."""
    axis = session.item(series).get("axis")
    out = []
    for frame in session.children(series):
        c = session.item(frame)["coord"].get(axis)
        if field is None:
            out.append((c, frame))
            continue
        if frame.type_id != GROUP:
            continue
        for h in session.children(frame):
            if h.name == field:
                out.append((c, h))
                break
    return out


def sweep(
    session: Session,
    process: Process,
    param: str,
    values: Sequence[Any],
    inputs: Optional[Dict[str, object]] = None,
    params: Optional[Dict[str, object]] = None,
    name: Optional[str] = None,
    parent: Optional[DataHandle] = None,
    fields: Optional[Iterable[str]] = None,
    converters=None,
) -> DataHandle:
    """Runs ``process`` once per value of its param ``param`` (``inputs``:
    port -> value, ``params``: the other params over their defaults) and
    returns the series (axis ``param``) of the runs' outputs, one frame per
    value. ``converters`` (default: the session's) coerces the inputs to the
    ports' types."""
    spec = process.spec
    spec.param(param)  # raises on an unknown name
    conv = converters if converters is not None else session._converters
    series = session.group(name or f"{spec.label or spec.id}: {param} sweep", parent=parent,
                           axis=param, provenance=spec.id,
                           info={"process": spec.id, "params": dict(params or {})})
    kwargs: Dict[str, object] = {}
    for port in spec.inputs:
        if port.name in (inputs or {}):
            kwargs[port.target] = conv.resolve(inputs[port.name], port.types)[0]
    given = dict(params or {})
    types = {o.name: o.types[0] for o in spec.outputs if len(o.types) == 1}
    n = len(values)
    for i, v in enumerate(values):
        progress.report(i, n, f"{param} sweep")
        call = dict(kwargs)
        for p in spec.params:
            if p.name == param:
                call[p.target] = v
            elif p.name in given:
                call[p.target] = given[p.name]
            elif p.has_default:
                call[p.target] = p.default_value
        record(session, series, process(**call), v, type_ids=types, fields=fields,
               provenance=spec.id)
    progress.report(n, n, f"{param} sweep")
    return series


def _fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)
