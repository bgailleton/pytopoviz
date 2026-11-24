"""A series' field through its frames, as a table.

``series_table(session, series, field, points_xy=None)`` reads ``field`` in
each frame of ``series`` (core/series.py) and returns a DataTable over the
series' axis: the field's minimum, mean and maximum per frame (NaN ignored),
and with ``points_xy`` (map coordinates, the field's CRS) the value at the
cell of each point. Frames moved to the disk store are read from it without
being put back in memory.

``series_profile(session, series, field, process, port, output, ...)`` runs a
profile process (e.g. a swath) on ``field`` of each frame and returns one
column of its table output per frame along the profile's x, as a heatmap
table (DataTable.heatmap): x on one axis, the series' axis on the other.

Author: B.G.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .core import CONVERTERS, Session, frames, progress
from .datatable import DataTable


def series_table(session: Session, series, field: str,
                 points_xy: Optional[Sequence[Sequence[float]]] = None) -> DataTable:
    axis = session.item(series).get("axis") or "frame"
    found = frames(session, series, field)
    if not found:
        raise ValueError(f"no frame of this series holds {field!r}")
    coords, lo, mean, hi = [], [], [], []
    pts = [] if points_xy is None else [tuple(map(float, p[:2])) for p in points_xy]
    at = [[] for _ in pts]
    for c, h in found:
        z, meta = _array(session, h)
        coords.append(float(c) if isinstance(c, (int, float)) else float(len(coords)))
        finite = np.isfinite(z)
        if finite.any():
            v = z[finite]
            lo.append(float(v.min()))
            mean.append(float(v.mean()))
            hi.append(float(v.max()))
        else:
            lo.append(np.nan)
            mean.append(np.nan)
            hi.append(np.nan)
        for k, (x, y) in enumerate(pts):
            rc = _cell(meta, z.shape, x, y)
            at[k].append(float(z[rc]) if rc is not None else np.nan)
    columns = {axis: np.array(coords), f"{field} min": np.array(lo),
               f"{field} mean": np.array(mean), f"{field} max": np.array(hi)}
    roles = {axis: "x", f"{field} min": "band_lo", f"{field} max": "band_hi",
             f"{field} mean": "series"}
    for k in range(len(pts)):
        name = f"{field} at point {k + 1}"
        columns[name] = np.array(at[k])
        roles[name] = "series"
    return DataTable(columns=columns, roles=roles,
                     bands=[(f"{field} min", f"{field} max")],
                     title=f"{session.item(series)['name']}: {field}")


def series_profile(session: Session, series, field: str, process, port: str, output: str,
                   column: str = "mean", inputs: Optional[dict] = None,
                   params: Optional[dict] = None, converters=None) -> DataTable:
    """``process`` run on ``field`` of every frame of ``series`` (given to its
    input ``port``; ``inputs``: its other ports -> values, ``params``: params
    over their defaults); from each run, ``column`` of the table output
    ``output`` along that table's x column. Frames whose x differs from the
    first frame's are interpolated onto it (NaN outside). The map_x / map_y
    columns of the first frame's table are kept."""
    axis = session.item(series).get("axis") or "frame"
    found = frames(session, series, field)
    if not found:
        raise ValueError(f"no frame of this series holds {field!r}")
    spec = process.spec
    conv = converters if converters is not None else session._converters
    target = {p.name: p for p in spec.inputs}
    if port not in target:
        raise ValueError(f"{spec.id} has no input {port!r}")
    kwargs = {}
    for name, value in (inputs or {}).items():
        if name != port and name in target:
            kwargs[target[name].target] = conv.resolve(value, target[name].types)[0]
    given = dict(params or {})
    for p in spec.params:
        if p.name in given:
            kwargs[p.target] = given[p.name]
        elif p.has_default:
            kwargs[p.target] = p.default_value
    x_name, x, rows, coords, first = None, None, [], [], None
    n = len(found)
    for i, (c, h) in enumerate(found):
        progress.report(i, n, f"{field} profile through frames")
        call = dict(kwargs)
        call[target[port].target] = conv.resolve(session.get(h), target[port].types)[0]
        table = process(**call).get(output)
        if table is None:
            raise ValueError(f"{spec.id} returned no {output!r} for frame {c!r}")
        if not isinstance(table, DataTable):
            table = CONVERTERS.convert(table, "datatable")
        xs = [k for k, r in table.roles.items() if r == "x"]
        if not xs or column not in table.columns:
            raise ValueError(f"{output!r} has no x column or no column {column!r}")
        tx, ty = table.columns[xs[0]], table.columns[column]
        if x is None:
            x_name, x, first = xs[0], tx, table
            rows.append(ty)
        elif len(tx) == len(x) and np.allclose(tx, x, equal_nan=True):
            rows.append(ty)
        else:
            order = np.argsort(tx)
            rows.append(np.interp(x, tx[order], ty[order], left=np.nan, right=np.nan))
        coords.append(float(c) if isinstance(c, (int, float)) else float(len(coords)))
    progress.report(n, n, f"{field} profile through frames")
    names = [f"{axis} = {v:g}" for v in coords]
    columns = {x_name: x}
    roles = {x_name: "x"}
    for name, row in zip(names, rows):
        columns[name], roles[name] = row, "series"
    for k, r in first.roles.items():
        if r in ("map_x", "map_y"):
            key = k if k not in columns else f"{k} (map)"
            columns[key], roles[key] = first.columns[k], r
    return DataTable(columns=columns, units={x_name: first.units.get(x_name, "")}, roles=roles,
                     title=f"{session.item(series)['name']}: {field} {column} through frames",
                     heatmap={"columns": names, "y": coords, "y_name": axis, "y_unit": "",
                              "value": f"{field} {column}"})


def _array(session: Session, handle):
    """The frame field as a 2D float array plus its georaster meta."""
    row = session.item(handle)
    if not row.get("resident", True) and session.spillable(handle):
        array, meta = session.read_array(handle)
    else:
        value = session.get(handle)
        geo = CONVERTERS.convert(value, "georaster", from_type=handle.type_id) \
            if handle.type_id != "field2d" else None
        array, meta = (geo.z, geo.meta()) if geo is not None else (np.asarray(value), {})
    return np.asarray(array, dtype=float), meta


def _cell(meta, shape, x, y):
    """(row, col) of map point (x, y) on a north-up grid, or None outside."""
    cs = float(meta.get("cell_size", 1.0))
    col = int(np.floor((x - float(meta.get("x_min", 0.0))) / cs))
    row = shape[0] - 1 - int(np.floor((y - float(meta.get("y_min", 0.0))) / cs))
    if 0 <= row < shape[0] and 0 <= col < shape[1]:
        return row, col
    return None
