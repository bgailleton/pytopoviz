"""A series' field through its frames, as a table.

``series_table(session, series, field, points_xy=None)`` reads ``field`` in
each frame of ``series`` (core/series.py) and returns a DataTable over the
series' axis: the field's minimum, mean and maximum per frame (NaN ignored),
and with ``points_xy`` (map coordinates, the field's CRS) the value at the
cell of each point. Frames moved to the disk store are read from it without
being put back in memory.

Author: B.G.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .core import CONVERTERS, Session, frames
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
