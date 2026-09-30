"""DataTable: pytopoviz's own library-agnostic table of numeric columns.

The hub type of the ``table`` kind (profiles, per-bin statistics, ...): library
results register a converter to it, so a frontend only reads this one
container. Each column says what it is for, so a plotter needs no knowledge of
the process that made it.

- ``columns``: {name: 1-D float64 array}, all the same length, in order.
- ``units``: {name: unit string}; missing = "".
- ``roles``: {name: role}; missing = "series". Roles (``ROLES``):
  ``x`` the abscissa (at most one column), ``series`` a value plotted against
  it, ``band_lo``/``band_hi`` the bounds of a band, ``count`` a sample count,
  ``aux`` kept but not plotted by default (e.g. coordinates).
- ``bands``: [(lo, hi)] pairs; every ``band_lo``/``band_hi`` column is in
  exactly one pair, lo with role ``band_lo`` and hi with ``band_hi``.
- ``title``: free text.

Author: B.G.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

ROLES = ("x", "series", "band_lo", "band_hi", "count", "aux")


@dataclass(eq=False)
class DataTable:
    columns: Dict[str, np.ndarray]
    units: Dict[str, str] = field(default_factory=dict)
    roles: Dict[str, str] = field(default_factory=dict)
    bands: List[Tuple[str, str]] = field(default_factory=list)
    title: str = ""

    def __post_init__(self) -> None:
        cols = {}
        for name, values in self.columns.items():
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 1:
                raise ValueError(f"column {name!r} must be 1-D, got shape {values.shape}")
            cols[str(name)] = values
        if len({len(v) for v in cols.values()}) > 1:
            raise ValueError("columns have different lengths: "
                             + ", ".join(f"{k}={len(v)}" for k, v in cols.items()))
        self.columns = cols
        for what, mapping in (("units", self.units), ("roles", self.roles)):
            unknown = set(mapping) - set(cols)
            if unknown:
                raise ValueError(f"{what} name unknown columns: {sorted(unknown)}")
        self.units = {k: str(self.units.get(k, "")) for k in cols}
        self.roles = {k: self.roles.get(k, "series") for k in cols}
        bad = {k: r for k, r in self.roles.items() if r not in ROLES}
        if bad:
            raise ValueError(f"unknown roles {bad}; roles are {ROLES}")
        if list(self.roles.values()).count("x") > 1:
            raise ValueError("at most one column can have the role 'x'")

        self.bands = [(str(lo), str(hi)) for lo, hi in self.bands]
        in_bands = [c for pair in self.bands for c in pair]
        for lo, hi in self.bands:
            if self.roles.get(lo) != "band_lo" or self.roles.get(hi) != "band_hi":
                raise ValueError(f"band ({lo!r}, {hi!r}) must pair a 'band_lo' column "
                                 "with a 'band_hi' column")
        for name, role in self.roles.items():
            if role in ("band_lo", "band_hi") and in_bands.count(name) != 1:
                raise ValueError(f"column {name!r} ({role}) must be in exactly one band")
        self.title = str(self.title or "")

    @property
    def n_rows(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def __len__(self) -> int:
        return self.n_rows

    # ---- transport: one float64 array + JSON metadata ----------------------

    def to_array(self) -> np.ndarray:
        """(n_rows, n_columns) float64, columns in ``meta()["columns"]`` order."""
        if not self.columns:
            return np.zeros((0, 0))
        return np.column_stack(list(self.columns.values()))

    def meta(self) -> Dict:
        """JSON-serialisable metadata (everything but the values)."""
        return {
            "n_rows": self.n_rows,
            "columns": list(self.columns),
            "units": dict(self.units),
            "roles": dict(self.roles),
            "bands": [list(pair) for pair in self.bands],
            "title": self.title,
        }

    @classmethod
    def from_meta(cls, array: np.ndarray, meta: Dict) -> "DataTable":
        """Inverse of :meth:`to_array` + :meth:`meta`."""
        names = list(meta.get("columns", []))
        array = np.asarray(array, dtype=np.float64).reshape(int(meta.get("n_rows", 0)), len(names))
        return cls(
            columns={n: array[:, i] for i, n in enumerate(names)},
            units=meta.get("units", {}),
            roles=meta.get("roles", {}),
            bands=[tuple(p) for p in meta.get("bands", [])],
            title=meta.get("title", ""),
        )
