"""GeoRaster: pytopoviz's own library-agnostic georeferenced grid.

The hub type for grids crossing library or frontend boundaries. Each grid-kind
library type registers a converter to/from ``georaster`` (see adapters), so a
transport only ever has to (de)serialise this one container.

Conventions:
- ``z`` is a 2D float array, north-up: row 0 is the northern edge.
- nodata is NaN (never a sentinel value).
- ``x_min``/``y_min`` are the lower-left corner of the grid extent (not a cell
  centre); cells are square with side ``cell_size``.

Author: B.G.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class GeoRaster:
    z: np.ndarray
    cell_size: float = 1.0
    x_min: float = 0.0
    y_min: float = 0.0
    epsg: int = 0
    crs_wkt: str = ""

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, dtype=float)
        if self.z.ndim != 2:
            raise ValueError(f"GeoRaster.z must be 2D, got shape {self.z.shape}")
        self.cell_size = float(self.cell_size)
        self.x_min = float(self.x_min)
        self.y_min = float(self.y_min)
        self.epsg = int(self.epsg or 0)
        self.crs_wkt = str(self.crs_wkt or "")

    @property
    def shape(self):
        return self.z.shape

    @property
    def n_rows(self) -> int:
        return self.z.shape[0]

    @property
    def n_cols(self) -> int:
        return self.z.shape[1]

    @property
    def x_max(self) -> float:
        return self.x_min + self.n_cols * self.cell_size

    @property
    def y_max(self) -> float:
        return self.y_min + self.n_rows * self.cell_size

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.z)

    def meta(self) -> Dict:
        """JSON-serialisable metadata (everything but ``z``)."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "cell_size": self.cell_size,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "epsg": self.epsg,
            "crs_wkt": self.crs_wkt,
        }

    @classmethod
    def from_meta(cls, z: np.ndarray, meta: Optional[Dict] = None) -> "GeoRaster":
        """Inverse of :meth:`meta`; missing keys fall back to defaults."""
        meta = meta or {}
        z = np.asarray(z, dtype=float)
        if "n_rows" in meta and "n_cols" in meta:
            z = z.reshape(int(meta["n_rows"]), int(meta["n_cols"]))
        return cls(
            z=z,
            cell_size=meta.get("cell_size", 1.0),
            x_min=meta.get("x_min", 0.0),
            y_min=meta.get("y_min", 0.0),
            epsg=meta.get("epsg", 0),
            crs_wkt=meta.get("crs_wkt", ""),
        )
