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

    def same_grid(self, meta: Dict) -> bool:
        """True when ``meta`` (as :meth:`meta` gives) describes this grid's
        lattice: shape, cell size, corner and CRS."""
        return (int(meta.get("n_rows", -1)) == self.n_rows
                and int(meta.get("n_cols", -1)) == self.n_cols
                and np.isclose(float(meta.get("cell_size", 0.0)), self.cell_size)
                and np.isclose(float(meta.get("x_min", 0.0)), self.x_min)
                and np.isclose(float(meta.get("y_min", 0.0)), self.y_min)
                and not self._other_crs(meta))

    def onto(self, meta: Dict) -> "GeoRaster":
        """This raster resampled onto the grid ``meta`` describes (reprojected
        when its CRS differs; cells outside this raster are NaN). Values that
        are all integers (labels, masks) take the nearest cell; else a coarser
        target averages the cells it covers and a finer one interpolates
        bilinearly. A raster or target without a CRS is taken to share the
        other's."""
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject

        target = GeoRaster.from_meta(np.zeros((int(meta["n_rows"]), int(meta["n_cols"]))), meta)
        src_crs, dst_crs = self._crs(), target._crs()
        if src_crs is None or dst_crs is None:
            src_crs = dst_crs = src_crs or dst_crs or CRS.from_epsg(3857)
        finite = self.z[np.isfinite(self.z)]
        if finite.size and np.all(finite == np.round(finite)):
            method = Resampling.nearest
        elif target.cell_size > self.cell_size and src_crs == dst_crs:
            method = Resampling.average
        else:
            method = Resampling.bilinear
        z = np.full(target.shape, np.nan)
        src_t = from_origin(self.x_min, self.y_max, self.cell_size, self.cell_size)
        dst_t = from_origin(target.x_min, target.y_max, target.cell_size, target.cell_size)
        reproject(self.z, z, src_transform=src_t, src_crs=src_crs, dst_transform=dst_t,
                  dst_crs=dst_crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=method)
        target.z = z
        return target

    @classmethod
    def read_file(cls, path: str) -> "GeoRaster":
        """Reads band 1 of the raster file at ``path`` (any format GDAL opens):
        nodata becomes NaN, CRS and origin are kept. It must be north-up with
        square cells; anything else is refused."""
        import rasterio

        with rasterio.open(path) as src:
            t = src.transform
            if t.b != 0 or t.d != 0 or t.e >= 0 or not np.isclose(t.a, -t.e):
                raise ValueError(f"{path} is not north-up with square cells: {t}")
            z = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            epsg = src.crs.to_epsg() if src.crs else None
            return cls(z=z, cell_size=t.a, x_min=src.bounds.left, y_min=src.bounds.bottom,
                       epsg=epsg or 0, crs_wkt=src.crs.to_wkt() if src.crs else "")

    def write_geotiff(self, path: str) -> None:
        """Writes the grid to a GeoTIFF at ``path``: float32, nodata NaN, its
        CRS when it has one."""
        import rasterio
        from rasterio.transform import from_origin

        with rasterio.open(path, "w", driver="GTiff", height=self.n_rows, width=self.n_cols,
                           count=1, dtype="float32", nodata=np.nan, crs=self._crs(),
                           transform=from_origin(self.x_min, self.y_max, self.cell_size,
                                                 self.cell_size)) as dst:
            dst.write(self.z.astype(np.float32), 1)

    def _crs(self):
        """The rasterio CRS (``crs_wkt`` first, else ``epsg``); None without one."""
        from rasterio.crs import CRS

        if self.crs_wkt:
            return CRS.from_wkt(self.crs_wkt)
        return CRS.from_epsg(self.epsg) if self.epsg else None

    def _other_crs(self, meta: Dict) -> bool:
        other = GeoRaster(np.zeros((1, 1)), epsg=meta.get("epsg", 0),
                          crs_wkt=meta.get("crs_wkt", ""))._crs()
        mine = self._crs()
        return mine is not None and other is not None and mine != other

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
