"""GeoVector: pytopoviz's own library-agnostic vector geometries.

The hub type for points, lines and polygons crossing library or frontend
boundaries, as GeoRaster is for grids. One GeoVector holds one geometry type
(``points``, ``lines`` or ``polygons``), registered as the type ids of
``GEOVECTOR_TYPES``; converters to/from library types live in the adapters.

Layout (the shapely/GeoArrow ragged layout, always in its multi form):
- ``xy``: (n_vertices, 2) float64 coordinates in the GeoVector's CRS.
- ``feature_offsets`` (n_features + 1): the parts of feature i are
  ``feature_offsets[i]:feature_offsets[i + 1]``. For points a part is a vertex.
- ``part_offsets`` (lines, polygons): the vertices of line part j, or the rings
  of polygon part j.
- ``ring_offsets`` (polygons): the vertices of each ring. The first ring of a
  polygon is its exterior, the others are holes; rings are closed (last vertex
  equals the first), as in shapely.
- ``multi`` (n_features, bool): the feature is a Multi* geometry (a single
  geometry has exactly one part). A feature without parts is missing.

Attributes: ``feature_attrs`` {name: numeric array or list of str/None, one
value per feature}; ``vertex_attrs`` {name: float64 array, one value per
vertex} -- a ``z`` vertex attribute holds the third coordinate.

CRS: ``epsg`` (0 = none) and/or ``crs_wkt``; any CRS, ``to_crs`` reprojects.

Author: B.G.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

GEOMETRIES = ("points", "lines", "polygons")

# Registered type id per geometry (adapters/_conceptual.py).
GEOVECTOR_TYPES: Dict[str, str] = {
    "points": "geopoints",
    "lines": "geolines",
    "polygons": "geopolygons",
}

FeatureValues = Union[np.ndarray, List[Optional[str]]]


@dataclass(eq=False)
class GeoVector:
    geometry: str
    xy: np.ndarray
    feature_offsets: Optional[np.ndarray] = None
    part_offsets: Optional[np.ndarray] = None
    ring_offsets: Optional[np.ndarray] = None
    multi: Optional[np.ndarray] = None
    feature_attrs: Dict[str, FeatureValues] = field(default_factory=dict)
    vertex_attrs: Dict[str, np.ndarray] = field(default_factory=dict)
    epsg: int = 0
    crs_wkt: str = ""

    def __post_init__(self) -> None:
        """Validates the layout. An omitted offsets level groups everything
        into one (except points: one feature per vertex by default)."""
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"GeoVector.geometry must be one of {GEOMETRIES}, got {self.geometry!r}")
        self.xy = np.asarray(self.xy, dtype=np.float64).reshape(-1, 2)
        n = len(self.xy)
        levels = {"points": 0, "lines": 1, "polygons": 2}[self.geometry]

        # Innermost first: the vertex count, then each level's length.
        count = n
        if levels == 2:
            self.ring_offsets = _offsets(self.ring_offsets, count, "ring_offsets")
            count = len(self.ring_offsets) - 1
        elif self.ring_offsets is not None:
            raise ValueError(f"{self.geometry} have no ring_offsets")
        if levels >= 1:
            self.part_offsets = _offsets(self.part_offsets, count, "part_offsets")
            count = len(self.part_offsets) - 1
        elif self.part_offsets is not None:
            raise ValueError("points have no part_offsets")
        if self.feature_offsets is None and levels == 0:
            self.feature_offsets = np.arange(count + 1, dtype=np.int64)
        self.feature_offsets = _offsets(self.feature_offsets, count, "feature_offsets")
        n_parts = np.diff(self.feature_offsets)

        if self.multi is None:
            self.multi = n_parts > 1
        self.multi = np.asarray(self.multi, dtype=bool).reshape(-1)
        if len(self.multi) != self.n_features:
            raise ValueError(f"multi has {len(self.multi)} values for {self.n_features} features")
        if np.any(~self.multi & (n_parts > 1)):
            raise ValueError("a single (not multi) feature has more than one part")

        if self.geometry == "lines" and np.any(np.diff(self.part_offsets) < 2):
            raise ValueError("a line part has fewer than 2 vertices")
        if self.geometry == "polygons":
            first, last = self.ring_offsets[:-1], self.ring_offsets[1:] - 1
            if np.any(last - first < 3) or not np.array_equal(self.xy[first], self.xy[last]):
                raise ValueError("a polygon ring is not closed or has fewer than 4 vertices")

        attrs = {}
        for name, values in self.feature_attrs.items():
            if isinstance(values, np.ndarray) and values.dtype.kind in "biuf":
                values = values.reshape(-1)
            else:
                values = [None if v is None else str(v) for v in values]
            if len(values) != self.n_features:
                raise ValueError(f"feature attribute {name!r} has {len(values)} values "
                                 f"for {self.n_features} features")
            attrs[str(name)] = values
        self.feature_attrs = attrs
        vattrs = {}
        for name, values in self.vertex_attrs.items():
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            if len(values) != n:
                raise ValueError(f"vertex attribute {name!r} has {len(values)} values for {n} vertices")
            vattrs[str(name)] = values
        self.vertex_attrs = vattrs
        self.epsg = int(self.epsg or 0)
        self.crs_wkt = str(self.crs_wkt or "")

    @property
    def n_features(self) -> int:
        return len(self.feature_offsets) - 1

    @property
    def n_vertices(self) -> int:
        return len(self.xy)

    def line_xy(self) -> np.ndarray:
        """The (n, 2) vertices of the one line this GeoVector holds; ValueError
        unless it is lines with exactly one feature of one part."""
        if self.geometry != "lines" or self.n_features != 1 or len(self.part_offsets) != 2:
            parts = len(self.part_offsets) - 1 if self.part_offsets is not None else self.n_vertices
            raise ValueError(f"expected a single line, got {self.n_features} {self.geometry} "
                             f"feature(s) with {parts} part(s)")
        return self.xy

    def crs(self):
        """The pyproj CRS (``crs_wkt`` first, else ``epsg``); None without one."""
        from pyproj import CRS

        if self.crs_wkt:
            return CRS.from_wkt(self.crs_wkt)
        return CRS.from_epsg(self.epsg) if self.epsg else None

    def to_crs(self, crs) -> "GeoVector":
        """This GeoVector reprojected to ``crs`` (anything pyproj's
        ``CRS.from_user_input`` takes: EPSG code, WKT, CRS). Vertex ``z`` is
        kept as is. Returns self when already in ``crs``."""
        from pyproj import CRS, Transformer

        src = self.crs()
        if src is None:
            raise ValueError("GeoVector has no CRS to reproject from")
        dst = CRS.from_user_input(crs)
        if src == dst:
            return self
        x, y = Transformer.from_crs(src, dst, always_xy=True).transform(self.xy[:, 0], self.xy[:, 1])
        epsg = dst.to_epsg() or 0
        return dataclasses.replace(self, xy=np.column_stack([x, y]), epsg=epsg,
                                   crs_wkt="" if epsg else dst.to_wkt())

    def in_crs(self, crs) -> "GeoVector":
        """Reprojected to ``crs`` when both it and this GeoVector have a CRS
        (``crs`` empty / None / 0: none); otherwise self, coordinates taken
        as already in it."""
        if not crs or self.crs() is None:
            return self
        return self.to_crs(crs)

    # ---- transport: one float64 array + JSON metadata ----------------------

    def to_array(self) -> np.ndarray:
        """(n_vertices, 2 + n_vertex_attrs) float64: x, y, then the vertex
        attributes in ``meta()["vertex_columns"]`` order."""
        return np.column_stack([self.xy, *self.vertex_attrs.values()]) if self.vertex_attrs \
            else self.xy.copy()

    def meta(self) -> Dict:
        """JSON-serialisable metadata (everything but the array); NaN feature
        values are written as null."""
        out = {
            "geometry": self.geometry,
            "n_vertices": self.n_vertices,
            "vertex_columns": ["x", "y", *self.vertex_attrs],
            "feature_offsets": self.feature_offsets.tolist(),
            "multi": self.multi.tolist(),
            "feature_attrs": [_attr_meta(k, v) for k, v in self.feature_attrs.items()],
            "epsg": self.epsg,
            "crs_wkt": self.crs_wkt,
        }
        if self.part_offsets is not None:
            out["part_offsets"] = self.part_offsets.tolist()
        if self.ring_offsets is not None:
            out["ring_offsets"] = self.ring_offsets.tolist()
        return out

    @classmethod
    def from_meta(cls, array: np.ndarray, meta: Dict) -> "GeoVector":
        """Inverse of :meth:`to_array` + :meth:`meta`."""
        columns = meta.get("vertex_columns", ["x", "y"])
        array = np.asarray(array, dtype=np.float64).reshape(-1, len(columns))
        return cls(
            geometry=meta["geometry"],
            xy=array[:, :2],
            feature_offsets=meta.get("feature_offsets"),
            part_offsets=meta.get("part_offsets"),
            ring_offsets=meta.get("ring_offsets"),
            multi=meta.get("multi"),
            feature_attrs={a["name"]: _attr_values(a) for a in meta.get("feature_attrs", [])},
            vertex_attrs={c: array[:, i] for i, c in enumerate(columns) if i >= 2},
            epsg=meta.get("epsg", 0),
            crs_wkt=meta.get("crs_wkt", ""),
        )


def _offsets(offsets, count: int, name: str) -> np.ndarray:
    """Offsets into ``count`` items: starts at 0, non-decreasing, ends at count.
    None means a single group of all of them."""
    if offsets is None:
        return np.array([0, count], dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if len(offsets) < 1 or offsets[0] != 0 or offsets[-1] != count or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{name} must start at 0, not decrease and end at {count}")
    return offsets


def _attr_meta(name: str, values: FeatureValues) -> Dict:
    if isinstance(values, np.ndarray):
        out = values.tolist()
        if values.dtype.kind == "f":
            out = [None if v != v else v for v in out]
        return {"name": name, "dtype": values.dtype.name, "values": out}
    return {"name": name, "dtype": "text", "values": list(values)}


def _attr_values(attr: Dict) -> FeatureValues:
    if attr["dtype"] == "text":
        return list(attr["values"])
    dtype = np.dtype(attr["dtype"])
    if dtype.kind == "f":
        return np.array([np.nan if v is None else v for v in attr["values"]], dtype=dtype)
    return np.array(attr["values"], dtype=dtype)
