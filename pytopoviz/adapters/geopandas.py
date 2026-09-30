"""geopandas adapter: GeoDataFrame <-> GeoVector.

Registration unit for geopandas (DESIGN.md §8): the GeoDataFrame type and its
converters to/from the geometry hubs (geopoints, geolines, geopolygons).
Hard-imports geopandas and shapely; the guard for a missing library is in
``adapters/__init__``.

A GeoDataFrame converts to the hub of its geometry: points/multipoints,
linestrings/multilinestrings, or polygons/multipolygons (holes kept). A frame
holding any other geometry for that hub raises ConversionError. Missing and
empty geometries become features without parts and come back as None. Numeric
columns keep their dtype, other columns come back as text, the index is not
kept. The third coordinate travels as the ``z`` vertex attribute; other vertex
attributes have no GeoDataFrame equivalent and are dropped on the way back.

Author: B.G.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import GeometryType

from ..core import ConversionError, register_converter, register_type
from ..geovector import GEOVECTOR_TYPES, GeoVector

register_type("geopandas.GeoDataFrame", "vector", gpd.GeoDataFrame,
              doc="GeoPandas GeoDataFrame (one geometry column, attribute columns, CRS).")

# shapely (single, multi) geometry types per GeoVector geometry.
_SHAPELY_TYPES = {
    "points": (GeometryType.POINT, GeometryType.MULTIPOINT),
    "lines": (GeometryType.LINESTRING, GeometryType.MULTILINESTRING),
    "polygons": (GeometryType.POLYGON, GeometryType.MULTIPOLYGON),
}


def _frame_to_geovector(frame: gpd.GeoDataFrame, geometry: str) -> GeoVector:
    geoms = np.asarray(frame.geometry.values, dtype=object)
    single, multi = _SHAPELY_TYPES[geometry]
    type_ids = shapely.get_type_id(geoms)
    present = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))
    others = set(type_ids[present].tolist()) - {int(single), int(multi)}
    if others:
        names = sorted(GeometryType(t).name for t in others)
        raise ConversionError(f"GeoDataFrame holds {', '.join(names)} geometries: "
                              f"not convertible to {GEOVECTOR_TYPES[geometry]}")

    parts, index = shapely.get_parts(geoms, return_index=True)
    keep = ~shapely.is_empty(parts)
    parts, index = parts[keep], index[keep]
    has_z = bool(len(parts)) and bool(shapely.has_z(parts).any())
    offsets = {}
    if geometry == "points":
        coords = shapely.get_coordinates(parts, include_z=has_z)
    elif len(parts):
        _, coords, ragged = shapely.to_ragged_array(parts, include_z=has_z)
        offsets["part_offsets"] = ragged[-1]
        if geometry == "polygons":
            offsets["ring_offsets"] = ragged[0]
    else:
        coords = np.zeros((0, 3 if has_z else 2))
        offsets["part_offsets"] = [0]
        if geometry == "polygons":
            offsets["ring_offsets"] = [0]
    counts = np.bincount(index, minlength=len(geoms))
    crs = frame.crs
    epsg = (crs.to_epsg() or 0) if crs is not None else 0
    return GeoVector(
        geometry=geometry,
        xy=coords[:, :2],
        feature_offsets=np.concatenate([[0], np.cumsum(counts)]),
        multi=type_ids == int(multi),
        feature_attrs={str(c): _column_values(frame[c]) for c in frame.columns
                       if c != frame.geometry.name},
        vertex_attrs={"z": coords[:, 2]} if has_z else {},
        epsg=epsg,
        crs_wkt=crs.to_wkt() if crs is not None and not epsg else "",
        **offsets,
    )


def _column_values(series: pd.Series):
    """Numeric columns as numpy arrays (nullable ones as float with NaN), the
    rest as text with None for missing values."""
    if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        values = series.to_numpy()
        if values.dtype.kind not in "biuf":
            values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        return values
    return [None if pd.api.types.is_scalar(v) and pd.isna(v) else str(v) for v in series]


def _geovector_to_frame(vec: GeoVector) -> gpd.GeoDataFrame:
    _, multi = _SHAPELY_TYPES[vec.geometry]
    coords = vec.xy
    if "z" in vec.vertex_attrs:
        coords = np.column_stack([coords, vec.vertex_attrs["z"]])
    offsets = {
        "points": (vec.feature_offsets,),
        "lines": (vec.part_offsets, vec.feature_offsets),
        "polygons": (vec.ring_offsets, vec.part_offsets, vec.feature_offsets),
    }[vec.geometry]
    geoms = shapely.from_ragged_array(multi, coords, offsets)
    n_parts = np.diff(vec.feature_offsets)
    one = ~vec.multi & (n_parts == 1)
    geoms[one] = shapely.get_geometry(geoms[one], 0)
    geoms[n_parts == 0] = None
    return gpd.GeoDataFrame(dict(vec.feature_attrs), geometry=geoms,
                            crs=vec.crs_wkt or vec.epsg or None)


for _geometry, _type_id in GEOVECTOR_TYPES.items():
    register_converter("geopandas.GeoDataFrame", _type_id,
                       lambda f, g=_geometry: _frame_to_geovector(f, g))
    register_converter(_type_id, "geopandas.GeoDataFrame", _geovector_to_frame)
