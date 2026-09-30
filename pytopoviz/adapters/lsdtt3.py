"""lsdtt3 adapter: raster IO, depression filling, D8 flow topology, swath
profiles, converters.

Registration unit for lsdtt3 (DESIGN.md §8). Hard-imports lsdtt3; the guard for a
missing library lives in ``adapters/__init__``.

Each process mirrors the lsdtt3 function it wraps: same parameter names, same
defaults. lsdtt3's conventions are kept as they are, not aligned on another
library's.

Author: B.G.
"""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
import lsdtt3
from pyproj import CRS
from shapely.geometry import LineString

from ..core import Output, Param, Port, process, register_converter, register_type
from ..datatable import DataTable
from ..georaster import GeoRaster

# ---- types ------------------------------------------------------------------

register_type("lsdtt3.Raster", "grid", lsdtt3.Raster)
register_type("lsdtt3.FlowInfo", "graph", lsdtt3.FlowInfo)

# North/east/south/west edge codes: open (o), no-flux (n), periodic (p),
# base-level (b). A string-kind param type so workflow validation rejects a
# malformed value before lsdtt3 sees it; params of this type take its doc.
_BC_RE = re.compile(r"^[onpb]{4}$")
register_type(
    "lsdtt3.boundary_conditions",
    "string",
    lambda v: isinstance(v, str) and bool(_BC_RE.match(v)),
    doc="Edges water can leave through (north, east, south, west): "
        "o open, n closed, p periodic, b base level.",
)


# ---- converters -------------------------------------------------------------
# lsdtt3 stores nodata as a sentinel value plus a validity mask; pytopoviz-side
# types use NaN. Both lsdtt3 and GeoRaster are north-up (row 0 = north) with a
# lower-left origin, so the metadata maps one to one.

def _raster_values(r):
    z = np.asarray(r.to_numpy(), dtype=float)
    z[~np.asarray(r.mask_numpy(), dtype=bool)] = np.nan
    return z


def _raster_to_georaster(r):
    m = r.metadata
    return GeoRaster(
        z=_raster_values(r),
        cell_size=m.cell_size,
        x_min=m.x_min,
        y_min=m.y_min,
        epsg=m.epsg_code,
        crs_wkt=m.crs_wkt,
    )


def _georaster_to_raster(geo):
    """GeoRaster -> Raster. A GeoRaster may carry only an EPSG code; lsdtt3 also
    gets its WKT, which it compares CRSs with (e.g. swath baselines)."""
    crs_wkt = geo.crs_wkt or (CRS.from_epsg(geo.epsg).to_wkt() if geo.epsg else "")
    meta = lsdtt3.RasterMeta.from_shape(
        geo.n_rows,
        geo.n_cols,
        x_min=geo.x_min,
        y_min=geo.y_min,
        cell_size=geo.cell_size,
        crs_wkt=crs_wkt,
        epsg_code=geo.epsg,
    )
    valid = geo.valid
    return lsdtt3.Raster.from_numpy(np.where(valid, geo.z, 0.0), valid, meta)


register_converter("lsdtt3.Raster", "field2d", _raster_values)
register_converter("lsdtt3.Raster", "georaster", _raster_to_georaster)
register_converter("georaster", "lsdtt3.Raster", _georaster_to_raster)


# Cross-library bridge: only if topotoolbox surfaced its type. Goes through the
# GeoRaster converters so there is one metadata mapping per library.
from ..core.registries import CONVERTERS, TYPES  # noqa: E402

if TYPES.has("topotoolbox.GridObject"):
    _grid_to_geo = CONVERTERS.get("topotoolbox.GridObject", "georaster").fn
    register_converter(
        "topotoolbox.GridObject",
        "lsdtt3.Raster",
        lambda g: _georaster_to_raster(_grid_to_geo(g)),
    )


# ---- processes --------------------------------------------------------------

@process(
    id="lsdtt3.read_raster",
    label="Read raster (lsdtt3)",
    params=[Param("path", "path", doc="Raster file to open (any GDAL format).")],
    outputs=[Output("raster", "lsdtt3.Raster")],
    impl="library",
)
def read_raster(path):
    """Open a raster file."""
    return lsdtt3.Raster.read(path)


@process(
    id="lsdtt3.fill_pits",
    label="Fill pits",
    inputs=[Port("dem", "lsdtt3.Raster", arg="raster")],
    params=[
        Param("min_slope", "float", default=1e-4, min=0.0,
              doc="Small slope given to filled areas so water keeps flowing."),
        Param("priority_flood_backend", "enum", default="binary",
              choices=["binary", "radix"],
              doc="Filling algorithm variant; same result, speed differs."),
        Param("n_threads", "int", default=0, min=0,
              doc="CPU threads to use; 0 = automatic."),
    ],
    outputs=[Output("filled", "lsdtt3.Raster")],
    impl="library",
)
def fill_pits(raster, min_slope=1e-4, priority_flood_backend="binary", n_threads=0):
    """Fill depressions so water can flow out of every cell."""
    return lsdtt3.fill_pits(
        raster,
        min_slope=min_slope,
        n_threads=n_threads,
        priority_flood_backend=priority_flood_backend,
    )


@process(
    id="lsdtt3.fill_pits_boundary_aware",
    label="Fill pits (boundary aware)",
    inputs=[Port("dem", "lsdtt3.Raster", arg="raster")],
    params=[
        Param("boundary_conditions", "lsdtt3.boundary_conditions", default="oooo"),
        Param("min_slope", "float", default=1e-4, min=0.0,
              doc="Small slope given to filled areas so water keeps flowing."),
        Param("priority_flood_backend", "enum", default="binary",
              choices=["binary", "radix"],
              doc="Filling algorithm variant; same result, speed differs."),
        Param("n_threads", "int", default=0, min=0,
              doc="CPU threads to use; 0 = automatic."),
    ],
    outputs=[Output("filled", "lsdtt3.Raster")],
    impl="library",
)
def fill_pits_boundary_aware(
    raster, boundary_conditions="oooo", min_slope=1e-4,
    priority_flood_backend="binary", n_threads=0,
):
    """Fill depressions, draining only through the open edges."""
    return lsdtt3.fill_pits_boundary_aware(
        raster,
        boundary_conditions=boundary_conditions,
        min_slope=min_slope,
        n_threads=n_threads,
        priority_flood_backend=priority_flood_backend,
    )


@process(
    id="lsdtt3.flow_info",
    label="D8 flow topology",
    inputs=[Port("dem", "lsdtt3.Raster", doc="A depression-free DEM (run Fill pits first).")],
    params=[
        Param("boundary_conditions", "lsdtt3.boundary_conditions", default="oooo"),
        Param("interior_nodata_is_outlet", "bool", default=True,
              doc="Water reaching a no-data hole inside the DEM leaves there."),
        Param("allow_pits_as_outlets", "bool", default=False,
              doc="Let leftover pits end the flow instead of failing."),
        Param("n_threads", "int", default=0, min=0,
              doc="CPU threads to use; 0 = automatic."),
    ],
    outputs=[Output("flow", "lsdtt3.FlowInfo")],
    impl="library",
)
def flow_info(
    dem, boundary_conditions="oooo", interior_nodata_is_outlet=True,
    allow_pits_as_outlets=False, n_threads=0,
):
    """Where each cell sends its water (steepest of its 8 neighbours).

    lsdtt3.FlowInfo: D8 receivers and topological order.
    """
    # forced_outlets (N x 2 row/col array) is not surfaced yet: it needs a
    # point-set data type.
    return lsdtt3.FlowInfo(
        dem,
        boundary_conditions=boundary_conditions,
        interior_nodata_is_outlet=interior_nodata_is_outlet,
        forced_outlets=None,
        allow_pits_as_outlets=allow_pits_as_outlets,
        n_threads=n_threads,
    )


@process(
    id="lsdtt3.drainage_area",
    label="Drainage area",
    inputs=[Port("flow", "lsdtt3.FlowInfo")],
    outputs=[Output("area", "lsdtt3.Raster", doc="Upstream area, in map units squared.")],
    impl="library",
)
def drainage_area(flow):
    """Area draining through each cell."""
    return flow.drainage_area()


@process(
    id="lsdtt3.contributing_pixels",
    label="Contributing pixels",
    inputs=[Port("flow", "lsdtt3.FlowInfo")],
    outputs=[Output("pixels", "lsdtt3.Raster", doc="Upstream cell count.")],
    impl="library",
)
def contributing_pixels(flow):
    """Number of cells draining through each cell."""
    return flow.contributing_pixels()


@process(
    id="lsdtt3.accumulate",
    label="Weighted accumulation",
    inputs=[
        Port("flow", "lsdtt3.FlowInfo"),
        Port("weights", "lsdtt3.Raster", optional=True,
             doc="Amount each cell contributes (e.g. rainfall); cell area if empty."),
    ],
    outputs=[Output("accumulation", "lsdtt3.Raster")],
    impl="library",
)
def accumulate(flow, weights=None):
    """Sum a per-cell quantity along the flow."""
    return flow.accumulate(weights)


# ---- swath profiles ---------------------------------------------------------

_BASELINE_DOC = "One line, in any CRS (reprojected to the raster's)."
_THREADS_DOC = "CPU threads to use; 0 = automatic."


def _baseline(raster, baseline):
    """The single line ``baseline`` as lsdtt3 takes it: a GeoDataFrame of one
    LineString in the raster's CRS (reprojected when both have a CRS)."""
    wkt = raster.metadata.crs_wkt
    if wkt and baseline.crs() is not None:
        baseline = baseline.to_crs(wkt)
    return gpd.GeoDataFrame(geometry=[LineString(baseline.line_xy())], crs=wkt or None)


def _profile_table(profile):
    """DataTable of a SwathProfile's table: along_axis_centre (x), mean and p50
    (series), p25-p75 and p0-p100 (bands), n_pixels (count), the other
    percentiles aux."""
    frame = profile.table
    columns = {str(c): frame[c].to_numpy() for c in frame.columns}
    roles = {c: "aux" for c in columns if re.fullmatch(r"p\d+", c)}
    roles.update({"along_axis_centre": "x", "n_pixels": "count", "mean": "series",
                  "p50": "series", "p25": "band_lo", "p75": "band_hi",
                  "p0": "band_lo", "p100": "band_hi"})
    return DataTable(columns, units={"along_axis_centre": "m"}, roles=roles,
                     bands=[("p0", "p100"), ("p25", "p75")], title="Along-line swath (lsdtt3)")


@process(
    id="lsdtt3.swath_distances",
    label="Swath distances",
    inputs=[Port("raster", "lsdtt3.Raster", doc="Grid whose valid cells get distances."),
            Port("baseline", "geolines", doc=_BASELINE_DOC)],
    params=[Param("n_threads", "int", default=0, min=0, doc=_THREADS_DOC)],
    outputs=[
        Output("along_axis", "lsdtt3.Raster",
               doc="Distance along the line to each cell's nearest point on it (m)."),
        Output("perpendicular", "lsdtt3.Raster", doc="Distance from each cell to the line (m)."),
        Output("signed_perpendicular", "lsdtt3.Raster",
               doc="Distance to the line, positive left of it, negative right (m)."),
    ],
    impl="library",
)
def swath_distances(raster, baseline, n_threads=0):
    """Distance of every cell along and across a line.

    lsdtt3.swath_distances: exact projection of each cell on every segment.
    """
    return lsdtt3.swath_distances(raster, _baseline(raster, baseline), n_threads=n_threads)


@process(
    id="lsdtt3.swath_profile",
    label="Swath profile",
    inputs=[
        Port("reference", "lsdtt3.Raster", doc="Grid defining the cells and the valid area."),
        Port("baseline", "geolines", doc=_BASELINE_DOC),
        Port("values", "lsdtt3.Raster", optional=True,
             doc="Grid to summarise, on the reference's cells; the reference itself if empty."),
    ],
    params=[
        Param("half_width_metres", "float", default=0.0, min=0.0,
              doc="Only cells within this distance of the line are used; 0 = every cell."),
        Param("bin_width_metres", "float", default=1000.0, min=0.0,
              doc="Length of each bin along the line (must be above 0)."),
        Param("n_threads", "int", default=0, min=0, doc=_THREADS_DOC),
    ],
    outputs=[
        Output("profile", "datatable", doc="Mean and percentiles per bin along the line."),
        Output("along_axis", "lsdtt3.Raster",
               doc="Distance along the line of each swath cell (m)."),
        Output("perpendicular_distance", "lsdtt3.Raster",
               doc="Distance from each swath cell to the line (m)."),
        Output("signed_perpendicular_distance", "lsdtt3.Raster",
               doc="Same, positive left of the line, negative right (m)."),
    ],
    impl="library",
)
def swath_profile(reference, baseline, values=None, half_width_metres=0.0,
                  bin_width_metres=1000.0, n_threads=0):
    """Statistics of a grid along a line, bin by bin.

    lsdtt3.swath_profile: each cell goes to the bin of its nearest point on
    the line; mean and percentiles 0-100 by 5 per bin.
    """
    result = lsdtt3.swath_profile(
        reference, _baseline(reference, baseline), values=values,
        half_width_metres=half_width_metres, bin_width_metres=bin_width_metres,
        n_threads=n_threads,
    )
    return {
        "profile": _profile_table(result),
        "along_axis": result.along_axis,
        "perpendicular_distance": result.perpendicular_distance,
        "signed_perpendicular_distance": result.signed_perpendicular_distance,
    }
