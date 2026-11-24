"""topotoolbox core: GridObject, FlowObject and StreamObject methods, and the
swath map object.

Thin wrappers, one process per pytopotoolbox method a user can call on a grid,
a flow network or a stream network (the "Core" of a frontend): each takes the
object, returns new objects or values. The flow and stream objects themselves
are made by ``topotoolbox.flow_object`` / ``topotoolbox.stream_object``
(adapters/topotoolbox), as are fill sinks, filter and the swath track steps.

Registered after adapters/topotoolbox (it reuses its helpers).

Author: B.G.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import topotoolbox as ttb
from rasterio.crs import CRS
from rasterio.enums import Resampling

from ..core import Output, Param, Port, process, register_type
from ..geovector import GeoVector
from .topotoolbox import (
    _crs_fields, _geovector, _point_cells, _row_major, _stream_point_nodes, _track_indices,
    _values,
)

_GRID = "topotoolbox.GridObject"
_FLOW = "topotoolbox.FlowObject"
_STREAM = "topotoolbox.StreamObject"
_POINTS_DOC = "Points, in any CRS (reprojected to the grid's); each falls in one cell."
_RESAMPLINGS = ["nearest", "bilinear", "cubic", "average", "min", "max", "med"]
_RESAMPLING_LABELS = ["Nearest", "Bilinear", "Cubic", "Average", "Minimum", "Maximum", "Median"]


def _grid_port(doc="Grid (e.g. the DEM)."):
    return Port("grid", _GRID, label="Grid", doc=doc)


def _flow_port():
    return Port("flow", _FLOW, label="Flow object", doc="Flow directions (Flow object).")


def _stream_port():
    return Port("stream", _STREAM, label="Stream object", doc="Stream network (Stream object).")


def _like(grid, z):
    """GridObject with ``z`` on the georeference of ``grid``, as float32."""
    return grid.duplicate_with_new_data(np.asarray(z, dtype=np.float32))


def _linear(obj, points, what):
    """Column-major linear indices (topotoolbox's node indices) of the cells
    the GeoVector ``points`` fall in."""
    rows, cols = _point_cells(obj, points, what)
    return np.ravel_multi_index((rows, cols), obj.shape, order=_order(obj))


def _order(obj):
    """Order ("F" or "C") of the linear indices of a flow / stream object
    (FlowObject.order, StreamObject._order)."""
    return getattr(obj, "order", None) or getattr(obj, "_order", None) or "F"


def _points_at(obj, rows, cols, **attrs):
    """geopoints at the centres of cells (rows, cols) of ``obj``."""
    return _geovector(obj, "points", np.asarray(rows), np.asarray(cols), **attrs)


def _mask(grid, shape, what):
    """Boolean array of the nonzero cells of GridObject ``grid``."""
    z = np.nan_to_num(np.asarray(grid.z, dtype=float)) != 0
    if z.shape != shape:
        raise ValueError(f"{what} is {z.shape[0]} x {z.shape[1]} cells, "
                         f"the network's grid {shape[0]} x {shape[1]}")
    return z


# ---- grid objects -----------------------------------------------------------

@process(
    id="topotoolbox.gradient8",
    category="Terrain",
    label="Gradient",
    inputs=[_grid_port("Elevation.")],
    params=[Param("unit", "enum", default="tangent",
                  choices=["tangent", "degree", "radian", "sine", "percent"],
                  choice_labels=["Tangent", "Degrees", "Radians", "Sine", "Percent"],
                  doc="Unit of the gradient.")],
    outputs=[Output("gradient", _GRID, doc="Steepest downhill gradient of each cell (8 neighbours).")],
    impl="library",
)
def gradient8(grid, unit="tangent"):
    """Steepest downhill gradient over the 8 neighbours.

    topotoolbox GridObject.gradient8.
    """
    return grid.gradient8(unit=unit)


@process(
    id="topotoolbox.curvature",
    category="Terrain",
    label="Curvature",
    inputs=[_grid_port("Elevation.")],
    params=[
        Param("ctype", "enum", label="Curvature", default="profc",
              choices=["profc", "planc", "tangc", "meanc", "total"],
              choice_labels=["Profile", "Planform", "Tangential", "Mean", "Total"],
              doc="Which curvature (1/m; total: 1/m²)."),
        Param("meanfilt", "bool", label="Mean filter first", default=False,
              doc="Smooth with a 3 x 3 mean filter before."),
    ],
    outputs=[Output("curvature", _GRID, doc="Curvature of each cell.")],
    impl="library",
)
def curvature(grid, ctype="profc", meanfilt=False):
    """Second derivative of the surface.

    topotoolbox GridObject.curvature.
    """
    return grid.curvature(ctype=ctype, meanfilt=meanfilt)


@process(
    id="topotoolbox.aspect",
    category="Terrain",
    label="Aspect",
    inputs=[_grid_port("Elevation.")],
    params=[Param("classify", "bool", label="Eight classes", default=False,
                  doc="Classes 1 to 8 (N, NE, ... NW) instead of degrees.")],
    outputs=[Output("aspect", _GRID, doc="Direction each cell faces (degrees from north, or class).")],
    impl="library",
)
def aspect(grid, classify=False):
    """Direction each slope faces.

    topotoolbox GridObject.aspect.
    """
    return grid.aspect(classify=classify)


@process(
    id="topotoolbox.hillshade",
    category="Terrain",
    label="Hillshade",
    inputs=[_grid_port("Elevation.")],
    params=[
        Param("azimuth", "float", label="Sun azimuth", unit="°", default=315.0, min=0.0, max=360.0,
              doc="Direction of the light, degrees clockwise from north."),
        Param("altitude", "float", label="Sun altitude", unit="°", default=60.0, min=0.0, max=90.0,
              doc="Height of the light above the horizon."),
        Param("exaggerate", "float", label="Exaggeration", default=1.0, min=0.0,
              doc="Vertical exaggeration."),
    ],
    outputs=[Output("hillshade", _GRID, doc="Shading from 0 (dark) to 1 (lit).")],
    impl="library",
)
def hillshade(grid, azimuth=315.0, altitude=60.0, exaggerate=1.0):
    """Shaded relief.

    topotoolbox GridObject.hillshade.
    """
    return grid.hillshade(azimuth=azimuth, altitude=altitude, exaggerate=exaggerate)


@process(
    id="topotoolbox.evansslope",
    category="Terrain",
    label="Slope (Evans)",
    inputs=[_grid_port("Elevation.")],
    params=[Param("modified", "bool", label="Modified", default=False,
                  doc="Smooth with a 3 x 3 window before (modified Evans).")],
    outputs=[Output("slope", _GRID, doc="Gradient from a quadratic surface fitted to 3 x 3 cells.")],
    impl="library",
)
def evansslope(grid, modified=False):
    """Gradient from Evans' second-order polynomial fit.

    topotoolbox GridObject.evansslope.
    """
    return grid.evansslope(modified=modified)


@process(
    id="topotoolbox.excesstopography",
    category="Terrain",
    label="Excess topography",
    inputs=[_grid_port("Elevation."),
            Port("threshold_grid", _GRID, label="Threshold slopes", optional=True,
                 doc="Threshold slope per cell; replaces the threshold.")],
    params=[
        Param("threshold", "float", label="Threshold slope", default=0.2, min=0.0,
              doc="Slope (tangent) above which relief is in excess."),
        Param("method", "enum", default="fsm2d", choices=["fsm2d", "fmm2d"],
              choice_labels=["Fast sweeping", "Fast marching"], doc="Solver."),
    ],
    outputs=[Output("excess", _GRID, doc="Elevation above the threshold-slope surface.")],
    impl="library",
)
def excesstopography(grid, threshold_grid=None, threshold=0.2, method="fsm2d"):
    """Relief standing above a surface of threshold slope.

    topotoolbox GridObject.excesstopography.
    """
    t = threshold if threshold_grid is None else threshold_grid
    return grid.excesstopography(threshold=t, method=method)


@process(
    id="topotoolbox.identifyflats",
    category="Terrain",
    label="Flats and sills",
    inputs=[_grid_port("Elevation (fill sinks first for flats from filled pits).")],
    outputs=[Output("flats", _GRID, doc="Flat cells (1)."),
             Output("sills", _GRID, doc="Sill cells (1).")],
    impl="library",
)
def identifyflats(grid):
    """Flat areas and their sills.

    topotoolbox GridObject.identifyflats.
    """
    sills, flats = grid.identifyflats(output=["sills", "flats"])
    return {"flats": _like(grid, flats.z), "sills": _like(grid, sills.z)}


@process(
    id="topotoolbox.prominence",
    category="Terrain",
    label="Peak prominence",
    inputs=[_grid_port("Elevation.")],
    params=[Param("tolerance", "float", label="Smallest prominence", unit="m", default=100.0, min=0.0,
                  doc="Stops once the next peak is less prominent (the first is always found).")],
    outputs=[Output("peaks", "geopoints", doc="Peaks; attribute prominence (m).")],
    impl="library",
)
def prominence(grid, tolerance=100.0):
    """Peaks and how far one descends from each before climbing higher.

    topotoolbox GridObject.prominence.
    """
    prom, (x, y) = grid.prominence(tolerance)
    epsg, crs_wkt = _crs_fields(grid.georef)
    xy = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    return GeoVector("points", xy, vertex_attrs={"prominence": np.asarray(prom, dtype=float)},
                     epsg=epsg, crs_wkt=crs_wkt)


@process(
    id="topotoolbox.zscore",
    category="Terrain",
    label="Z-score",
    inputs=[_grid_port()],
    outputs=[Output("zscore", _GRID, doc="(value - mean) / standard deviation.")],
    impl="library",
)
def zscore(grid):
    """Standardised values.

    topotoolbox GridObject.zscore.
    """
    return grid.zscore()


@process(
    id="topotoolbox.dilate",
    category="DEM preparation",
    label="Dilate",
    inputs=[_grid_port()],
    params=[Param("size", "int", label="Window", unit="cells", default=3, min=1,
                  doc="Square window side.")],
    outputs=[Output("dilated", _GRID, doc="Maximum over the window.")],
    impl="library",
)
def dilate(grid, size=3):
    """Grey dilation (maximum filter).

    topotoolbox GridObject.dilate.
    """
    return grid.dilate(size=(int(size), int(size)))


@process(
    id="topotoolbox.erode",
    category="DEM preparation",
    label="Erode",
    inputs=[_grid_port()],
    params=[Param("size", "int", label="Window", unit="cells", default=3, min=1,
                  doc="Square window side.")],
    outputs=[Output("eroded", _GRID, doc="Minimum over the window.")],
    impl="library",
)
def erode(grid, size=3):
    """Grey erosion (minimum filter).

    topotoolbox GridObject.erode.
    """
    return grid.erode(size=(int(size), int(size)))


@process(
    id="topotoolbox.resample",
    category="DEM preparation",
    label="Resample",
    inputs=[_grid_port()],
    params=[
        Param("resolution", "float", label="Cell size", min=0.0,
              doc="New cell size (map units)."),
        Param("resampling", "enum", default="bilinear", choices=_RESAMPLINGS,
              choice_labels=_RESAMPLING_LABELS, doc="How new cells take values."),
    ],
    outputs=[Output("resampled", _GRID, doc="The grid at the new cell size.")],
    impl="library",
)
def resample(grid, resolution, resampling="bilinear"):
    """Change the cell size.

    topotoolbox GridObject.resample.
    """
    return grid.resample(resolution, resampling=Resampling[resampling])


@process(
    id="topotoolbox.reproject",
    category="DEM preparation",
    label="Reproject",
    inputs=[_grid_port()],
    params=[
        Param("epsg", "int", label="EPSG code", default=4326, min=1, doc="Target coordinate system."),
        Param("resolution", "float", label="Cell size (0: automatic)", default=0.0, min=0.0,
              doc="Cell size in the target system's units; 0 = chosen by the library."),
        Param("resampling", "enum", default="bilinear", choices=_RESAMPLINGS,
              choice_labels=_RESAMPLING_LABELS, doc="How new cells take values."),
    ],
    outputs=[Output("reprojected", _GRID, doc="The grid in the new coordinate system.")],
    impl="library",
)
def reproject(grid, epsg=4326, resolution=0.0, resampling="bilinear"):
    """Change the coordinate system.

    topotoolbox GridObject.reproject.
    """
    return grid.reproject(CRS.from_epsg(int(epsg)), resolution=resolution or None,
                          resampling=Resampling[resampling])


@process(
    id="topotoolbox.crop",
    category="DEM preparation",
    label="Crop",
    inputs=[_grid_port()],
    params=[
        Param("mode", "enum", default="percent", choices=["percent", "coordinate", "pixel"],
              choice_labels=["Fraction of the grid (0-1)", "Map coordinates", "Cells"],
              doc="Units of the four edges."),
        Param("left", "float", default=0.0, doc="Left edge."),
        Param("right", "float", default=1.0, doc="Right edge."),
        Param("top", "float", default=0.0, doc="Top edge (fraction / cell: from the top)."),
        Param("bottom", "float", default=1.0, doc="Bottom edge."),
    ],
    outputs=[Output("cropped", _GRID, doc="The part of the grid inside the edges.")],
    impl="library",
)
def crop(grid, mode="percent", left=0.0, right=1.0, top=0.0, bottom=1.0):
    """Cut a rectangle out of the grid.

    topotoolbox GridObject.crop.
    """
    if mode == "pixel":
        left, right, top, bottom = (int(v) for v in (left, right, top, bottom))
    return grid.crop(left, right, top, bottom, mode=mode)


# ---- flow objects -----------------------------------------------------------

@process(
    id="topotoolbox.imposemin",
    category="DEM preparation",
    label="Impose minima",
    inputs=[_flow_port(), _grid_port("Elevation to carve (on the flow object's grid).")],
    params=[Param("minimum_slope", "float", default=0.0, min=0.0,
                  doc="Smallest downstream slope imposed (0: no rise downstream).")],
    outputs=[Output("imposed", _GRID, doc="Elevations never rising downstream.")],
    impl="library",
)
def imposemin(flow, grid, minimum_slope=0.0):
    """Carve elevations so they never rise downstream along the flow.

    topotoolbox.imposemin. A new flow object can be made from the result.
    """
    return ttb.imposemin(flow, grid, minimum_slope=minimum_slope)


@process(
    id="topotoolbox.flow_accumulation",
    category="Flow",
    label="Flow accumulation",
    inputs=[_flow_port(),
            Port("weights", _GRID, optional=True,
                 doc="Amount each cell contributes (e.g. rainfall); default 1 per cell.")],
    params=[Param("units", "enum", default="pixels", choices=["pixels", "mapunits"],
                  choice_labels=["Cells", "Map units² (area)"],
                  doc="Without weights: count of upstream cells, or upstream area.")],
    outputs=[Output("accumulation", _GRID, doc="Accumulated flow (includes the cell itself).")],
    impl="library",
)
def flow_accumulation(flow, weights=None, units="pixels"):
    """What flows through each cell.

    topotoolbox FlowObject.flow_accumulation.
    """
    if weights is not None:
        return flow.flow_accumulation(weights=np.nan_to_num(_values(weights, np.float32)))
    acc = flow.flow_accumulation()
    if units == "mapunits":
        acc = acc.duplicate_with_new_data(np.asarray(acc.z, dtype=np.float32) * flow.cellsize ** 2)
    return acc


@process(
    id="topotoolbox.drainagebasins",
    category="Flow",
    label="Drainage basins",
    inputs=[_flow_port(),
            Port("outlets", "geopoints", optional=True,
                 doc="Outlets of the basins; default every outlet. " + _POINTS_DOC)],
    outputs=[Output("basins", _GRID, doc="Basin label of each cell (0: none).")],
    impl="library",
)
def drainagebasins(flow, outlets=None):
    """The area draining to each outlet.

    topotoolbox FlowObject.drainagebasins.
    """
    idx = None if outlets is None else _linear(flow, outlets, "outlets")
    return flow.drainagebasins(idx)


@process(
    id="topotoolbox.upstream_distance",
    category="Flow",
    label="Distance from the divide",
    inputs=[_flow_port()],
    outputs=[Output("distance", _GRID, doc="Longest flow distance upstream of each cell (map units).")],
    impl="library",
)
def upstream_distance(flow):
    """Flow distance to the farthest upstream ridge.

    topotoolbox FlowObject.upstream_distance.
    """
    return flow.upstream_distance()


@process(
    id="topotoolbox.downstream_distance",
    category="Flow",
    label="Distance to the outlet",
    inputs=[_flow_port()],
    outputs=[Output("distance", _GRID, doc="Flow distance from each cell to its outlet (map units).")],
    impl="library",
)
def downstream_distance(flow):
    """Flow distance to the outlet.

    topotoolbox FlowObject.downstream_distance.
    """
    return flow.downstream_distance()


def _seed_grid(flow, points, cells, what):
    """GridObject on the flow's grid, True at the points' cells or the nonzero
    cells of ``cells``."""
    seed = np.zeros(flow.shape, dtype=bool)
    if points is not None:
        rows, cols = _point_cells(flow, points, what)
        seed[rows, cols] = True
    if cells is not None:
        seed |= _mask(cells, flow.shape, what)
    if not seed.any():
        raise ValueError(f"no {what} given")
    grid = ttb.GridObject()
    grid.z, grid.cellsize, grid.transform, grid.georef = seed, flow.cellsize, flow.transform, flow.georef
    grid.bounds = flow.bounds
    return grid


@process(
    id="topotoolbox.dependencemap",
    category="Flow",
    label="Upslope area of points",
    inputs=[_flow_port(),
            Port("points", "geopoints", optional=True, doc="Start cells. " + _POINTS_DOC),
            Port("cells", _GRID, optional=True, doc="Start cells (nonzero).")],
    outputs=[Output("upslope", _GRID, doc="Cells draining to the start cells (1).")],
    impl="library",
)
def dependencemap(flow, points=None, cells=None):
    """Everything upslope of some cells.

    topotoolbox FlowObject.dependencemap.
    """
    return _as_float(flow.dependencemap(_seed_grid(flow, points, cells, "start cells")))


@process(
    id="topotoolbox.influencemap",
    category="Flow",
    label="Downslope path of points",
    inputs=[_flow_port(),
            Port("points", "geopoints", optional=True, doc="Start cells. " + _POINTS_DOC),
            Port("cells", _GRID, optional=True, doc="Start cells (nonzero).")],
    outputs=[Output("downslope", _GRID, doc="Cells the start cells drain through (1).")],
    impl="library",
)
def influencemap(flow, points=None, cells=None):
    """Everything downslope of some cells.

    topotoolbox FlowObject.influencemap.
    """
    return _as_float(flow.influencemap(_seed_grid(flow, points, cells, "start cells")))


def _as_float(grid):
    return grid.duplicate_with_new_data(np.asarray(grid.z, dtype=np.float32))


@process(
    id="topotoolbox.flowpathextract",
    category="Flow",
    label="Flow path from points",
    inputs=[_flow_port(), Port("points", "geopoints", doc="Start of each path. " + _POINTS_DOC)],
    outputs=[Output("paths", "geolines", doc="The flow path from each point to its outlet.")],
    impl="library",
)
def flowpathextract(flow, points):
    """The path water takes from each point.

    topotoolbox FlowObject.flowpathextract.
    """
    epsg, crs_wkt = _crs_fields(flow.georef)
    xy, offsets = [], [0]
    for idx in _linear(flow, points, "points"):
        rows, cols = np.unravel_index(np.asarray(flow.flowpathextract(int(idx))), flow.shape,
                                      order=_order(flow))
        x, y = flow.transform * (np.asarray(cols, dtype=float) + 0.5, np.asarray(rows, dtype=float) + 0.5)
        if len(x) < 2:
            continue
        xy.append(np.column_stack([x, y]))
        offsets.append(offsets[-1] + len(x))
    if not xy:
        raise ValueError("every point is already an outlet (no path)")
    return GeoVector("lines", np.vstack(xy), feature_offsets=np.arange(len(xy) + 1),
                     part_offsets=np.asarray(offsets), epsg=epsg, crs_wkt=crs_wkt)


@process(
    id="topotoolbox.getoutlets",
    category="Flow",
    label="Outlets",
    inputs=[_flow_port()],
    outputs=[Output("outlets", "geopoints", doc="Cells with no downstream neighbour (edges and sinks).")],
    impl="library",
)
def getoutlets(flow):
    """Where the flow network ends.

    topotoolbox FlowObject.getoutlets.
    """
    rows, cols = np.unravel_index(np.asarray(flow.getoutlets()), flow.shape, order=_order(flow))
    return _points_at(flow, rows, cols)


@process(
    id="topotoolbox.vertdistance2stream",
    category="Flow",
    label="Height above the stream",
    inputs=[_flow_port(), _stream_port(), _grid_port("Elevation.")],
    outputs=[Output("height", _GRID,
                    doc="Height of each cell above the stream it drains to (HAND, map units).")],
    impl="library",
)
def vertdistance2stream(flow, stream, grid):
    """Height above nearest drainage, along flow paths.

    topotoolbox FlowObject.vertdistance2stream.
    """
    return flow.vertdistance2stream(stream, grid)


# ---- stream objects ---------------------------------------------------------

@process(
    id="topotoolbox.clean",
    category="Stream",
    label="Clean",
    inputs=[_stream_port()],
    outputs=[Output("cleaned", _STREAM, doc="The network without disconnected nodes.")],
    impl="library",
)
def clean(stream):
    """Drop nodes connected to nothing.

    topotoolbox StreamObject.clean.
    """
    return stream.clean()


@process(
    id="topotoolbox.removeedgeeffects",
    category="Stream",
    label="Remove edge effects",
    inputs=[_stream_port(), _flow_port(),
            Port("grid", _GRID, label="DEM", optional=True,
                 doc="Its no-data cells count as edges too.")],
    outputs=[Output("trimmed", _STREAM, doc="The network without the reaches fed across the grid edge.")],
    impl="library",
)
def removeedgeeffects(stream, flow, grid=None):
    """Drop the reaches downstream of grid edges.

    topotoolbox StreamObject.removeedgeeffects.
    """
    return stream.removeedgeeffects(flow, grid)


@process(
    id="topotoolbox.streampoi",
    category="Stream",
    label="Points of interest",
    inputs=[_stream_port()],
    params=[Param("point_type", "enum", label="Points", default="channelheads",
                  choices=["channelheads", "confluences", "outlets"],
                  choice_labels=["Channel heads", "Confluences", "Outlets"],
                  doc="Which points of the network.")],
    outputs=[Output("points", "geopoints", doc="The points, at stream cell centres.")],
    impl="library",
)
def streampoi(stream, point_type="channelheads"):
    """Channel heads, confluences or outlets of the network.

    topotoolbox StreamObject.streampoi.
    """
    nodes = np.asarray(stream.streampoi(point_type), dtype=bool)
    rows, cols = stream.node_indices_where(nodes)
    return _points_at(stream, rows, cols)


@process(
    id="topotoolbox.knickpointfinder",
    category="Stream",
    label="Knickpoints",
    inputs=[_stream_port(), _grid_port("Elevation (made to decrease downstream first).")],
    params=[Param("tolerance", "float", unit="m", default=100.0, min=0.0,
                  doc="Smallest elevation offset to the concave profile kept as a knickpoint.")],
    outputs=[Output("knickpoints", "geopoints", doc="Knickpoints; attribute dz (offset, m).")],
    impl="library",
)
def knickpointfinder(stream, grid, tolerance=100.0):
    """Sharp convex steps in the river profiles.

    topotoolbox StreamObject.knickpointfinder.
    """
    res = stream.knickpointfinder(grid, tolerance=tolerance)
    kp = np.asarray(res[0] if isinstance(res, tuple) else res)
    nodes = kp.astype(bool) if kp.dtype == bool or kp.size == stream.stream.size else None
    if nodes is None:
        raise RuntimeError("unexpected knickpointfinder result")
    rows, cols = stream.node_indices_where(nodes)
    attrs = {}
    if isinstance(res, tuple) and len(res) > 1:
        z = np.asarray(stream.ezgetnal(grid), dtype=float)
        attrs["dz"] = (z - np.asarray(res[1], dtype=float))[nodes]
    return _points_at(stream, rows, cols, **attrs)


@process(
    id="topotoolbox.quantcarve",
    category="DEM preparation",
    label="Quantile carving",
    inputs=[_stream_port(), _grid_port("Elevation.")],
    params=[
        Param("tau", "float", label="Quantile", default=0.5, min=0.0, max=1.0,
              doc="Quantile of elevation the profile follows."),
        Param("mingradient", "float", label="Smallest gradient", default=0.0, min=0.0,
              doc="Smallest downstream gradient."),
        Param("fixedoutlet", "bool", label="Fixed outlets", default=False,
              doc="Keep the outlets' elevations."),
    ],
    outputs=[Output("carved", _GRID, doc="Carved elevation on the stream cells (no data elsewhere).")],
    impl="library",
)
def quantcarve(stream, grid, tau=0.5, mingradient=0.0, fixedoutlet=False):
    """Stream profiles following a quantile of elevation, never rising downstream.

    topotoolbox StreamObject.quantcarve.
    """
    zs = stream.quantcarve(grid, tau=tau, mingradient=mingradient, fixedoutlet=fixedoutlet)
    return _nal_grid(stream, zs)


_STREAM_VALUES = ["mask", "streamorder_strahler", "streamorder_shreve", "conncomps",
                  "distance_from_outlet", "upstream_distance", "gradient", "ksn", "chi"]
_STREAM_VALUE_LABELS = ["Stream cells (1)", "Strahler order", "Shreve order", "Network label",
                        "Distance to the outlet", "Distance from the head", "Gradient", "ksn",
                        "Chi"]


@process(
    id="topotoolbox.stream_grid",
    category="Stream",
    label="Stream values as a grid",
    inputs=[_stream_port(),
            Port("flow", _FLOW, label="Flow object", optional=True,
                 doc="The network's flow object; needed for ksn and chi."),
            Port("grid", _GRID, label="DEM", optional=True, doc="Elevation; needed for gradient and ksn.")],
    params=[
        Param("value", "enum", default="mask", choices=_STREAM_VALUES,
              choice_labels=_STREAM_VALUE_LABELS, doc="Value of each stream cell."),
        Param("impose", "bool", label="Impose minima", default=False,
              doc="Gradient and ksn: impose downstream minima first (no negative slopes)."),
        Param("theta", "float", label="ksn concavity", default=0.45, min=0.0, doc="ksn: reference concavity."),
        Param("mn", "float", label="Chi m/n", default=0.45, min=0.0, doc="chi: m/n ratio."),
        Param("a0", "float", label="Chi reference area A0", default=1e6, min=0.0,
              doc="chi: reference area (map units²)."),
    ],
    outputs=[Output("values", _GRID, doc="The value on the stream cells (no data elsewhere).")],
    impl="library",
)
def stream_grid(stream, flow=None, grid=None, value="mask", impose=False, theta=0.45, mn=0.45,
                a0=1e6):
    """A stream network value written to the stream cells of a grid.

    topotoolbox StreamObject.to_grid with streamorder, conncomps, distance,
    upstream_distance, gradient, ksn or chitransform.
    """
    if value in ("gradient", "ksn") and grid is None:
        raise ValueError(f"{value} needs the DEM")
    if value in ("ksn", "chi") and flow is None:
        raise ValueError(f"{value} needs the flow object")
    if value == "mask":
        return _as_float(stream.to_grid())
    if value.startswith("streamorder_"):
        nal = stream.streamorder(method=value.split("_")[1])
    elif value == "conncomps":
        nal = stream.conncomps()
    elif value == "distance_from_outlet":
        nal = stream.downstream_distance()
    elif value == "upstream_distance":
        nal = stream.upstream_distance()
    elif value == "gradient":
        nal = stream.gradient(grid, impose=impose)
    elif value == "ksn":
        nal = stream.ksn(grid, flow.flow_accumulation(), impose=impose, theta=theta)
    else:
        nal = stream.chitransform(flow.flow_accumulation(), a0=a0, mn=mn)
    return _nal_grid(stream, nal)


def _nal_grid(stream, nal):
    """GridObject on the stream's grid: node values ``nal`` on the stream
    cells, NaN elsewhere (StreamObject.to_grid takes a grid-shaped array)."""
    z = np.full(stream.shape, np.nan, dtype=np.float32)
    z[stream.node_indices] = np.asarray(nal, dtype=np.float32)
    return _as_float(stream.to_grid()).duplicate_with_new_data(z)


# ---- swath maps -------------------------------------------------------------

@dataclasses.dataclass
class SwathMap:
    """A swath laid on a grid: the track (one vertex per cell when prepared),
    the signed distance of each cell to it, each cell's nearest track vertex
    and the half-width; the base of the cross- and along-track profiles."""
    track: GeoVector
    distance_map: ttb.GridObject
    nearest_point: ttb.GridObject
    half_width: float

    def describe(self):
        return (f"swath map, half-width {self.half_width:g}, "
                f"{self.track.n_vertices} track vertices")


register_type("topotoolbox.SwathMap", "graph", SwathMap,
              doc="A swath laid on a grid: its track, distance and nearest-vertex grids, half-width.")


@process(
    id="topotoolbox.swath_map",
    category="Swath",
    label="Swath map",
    inputs=[
        Port("grid", _GRID, label="Grid", doc="Grid the swath is laid on (e.g. the DEM)."),
        Port("track", "geolines", label="Line", doc="One line, in any CRS (reprojected to the grid's)."),
        Port("mask", _GRID, optional=True,
             doc="Cells to use: nonzero = used. No-data cells of the grid are never used."),
    ],
    params=[
        Param("half_width", "float", label="Half-width", min=0.0,
              doc="Cells farther from the track are left out (map units)."),
        Param("prepare", "bool", label="Prepare the track", default=True,
              doc="Lay the line on the grid as an unbroken chain of cells first (one "
                  "along-track profile point per cell)."),
        Param("return_centre_line", "bool", label="Centreline", default=False,
              doc="Also find the swath's centreline."),
    ],
    outputs=[
        Output("swath_map", "topotoolbox.SwathMap", doc="The swath, for the swath profiles."),
        Output("prepared", "geolines", doc="The track used (prepared: one vertex per cell)."),
        Output("distance_map", _GRID, doc="Signed distance to the track (+ left, - right; map units)."),
        Output("outline", "geopolygons", doc="The corridor: the track widened by the half-width."),
        Output("centre_line", "geolines", optional=True,
               doc="Centreline of the swath; vertex attribute half_width = its local half-width."),
    ],
    impl="composite",
)
def swath_map(grid, track, half_width, mask=None, prepare=True, return_centre_line=False):
    """Lay a swath on a grid: (prepare the track,) then the distance map.

    Chains topotoolbox.prepare_track -> topotoolbox.compute_swath_distance_map.
    """
    from ..core.registries import PROCESSES
    from ..routines.swath import _outline

    if prepare:
        track = PROCESSES.get("topotoolbox.prepare_track")(grid=grid, track=track)["prepared_track"]
    maps = PROCESSES.get("topotoolbox.compute_swath_distance_map")(
        grid=grid, track=track, mask=mask, half_width=half_width,
        return_centre_line=return_centre_line)
    out = {"swath_map": SwathMap(track, maps["distance_map"], maps["nearest_point"], float(half_width)),
           "prepared": track, "distance_map": maps["distance_map"],
           "outline": _outline(track, half_width)}
    if return_centre_line:
        out["centre_line"] = maps["centre_line"]
    return out


@process(
    id="topotoolbox.swath_point_cells",
    category="Swath",
    label="Cells of an along-track point",
    inputs=[Port("grid", _GRID, label="Grid", doc="Grid the swath is laid on."),
            Port("swath_map", "topotoolbox.SwathMap", label="Swath map", doc="The swath.")],
    params=[
        Param("point_index", "int", label="Track vertex", default=0, min=0,
              doc="Track vertex (0 = first)."),
        Param("binning_distance", "float", label="Along-track half-window", min=0.0,
              doc="Each profile point gathers the cells within this distance of it on either "
                  "side along the track (map units): half the window's length; 0 = only the "
                  "cells nearest to it."),
        Param("windowed", "bool", label="Oriented window", default=False,
              doc="The cells of the oriented-window method instead of the nearest-vertex one."),
        Param("n_points_regression", "int", label="Window: vertices to orient", default=5, min=1,
              doc="Oriented window: track vertices used to orient it."),
    ],
    outputs=[Output("pixels", "geopoints", doc="Centres of the cells the point gathers.")],
    impl="composite",
)
def swath_point_cells(grid, swath_map, point_index=0, binning_distance=0.0, windowed=False,
                      n_points_regression=5):
    """The cells one point of an along-track profile gathers.

    topotoolbox.get_point_pixels or get_windowed_point_samples on the swath.
    """
    from ..core.registries import PROCESSES

    sm = swath_map
    if windowed:
        return PROCESSES.get("topotoolbox.get_windowed_point_samples")(
            grid=grid, track=sm.track, point_index=point_index, half_width=sm.half_width,
            binning_distance=binning_distance, n_points_regression=n_points_regression)
    return PROCESSES.get("topotoolbox.get_point_pixels")(
        grid=grid, track=sm.track, distance_map=sm.distance_map, nearest_point=sm.nearest_point,
        point_index=point_index, half_width=sm.half_width, binning_distance=binning_distance)
