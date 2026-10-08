"""lsdtt3 adapter: raster IO, depression filling, D8 flow topology, channel
networks, swath profiles, converters.

Registration unit for lsdtt3 (DESIGN.md §8). Hard-imports lsdtt3; the guard for a
missing library lives in ``adapters/__init__``.

Each process mirrors the lsdtt3 function it wraps: same parameter names, same
defaults. lsdtt3's conventions are kept as they are, not aligned on another
library's.

Author: B.G.
"""

from __future__ import annotations

import dataclasses
import re

import geopandas as gpd
import numpy as np
import lsdtt3
from pyproj import CRS
from shapely.geometry import LineString

from ..core import Output, Param, Port, process, register_converter, register_type
from ..datatable import DataTable
from ..georaster import GeoRaster
from ..geovector import GeoVector

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
    return gpd.GeoDataFrame(geometry=[LineString(baseline.in_crs(wkt).line_xy())], crs=wkt or None)


def _profile_table(profile, line):
    """DataTable of a SwathProfile's table: along_axis_centre (x), mean and p50
    (series), p25-p75 and p0-p100 (bands), n_pixels (count), the other
    percentiles aux, then x, y (map_x, map_y): the point of ``line`` (the
    shapely baseline lsdtt3 got) at each bin centre's distance along it,
    which is what lsdtt3's along-axis distance measures."""
    frame = profile.table
    columns = {str(c): frame[c].to_numpy() for c in frame.columns}
    roles = {c: "aux" for c in columns if re.fullmatch(r"p\d+", c)}
    roles.update({"along_axis_centre": "x", "n_pixels": "count", "mean": "series",
                  "p50": "series", "p25": "band_lo", "p75": "band_hi",
                  "p0": "band_lo", "p100": "band_hi"})
    points = [line.interpolate(d) for d in columns["along_axis_centre"]]
    columns["x"], roles["x"] = np.array([p.x for p in points]), "map_x"
    columns["y"], roles["y"] = np.array([p.y for p in points]), "map_y"
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
        Output("profile", "datatable", doc="Mean and percentiles per bin along the line; x, y = the bin centre on the line."),
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
    frame = _baseline(reference, baseline)
    result = lsdtt3.swath_profile(
        reference, frame, values=values,
        half_width_metres=half_width_metres, bin_width_metres=bin_width_metres,
        n_threads=n_threads,
    )
    return {
        "profile": _profile_table(result, frame.geometry.iloc[0]),
        "along_axis": result.along_axis,
        "perpendicular_distance": result.perpendicular_distance,
        "signed_perpendicular_distance": result.signed_perpendicular_distance,
    }


# ---- channel networks -------------------------------------------------------

register_type("lsdtt3.ChannelNetwork", "graph", lsdtt3.ChannelNetwork)

_POINTS_DOC = "Points, in any CRS (reprojected to the grid's); each falls in one cell."


def _cells_of_points(meta, points, what):
    """(n, 2) int64 (row, col) of the cells the GeoVector ``points`` fall in,
    on the grid of RasterMeta ``meta``; ValueError for a point off the grid or
    for no points (lsdtt3 would take an empty array as none given)."""
    if points.n_vertices == 0:
        raise ValueError(f"no {what} given")
    xy = points.in_crs(meta.crs_wkt).xy
    rows = np.floor((meta.y_max - xy[:, 1]) / meta.cell_size).astype(np.int64)
    cols = np.floor((xy[:, 0] - meta.x_min) / meta.cell_size).astype(np.int64)
    off = (rows < 0) | (rows >= meta.n_rows) | (cols < 0) | (cols >= meta.n_cols)
    if off.any():
        raise ValueError(f"{int(off.sum())} of the {what} fall outside the grid")
    return np.column_stack([rows, cols])


def _cell_points(meta, rows, cols, **attrs):
    """geopoints GeoVector at the centres of cells (rows, cols), in the CRS of
    RasterMeta ``meta``, one feature per cell with ``attrs`` as feature
    attributes."""
    x = meta.x_min + (np.asarray(cols, dtype=float) + 0.5) * meta.cell_size
    y = meta.y_max - (np.asarray(rows, dtype=float) + 0.5) * meta.cell_size
    return GeoVector("points", np.column_stack([x, y]), feature_attrs=attrs,
                     epsg=meta.epsg_code, crs_wkt="" if meta.epsg_code else meta.crs_wkt)


def _vertex_flats(meta, vec):
    """Flat index (row * n_cols + col) of the cell of every vertex of ``vec``
    (vertices on cell centres, in the grid's CRS)."""
    rows = np.floor((meta.y_max - vec.xy[:, 1]) / meta.cell_size).astype(np.int64)
    cols = np.floor((vec.xy[:, 0] - meta.x_min) / meta.cell_size).astype(np.int64)
    return rows * meta.n_cols + cols


def _node_ksn(segments, nodes):
    """ksn of every chi-fit node, from its segment: each segment runs from its
    upstream node down the receivers to its downstream node; a node belongs to
    the segment it is not the downstream end of (a breakpoint goes upstream),
    an outlet to the segment it ends. Returns (flats, ksn)."""
    receiver = dict(zip(nodes["flat"].to_numpy().tolist(), nodes["receiver_flat"].to_numpy().tolist()))
    value = {}
    ends = {}
    for up, down, ksn in zip(segments["upstream_flat"].to_numpy().tolist(),
                             segments["downstream_flat"].to_numpy().tolist(),
                             segments["ksn"].to_numpy().tolist()):
        f = up
        while f != down:
            value[f] = ksn
            nxt = receiver.get(f)
            if nxt is None or nxt == f:
                raise RuntimeError(f"chi-fit segment from node {up} never reaches node {down}")
            f = nxt
        ends.setdefault(down, ksn)
    for f, ksn in ends.items():
        value.setdefault(f, ksn)
    flats = np.fromiter(value.keys(), dtype=np.int64, count=len(value))
    return flats, np.fromiter(value.values(), dtype=np.float64, count=len(value))


def _segments_table(segments, meta):
    """DataTable of the chi-fit segments: downstream chi (x), ksn (series), the
    rest aux, then the map position of each segment's upstream node."""
    columns = {str(c): segments[c].to_numpy() for c in segments.columns}
    roles = {c: "aux" for c in columns}
    roles.update({"downstream_chi": "x", "ksn": "series"})
    up = segments["upstream_flat"].to_numpy()
    rows, cols = up // meta.n_cols, up % meta.n_cols
    columns["x"] = meta.x_min + (cols + 0.5) * meta.cell_size
    columns["y"] = meta.y_max - (rows + 0.5) * meta.cell_size
    roles["x"], roles["y"] = "map_x", "map_y"
    return DataTable(columns, units={"downstream_chi": "m", "upstream_chi": "m",
                                     "downstream_elevation": "m", "upstream_elevation": "m"},
                     roles=roles, title="Chi-fit segments (lsdtt3)")


@process(
    id="lsdtt3.channel_heads",
    label="Channel heads",
    inputs=[Port("dem", "lsdtt3.Raster", doc="The DEM the flow directions come from."),
            Port("flow", "lsdtt3.FlowInfo", doc="Flow directions of the DEM.")],
    params=[
        Param("method", "enum", default="threshold", choices=["threshold", "wiener"],
              choice_labels=["Drainage threshold", "Wiener curvature"],
              doc="Heads where the upstream cell count reaches the threshold, or where the "
                  "Wiener-filtered tangential curvature marks a channel (projected metre DEM)."),
        Param("threshold_contributing_pixels", "int", default=1000, min=1,
              doc="Threshold method: upstream cells needed to start a channel."),
        Param("pruning_drainage_area_m2", "float", default=1000.0, min=0.0,
              doc="Wiener method: channels draining less than this are pruned (m²)."),
        Param("surface_fitting_radius_metres", "float", default=6.0, min=0.0,
              doc="Wiener method: radius of the surface fitted for curvature (m)."),
        Param("connected_components_threshold_pixels", "int", default=100, min=0,
              doc="Wiener method: channel patches smaller than this are dropped (cells)."),
        Param("return_channel_mask", "bool", default=False,
              doc="Wiener method: also return the candidate channel mask."),
    ],
    outputs=[
        Output("heads", "geopoints", doc="One point per channel head, at its cell centre."),
        Output("channel_mask", "lsdtt3.Raster", optional=True,
               doc="Wiener method: cells taken as channel (1) before skeletonising."),
    ],
    impl="library",
)
def channel_heads(dem, flow, method="threshold", threshold_contributing_pixels=1000,
                  pruning_drainage_area_m2=1000.0, surface_fitting_radius_metres=6.0,
                  connected_components_threshold_pixels=100, return_channel_mask=False):
    """Where channels start.

    lsdtt3.extract_channel_heads: by drainage threshold or by Wiener-filtered
    curvature.
    """
    heads = lsdtt3.extract_channel_heads(
        dem, flow, method=method,
        threshold_contributing_pixels=threshold_contributing_pixels,
        pruning_drainage_area_m2=pruning_drainage_area_m2,
        surface_fitting_radius_metres=surface_fitting_radius_metres,
        connected_components_threshold_pixels=connected_components_threshold_pixels,
        diagnostics=bool(return_channel_mask and method == "wiener"),
    )
    pixels = np.asarray(heads.pixels, dtype=np.int64).reshape(-1, 2)
    out = {"heads": _cell_points(flow.metadata, pixels[:, 0], pixels[:, 1])}
    if return_channel_mask and method == "wiener":
        out["channel_mask"] = heads.channel_mask
    return out


@process(
    id="lsdtt3.channel_network",
    label="Channel network",
    inputs=[
        Port("flow", "lsdtt3.FlowInfo", doc="Flow directions to trace channels on."),
        Port("sources", "geopoints", optional=True,
             doc="Channel heads to start from (e.g. Channel heads, or drawn); " + _POINTS_DOC),
        Port("outlets", "geopoints", optional=True,
             doc="Where channels stop; " + _POINTS_DOC),
    ],
    params=[
        Param("threshold_contributing_pixels", "int", default=1000, min=1,
              doc="Upstream cells needed to be a channel (sources: the snapping target)."),
        Param("snap_sources", "bool", default=True,
              doc="Move each source to the nearest channel cell."),
        Param("source_snap_max_radius_pixels", "int", default=25, min=0,
              doc="Farthest a source moves when snapped (cells)."),
        Param("outlet_snap_max_distance_metres", "float", default=0.0, min=0.0,
              doc="Farthest an outlet moves onto a channel (m)."),
        Param("auto_detect_roles", "bool", default=False,
              doc="Tell sources and outlets apart from the flow, whichever port they came in."),
    ],
    outputs=[Output("network", "lsdtt3.ChannelNetwork",
                    doc="Junction graph of the channels, reusable by later runs.")],
    impl="library",
)
def channel_network(flow, sources=None, outlets=None, threshold_contributing_pixels=1000,
                    snap_sources=True, source_snap_max_radius_pixels=25,
                    outlet_snap_max_distance_metres=0.0, auto_detect_roles=False):
    """Channels traced down the flow from a threshold or from given sources.

    lsdtt3 FlowInfo.channel_scope -> FlowInfo.channel_network.
    """
    meta = flow.metadata
    scope = flow.channel_scope(
        threshold_contributing_pixels,
        sources=None if sources is None else _cells_of_points(meta, sources, "sources"),
        outlets=None if outlets is None else _cells_of_points(meta, outlets, "outlets"),
        snap_sources=snap_sources,
        source_snap_max_radius_pixels=source_snap_max_radius_pixels,
        outlet_snap_max_distance_metres=outlet_snap_max_distance_metres,
        auto_detect_roles=auto_detect_roles,
    )
    return flow.channel_network(scope)


@process(
    id="lsdtt3.network_lines",
    label="Channel lines",
    inputs=[
        Port("network", "lsdtt3.ChannelNetwork", doc="The channel network."),
        Port("dem", "lsdtt3.Raster", optional=True,
             doc="Depression-free DEM of the network's flow; needed for ksn."),
    ],
    params=[
        Param("chi", "bool", default=True, doc="Add chi to every vertex."),
        Param("m_over_n", "float", default=0.45, min=0.0, doc="Concavity m/n of chi."),
        Param("reference_area", "float", default=1.0, min=0.0,
              doc="Reference drainage area A0 of chi (m²)."),
        Param("ksn", "bool", default=False,
              doc="Add ksn to every vertex: chi-elevation profiles fitted piecewise, each node "
                  "taking its segment's slope (needs the DEM)."),
        Param("critical_divergence_metres", "float", default=5.0, min=0.0,
              doc="ksn: largest elevation misfit before a profile segment is split (m)."),
        Param("mainstem", "enum", default="max_flow_length", choices=["max_flow_length", "max_chi"],
              choice_labels=["Longest flow path", "Largest chi"],
              doc="ksn: how each basin's main stem is picked for the fit."),
        Param("return_channel_mask", "bool", default=False, doc="Also return the channel cells as a grid."),
        Param("return_junctions", "bool", default=False, doc="Also return the junctions as points."),
    ],
    outputs=[
        Output("lines", "geolines",
               doc="One line per junction-to-junction link (upstream/downstream junction, stream "
                   "order, length, drainage area); vertex values: drainage area, flow distance, "
                   "then chi and ksn when asked."),
        Output("channel_mask", "lsdtt3.Raster", optional=True, doc="Channel cells (1), others 0."),
        Output("junctions", "geopoints", optional=True,
               doc="Junctions (id, receiver junction, donor count, stream order, outlet)."),
        Output("segments", "datatable", optional=True,
               doc="ksn: the fitted chi-elevation segments, with their ksn."),
    ],
    impl="library",
)
def network_lines(network, dem=None, chi=True, m_over_n=0.45, reference_area=1.0, ksn=False,
                  critical_divergence_metres=5.0, mainstem="max_flow_length",
                  return_channel_mask=False, return_junctions=False):
    """The channel network as lines carrying per-link and per-cell values.

    lsdtt3 ChannelNetwork.lines_geodataframe, points_geodataframe, chi,
    ChiCollection.fit.
    """
    if ksn and dem is None:
        raise ValueError("ksn needs the DEM")
    mask = network.channel_mask
    meta = mask.metadata
    vec = CONVERTERS.convert(network.lines_geodataframe(), "geolines",
                             from_type="geopandas.GeoDataFrame")
    vertex_flat = _vertex_flats(meta, vec)

    def per_vertex(flats, values):
        full = np.full(meta.n_pixels, np.nan)
        full[np.asarray(flats, dtype=np.int64)] = values
        return full[vertex_flat]

    cells = network.points_geodataframe()
    attrs = {"drainage_area": per_vertex(cells["flat"], cells["drainage_area"]),
             "flow_distance": per_vertex(cells["flat"], cells["flow_distance_to_outlet"])}
    out = {}
    if chi or ksn:
        collection = network.chi(m_over_n=m_over_n, reference_area=reference_area)
        if chi:
            nodes = collection.nodes()
            attrs["chi"] = per_vertex(nodes["flat"], nodes["chi"])
        if ksn:
            fit = collection.fit(dem, critical_divergence_metres=critical_divergence_metres,
                                 mainstem=mainstem)
            attrs["ksn"] = per_vertex(*_node_ksn(fit.segments, fit.nodes))
            out["segments"] = _segments_table(fit.segments, meta)
    out["lines"] = dataclasses.replace(vec, vertex_attrs={**vec.vertex_attrs, **attrs})
    if return_channel_mask:
        out["channel_mask"] = mask
    if return_junctions:
        j = network.junctions()
        out["junctions"] = _cell_points(
            meta, j["row"], j["col"],
            **{c: j[c].to_numpy() for c in ("junction_id", "receiver_junction", "donor_count",
                                             "stream_order", "is_outlet")})
    return out


# ---- surface metrics --------------------------------------------------------
# lsdtt3.polyfit_metrics: one quadratic fit z = a x² + b y² + c xy + d x + e y + f
# per cell over a window of lambda_metres, every requested metric from the same
# fit (p, q = dz/dx, dz/dy; r, s, t = d²z/dx², d²z/dxdy, d²z/dy²; Florinsky
# 2017 notation). Curvatures: concave-up positive, convex-up negative (the
# mathematical sign, not Zevenbergen & Thorne 1987's).

_POLYFIT_METRICS = [
    ("fitted_surface", "Elevation of the fitted surface: the DEM smoothed at the fit scale (m)."),
    ("dzdx", "Elevation derivative eastwards, p (m/m)."),
    ("dzdy", "Elevation derivative northwards, q (m/m)."),
    ("d2zdx2", "Second derivative eastwards, r (1/m)."),
    ("d2zdxdy", "Mixed second derivative, s (1/m)."),
    ("d2zdy2", "Second derivative northwards, t (1/m)."),
    ("slope", "Slope gradient, √(p² + q²) (m/m)."),
    ("slope_degrees", "Slope angle (degrees)."),
    ("aspect_radians", "Downslope direction, clockwise from north (radians); nodata on flats."),
    ("aspect_degrees", "Downslope direction, clockwise from north (degrees); nodata on flats."),
    ("laplacian", "Laplacian, r + t (1/m)."),
    ("hessian_determinant", "Hessian determinant, r t − s² (1/m²)."),
    ("mean_curvature", "Mean curvature (1/m); concave-up positive."),
    ("gaussian_curvature", "Gaussian curvature (1/m²)."),
    ("maximum_curvature", "Maximum principal curvature, k_max (1/m)."),
    ("minimum_curvature", "Minimum principal curvature, k_min (1/m)."),
    ("curvedness", "Curvedness: how strongly the surface bends, whatever its form (1/m)."),
    ("shape_index", "Shape index: the form of the surface (cap, ridge, saddle, rut, cup), "
                    "whatever its bending (no unit)."),
    ("profile_curvature", "Profile (vertical) curvature along the slope, k_v (1/m); nodata on flats."),
    ("plan_curvature", "Plan curvature, of the contour lines (1/m); nodata on flats."),
    ("tangential_curvature", "Tangential (horizontal) curvature, k_h (1/m); nodata on flats."),
    ("difference_curvature", "Difference curvature, (k_v − k_h) / 2 (1/m); nodata on flats."),
    ("horizontal_excess_curvature", "Horizontal excess curvature, k_h − k_min (1/m); nodata on flats."),
    ("vertical_excess_curvature", "Vertical excess curvature, k_v − k_min (1/m); nodata on flats."),
    ("accumulation_curvature", "Accumulation curvature, k_h k_v (1/m²); nodata on flats."),
    ("ring_curvature", "Ring curvature, k_he k_ve (1/m²); nodata on flats."),
    ("rotor", "Rotor: twisting of the flow lines (1/m); nodata on flats."),
]


@process(
    id="lsdtt3.polyfit_metrics",
    label="Surface metrics (polyfit)",
    inputs=[Port("dem", "lsdtt3.Raster", arg="raster",
                 doc="Elevation, on a projected grid in metres.")],
    params=[
        Param("lambda_metres", "float", default=0.0, min=0.0,
              doc="Fit scale: width of the window the surface is fitted over (m). "
                  "0 = three cells."),
        Param("z_factor", "float", default=1.0, min=0.0,
              doc="Elevations are multiplied by this before the metrics (e.g. 0.3048 "
                  "for feet)."),
        Param("n_threads", "int", default=0, min=0, doc=_THREADS_DOC),
        Param("flat_slope_tolerance", "float", default=1e-12, min=0.0,
              doc="Direction-dependent metrics are nodata where the fitted slope is "
                  "below this (m/m)."),
    ] + [
        Param("return_" + name, "bool", default=name == "slope", doc="Compute: " + doc)
        for name, doc in _POLYFIT_METRICS
    ],
    outputs=[Output(name, "lsdtt3.Raster", optional=True, doc=doc)
             for name, doc in _POLYFIT_METRICS],
    impl="library",
)
def polyfit_metrics(raster, lambda_metres=0.0, z_factor=1.0, n_threads=0,
                    flat_slope_tolerance=1e-12, **returns):
    """Slope, aspect and curvatures from one quadratic surface fit.

    lsdtt3.polyfit_metrics: the metrics whose ``return_<metric>`` is set, all
    from the same fit.
    """
    metrics = [name for name, _doc in _POLYFIT_METRICS if returns.get("return_" + name)]
    if not metrics:
        raise ValueError("polyfit metrics: no metric selected")
    return dict(lsdtt3.polyfit_metrics(
        raster, metrics, lambda_metres=lambda_metres, z_factor=z_factor,
        n_threads=n_threads, flat_slope_tolerance=flat_slope_tolerance,
    ))
