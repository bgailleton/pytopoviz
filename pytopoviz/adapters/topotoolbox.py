"""topotoolbox adapter: surface types, converters and processes.

Registration unit for topotoolbox (DESIGN.md §8). Hard-imports topotoolbox; the
guard for a missing library is in ``adapters/__init__``.

Author: B.G.
"""

from __future__ import annotations

import dataclasses
import os
import re
import urllib.request

import numpy as np
import topotoolbox as ttb
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
from topotoolbox.utils import DEM_NAMES as _DEM_NAMES_URL

from ..core import Output, Param, Port, process, register_converter, register_type
from ..core.registries import CONVERTERS
from ..datatable import DataTable
from ..georaster import GeoRaster
from ..geovector import GeoVector

# ---- types ------------------------------------------------------------------

register_type("topotoolbox.GridObject", "grid", ttb.GridObject)


# ---- converters -------------------------------------------------------------
# GridObject -> raw 2D float field (conceptual `field2d`, registered in
# adapters/_conceptual), so array-consuming processes/frontends can accept a grid.

register_converter(
    "topotoolbox.GridObject", "field2d", lambda g: np.asarray(g.z)
)


def _crs_fields(crs):
    """(epsg, crs_wkt) of a rasterio CRS: its EPSG code when it has one, else
    its WKT (0, "" without a CRS)."""
    epsg = (crs.to_epsg() or 0) if crs is not None else 0
    return epsg, crs.to_wkt() if crs is not None and not epsg else ""


def _grid_to_georaster(grid):
    """GridObject -> GeoRaster. GridObject is north-up with NaN nodata already."""
    b = grid.bounds
    epsg, crs_wkt = _crs_fields(grid.georef)
    return GeoRaster(
        z=np.array(grid.z, dtype=float),
        cell_size=float(grid.cellsize),
        x_min=b.left if b is not None else 0.0,
        y_min=b.bottom if b is not None else 0.0,
        epsg=epsg,
        crs_wkt=crs_wkt,
    )


def _georaster_to_grid(geo):
    """GeoRaster -> GridObject, rebuilding bounds/transform/CRS."""
    grid = ttb.GridObject()
    grid.z = np.ascontiguousarray(geo.z, dtype=np.float32)
    grid.cellsize = geo.cell_size
    grid.bounds = BoundingBox(geo.x_min, geo.y_min, geo.x_max, geo.y_max)
    grid.transform = Affine(geo.cell_size, 0.0, geo.x_min, 0.0, -geo.cell_size, geo.y_max)
    if geo.epsg:
        grid.georef = CRS.from_epsg(geo.epsg)
    elif geo.crs_wkt:
        grid.georef = CRS.from_wkt(geo.crs_wkt)
    return grid


register_converter("topotoolbox.GridObject", "georaster", _grid_to_georaster)
register_converter("georaster", "topotoolbox.GridObject", _georaster_to_grid)


# ---- processes --------------------------------------------------------------

@process(
    id="topotoolbox.read_tif",
    label="Read GeoTIFF",
    params=[Param("path", "path", doc="GeoTIFF file to read.")],
    outputs=[Output("grid", "topotoolbox.GridObject")],
    impl="library",
)
def read_tif(path):
    """Read a GeoTIFF into a topotoolbox GridObject."""
    return ttb.read_tif(path)


def _sample_dem_names():
    """Names of the TopoToolbox/DEMs samples: the repository's list (what
    ``ttb.get_dem_names()`` reads, with a timeout), or, when it cannot be
    reached, the samples already in topotoolbox's cache -- those load offline."""
    try:
        with urllib.request.urlopen(_DEM_NAMES_URL, timeout=5.0) as f:
            return f.read().decode().split()
    except OSError:
        cache = ttb.utils.get_save_location()
        files = os.listdir(cache) if os.path.isdir(cache) else []
        return sorted(os.path.splitext(f)[0] for f in files if f.endswith(".tif"))


_SAMPLE_DEMS = _sample_dem_names()


@process(
    id="topotoolbox.load_dem",
    label="Load sample DEM",
    params=[Param("name", "string",
                  default="bigtujunga" if "bigtujunga" in _SAMPLE_DEMS or not _SAMPLE_DEMS
                  else _SAMPLE_DEMS[0],
                  choices=_SAMPLE_DEMS or None,
                  doc="Name of a DEM in the TopoToolbox/DEMs repository.")],
    outputs=[Output("grid", "topotoolbox.GridObject")],
    impl="library",
)
def load_dem(name="bigtujunga"):
    """Download (and cache) a sample DEM from the TopoToolbox/DEMs repository."""
    return ttb.load_dem(name)


@process(
    id="topotoolbox.gaussian_smooth",
    label="Gaussian smooth",
    inputs=[Port("dem", "topotoolbox.GridObject")],
    params=[
        Param("sigma", "float", default=2.0, min=0.0,
              doc="Blur radius, in cells."),
        Param("mode", "string", default="nearest",
              choices=["reflect", "constant", "nearest", "mirror", "wrap"],
              doc="How edges are padded."),
    ],
    outputs=[Output("smoothed", "topotoolbox.GridObject")],
    impl="library",
)
def gaussian_smooth(dem, sigma=2.0, mode="nearest"):
    """Blur the DEM (ignores no-data cells)."""
    data = np.asarray(dem.z, dtype=float)
    finite = np.isfinite(data)
    filled = np.where(finite, data, 0.0)
    smoothed = gaussian_filter(filled, sigma=sigma, mode=mode)
    weights = gaussian_filter(finite.astype(float), sigma=sigma, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = smoothed / weights
    result[weights == 0.0] = np.nan
    return dem.duplicate_with_new_data(result.astype(np.float32))


@process(
    id="topotoolbox.fillsinks",
    label="Fill sinks",
    inputs=[Port("dem", "topotoolbox.GridObject")],
    params=[Param("hybrid", "bool", default=True,
                  doc="Faster algorithm that uses more memory.")],
    outputs=[Output("filled", "topotoolbox.GridObject")],
    impl="library",
)
def fillsinks(dem, hybrid=True):
    """Fill depressions so water can flow out of every cell.

    topotoolbox GridObject.fillsinks.
    """
    return dem.fillsinks(hybrid=hybrid)


@process(
    id="topotoolbox.filter",
    label="Filter DEM",
    inputs=[Port("dem", "topotoolbox.GridObject")],
    params=[
        Param(
            "method",
            "enum",
            default="mean",
            choices=["mean", "average", "median", "sobel", "scharr", "wiener", "std"],
            doc="What the window computes.",
        ),
        Param("kernelsize", "int", default=3, min=1,
              doc="Window size in cells (odd; 3 for sobel and scharr)."),
    ],
    outputs=[Output("filtered", "topotoolbox.GridObject")],
    impl="library",
)
def filter_dem(dem, method="mean", kernelsize=3):
    """Smooth or edge-detect the DEM with a moving window.

    topotoolbox GridObject.filter.
    """
    return dem.filter(method=method, kernelsize=kernelsize)


# ---- swath profiles ---------------------------------------------------------
# topotoolbox.swath takes a track as row/col indices or map coordinates. Its
# coordinate mode maps a coordinate to cell corners, half a cell off the cell
# centres (a line through the centres of column k comes out half a cell from
# it), and its coordinate outputs are corners. So lines go in as fractional
# (row, col) with cell centres on whole indices ("indices2D" mode), after
# reprojection to the grid's CRS, and index outputs come back as cell-centre
# coordinates. Its results also differ on a column-major copy of the same
# grid, so grids go in row-major.

_PERCENTILES_RE = re.compile(r"^\s*(\d+\s*(,\s*\d+\s*)*)?$")


def _percentile_list(text):
    """'10, 90' -> [10, 90] (sorted, without repeats); '' -> []."""
    return sorted({int(p) for p in text.split(",")}) if text.strip() else []


register_type(
    "topotoolbox.percentiles",
    "string",
    lambda v: isinstance(v, str) and bool(_PERCENTILES_RE.match(v))
    and all(p <= 100 for p in _percentile_list(v)),
    doc="Extra percentiles per bin: whole numbers from 0 to 100 separated by commas "
        "(e.g. \"10, 90\"); empty for none.",
)

_GRID_DOC = "Grid to profile (e.g. the DEM)."
_TRACK_DOC = "One line, in any CRS (reprojected to the grid's)."
_HALF_WIDTH_DOC = "Swath half-width: cells farther from the track are left out (map units)."
_DISTANCE_MAP_DOC = "Signed distance map of the track (Swath distance map)."
_NEAREST_DOC = "Nearest-vertex grid of the same track (Swath distance map)."
_BINNING_DOC = ("Each profile point gathers the cells within this distance of it along the "
                "track (map units).")
_SKIP_DOC = "Keep every n-th track vertex as a profile point."
_REGRESSION_DOC = "Track vertices used to orient each window."


def _row_major(grid):
    """``grid`` with row-major (C-ordered) values."""
    if grid.z.flags.c_contiguous:
        return grid
    return grid.duplicate_with_new_data(np.ascontiguousarray(grid.z))


def _values(grid, dtype):
    """Row-major values of a GridObject (or array) as ``dtype``."""
    z = grid.z if isinstance(grid, ttb.GridObject) else grid
    return np.ascontiguousarray(z, dtype=dtype)


def _track_indices(grid, track):
    """(rows, cols) of the single line ``track`` on ``grid``: fractional indices
    with cell centres on whole numbers. The line is reprojected to the grid's
    CRS when both have one."""
    xy = track.in_crs(grid.georef.to_wkt() if grid.georef is not None else None).line_xy()
    cols, rows = ~grid.transform * (xy[:, 0], xy[:, 1])
    return np.asarray(rows) - 0.5, np.asarray(cols) - 0.5


def _cell_centres(grid, rows, cols):
    """(n, 2) map coordinates of the centres of cells (rows, cols)."""
    x, y = grid.transform * (np.asarray(cols, dtype=float) + 0.5,
                             np.asarray(rows, dtype=float) + 0.5)
    return np.column_stack([x, y])


def _geovector(grid, geometry, rows, cols, **vertex_attrs):
    """GeoVector through the centres of cells (rows, cols), in the grid's CRS:
    one line, or one point per cell."""
    epsg, crs_wkt = _crs_fields(grid.georef)
    return GeoVector(geometry, _cell_centres(grid, rows, cols), vertex_attrs=vertex_attrs,
                     epsg=epsg, crs_wkt=crs_wkt)


def _swath_table(stats, x_name, x, percentiles, title, map_xy=None):
    """DataTable of a Transverse/LongitudinalSwath: ``x_name`` (x), mean and
    median (series), min-max and q1-q3 (bands), the requested percentiles
    (p with 100-p as a band, the others series), std (aux), count, then the
    (n, 2) ``map_xy`` of each row as x, y (map_x, map_y)."""
    nan = np.full(len(x), np.nan)
    columns = {x_name: x, "mean": stats.means,
               "median": stats.medians if stats.medians is not None else nan,
               "min": stats.mins, "max": stats.maxs,
               "q1": stats.q1 if stats.q1 is not None else nan,
               "q3": stats.q3 if stats.q3 is not None else nan}
    roles = {x_name: "x", "min": "band_lo", "max": "band_hi", "q1": "band_lo", "q3": "band_hi"}
    bands = [("min", "max"), ("q1", "q3")]
    for p in percentiles:
        name = f"p{p}"
        columns[name] = (stats.percentiles or {}).get(p, nan)
        if p < 50 and 100 - p in percentiles:
            roles[name], roles[f"p{100 - p}"] = "band_lo", "band_hi"
            bands.append((name, f"p{100 - p}"))
        elif name not in roles:
            roles[name] = "series"
    columns["std"], roles["std"] = stats.stddevs, "aux"
    columns["count"], roles["count"] = stats.counts, "count"
    if map_xy is not None:
        columns["x"], roles["x"] = map_xy[:, 0], "map_x"
        columns["y"], roles["y"] = map_xy[:, 1], "map_y"
    return DataTable(columns, units={x_name: "m"}, roles=roles, bands=bands, title=title)


@process(
    id="topotoolbox.prepare_track",
    label="Prepare swath track",
    inputs=[Port("grid", "topotoolbox.GridObject", doc="Grid the track is laid on."),
            Port("track", "geolines", doc=_TRACK_DOC)],
    outputs=[Output("prepared_track", "geolines",
                    doc="The track through every cell it crosses, one vertex per cell centre.")],
    impl="library",
)
def prepare_track(grid, track):
    """Lay a line on the grid as an unbroken chain of cells.

    topotoolbox.prepare_track (Bresenham gap filling). The along-track swath
    gives one profile point per track vertex, so its track is prepared first.
    """
    grid = _row_major(grid)
    rows, cols = ttb.prepare_track(grid, *_track_indices(grid, track), input_mode="indices2D")
    return _geovector(grid, "lines", rows, cols)


@process(
    id="topotoolbox.rasterize_path",
    label="Rasterize path",
    inputs=[Port("grid", "topotoolbox.GridObject", doc="Grid the path is laid on."),
            Port("track", "geolines", doc=_TRACK_DOC)],
    params=[
        Param("close_loop", "bool", default=False,
              doc="Also join the last vertex back to the first."),
        Param("use_d4", "bool", default=False,
              doc="Step only between cells sharing a side (4 neighbours) instead of 8."),
    ],
    outputs=[Output("path", "geolines", doc="The cells along the line, one vertex per cell centre.")],
    impl="library",
)
def rasterize_path(grid, track, close_loop=False, use_d4=False):
    """Turn a line into the chain of cells it crosses.

    topotoolbox.rasterize_path (Bresenham lines between the vertices).
    """
    grid = _row_major(grid)
    rows, cols = ttb.rasterize_path(grid, *_track_indices(grid, track), input_mode="indices2D",
                                    close_loop=close_loop, use_d4=use_d4)
    return _geovector(grid, "lines", rows, cols)


@process(
    id="topotoolbox.simplify_line",
    label="Simplify line",
    inputs=[Port("grid", "topotoolbox.GridObject", doc="Grid the line is laid on."),
            Port("track", "geolines", doc=_TRACK_DOC)],
    params=[
        Param("tolerance", "float", default=1.0, min=0.0,
              doc="Fixed number: how many vertices to keep. Automatic: unused. "
                  "Area threshold: smallest triangle kept, in cells squared."),
        Param("method", "int", default=0, choices=[0, 1, 2],
              choice_labels=["Fixed number of vertices", "Automatic (knee of the error curve)",
                             "Area threshold (Visvalingam-Whyatt)"],
              doc="How vertices are removed; the first and last are always kept."),
    ],
    outputs=[Output("simplified", "geolines", doc="The line with fewer vertices.")],
    impl="library",
)
def simplify_line(grid, track, tolerance=1.0, method=0):
    """Keep fewer vertices of a line.

    topotoolbox.simplify_line: iterative end-point fit to a fixed count or to
    the knee of its error curve, or Visvalingam-Whyatt area threshold.
    """
    grid = _row_major(grid)
    rows, cols = ttb.simplify_line(grid, *_track_indices(grid, track), tolerance=tolerance,
                                   method=method, input_mode="indices2D")
    return _geovector(grid, "lines", rows, cols)


@process(
    id="topotoolbox.compute_swath_distance_map",
    label="Swath distance map",
    inputs=[
        Port("grid", "topotoolbox.GridObject", doc="Grid the swath is laid on (e.g. the DEM)."),
        Port("track", "geolines",
             doc=_TRACK_DOC + " Prepare it first for an along-track profile per cell."),
        Port("mask", "topotoolbox.GridObject", optional=True,
             doc="Cells to use: nonzero = used. No-data cells of the grid are never used."),
    ],
    params=[
        Param("half_width", "float", optional=True, min=0.0,
              doc=_HALF_WIDTH_DOC + " Empty: no limit."),
        Param("compute_signed", "bool", default=True,
              doc="Distances positive left of the track and negative right; unsigned otherwise."),
        Param("return_centre_line", "bool", default=False,
              doc="Also find the swath's centreline (needs a half-width)."),
    ],
    outputs=[
        Output("distance_map", "topotoolbox.GridObject",
               doc="Distance from each cell to the track (map units); no data outside the swath."),
        Output("nearest_point", "topotoolbox.GridObject",
               doc="Index of each cell's nearest track vertex (-1 outside); used by the "
                   "along-track profile."),
        Output("dist_from_boundary", "topotoolbox.GridObject", optional=True,
               doc="Distance from each swath cell to the swath edge (with the centreline)."),
        Output("centre_line", "geolines", optional=True,
               doc="Centreline of the swath; vertex attribute half_width = its local half-width."),
    ],
    impl="library",
)
def compute_swath_distance_map(grid, track, mask=None, half_width=None, compute_signed=True,
                               return_centre_line=False):
    """Distance from every cell to a track, the base of a swath profile.

    topotoolbox.compute_swath_distance_map (Dijkstra front grown from the
    track); the centreline is where the fronts from both swath edges meet.
    """
    if return_centre_line and half_width is None:
        raise ValueError("the swath centreline needs a half_width")
    grid = _row_major(grid)
    cells = None if mask is None else (np.nan_to_num(_values(mask, float)) != 0).astype(np.int8)
    rows, cols = _track_indices(grid, track)
    res = ttb.compute_swath_distance_map(
        grid, rows, cols, half_width=half_width, input_mode="indices2D",
        compute_signed=compute_signed, return_nearest_point=True,
        return_centre_line=return_centre_line, mask=cells,
    )
    out = {"distance_map": res.distance_map, "nearest_point": res.nearest_point}
    if return_centre_line:
        out["dist_from_boundary"] = res.dist_from_boundary
        out["centre_line"] = _geovector(grid, "lines", res.centre_line_x, res.centre_line_y,
                                        half_width=res.centre_width)
    return out


@process(
    id="topotoolbox.transverse_swath",
    label="Cross-track swath profile",
    inputs=[Port("grid", "topotoolbox.GridObject", doc=_GRID_DOC),
            Port("distance_map", "topotoolbox.GridObject", doc=_DISTANCE_MAP_DOC)],
    params=[
        Param("half_width", "float", min=0.0, doc=_HALF_WIDTH_DOC),
        Param("bin_resolution", "float", default=10.0, min=0.0,
              doc="Width of each distance bin (map units)."),
        Param("normalize", "bool", default=False,
              doc="Values relative to the mean of the cells within one bin of the track."),
        Param("percentiles", "topotoolbox.percentiles", default=""),
    ],
    outputs=[Output("profile", "datatable",
                    doc="Statistics per distance bin; negative distances are right of the track.")],
    impl="library",
)
def transverse_swath(grid, distance_map, half_width, bin_resolution=10.0, normalize=False,
                     percentiles=""):
    """Statistics of a grid across a track, by distance from it.

    topotoolbox.transverse_swath. ``normalize`` is applied here: topotoolbox's
    own makes the means relative but leaves min, max, median and percentiles
    absolute. Every statistic is made relative to the same reference,
    topotoolbox's: the mean of the swath cells within one bin of the track.
    """
    grid = _row_major(grid)
    dist = _values(distance_map, np.float32)
    plist = _percentile_list(percentiles)
    stats = ttb.transverse_swath(grid, dist, half_width, bin_resolution=bin_resolution,
                                 percentiles=plist)
    title = "Cross-track swath"
    if normalize:
        z, d = grid.z.ravel(), dist.ravel()
        near = (~np.isnan(z) & ~np.isnan(d) & (np.abs(d) <= half_width)
                & (np.abs(d) <= bin_resolution))
        ref = float(np.mean(z[near])) if near.any() else 0.0
        for name in ("means", "mins", "maxs", "medians", "q1", "q3"):
            if getattr(stats, name) is not None:
                setattr(stats, name, getattr(stats, name) - ref)
        stats.percentiles = {p: v - ref for p, v in (stats.percentiles or {}).items()}
        title += " (relative to the track)"
    return _swath_table(stats, "distance", stats.distances, plist, title)


@process(
    id="topotoolbox.longitudinal_swath",
    label="Along-track swath profile",
    inputs=[
        Port("grid", "topotoolbox.GridObject", doc=_GRID_DOC),
        Port("track", "geolines", doc="The track the distance map was computed from."),
        Port("distance_map", "topotoolbox.GridObject", doc=_DISTANCE_MAP_DOC),
        Port("nearest_point", "topotoolbox.GridObject", doc=_NEAREST_DOC),
    ],
    params=[
        Param("half_width", "float", min=0.0, doc=_HALF_WIDTH_DOC),
        Param("binning_distance", "float", min=0.0,
              doc=_BINNING_DOC + " 0 = only the cells nearest to it."),
        Param("percentiles", "topotoolbox.percentiles", default=""),
        Param("skip", "int", default=1, min=1, doc=_SKIP_DOC),
    ],
    outputs=[Output("profile", "datatable",
                    doc="Statistics per track vertex by distance along the track; x, y = the vertex.")],
    impl="library",
)
def longitudinal_swath(grid, track, distance_map, nearest_point, half_width, binning_distance,
                       percentiles="", skip=1):
    """Statistics of a grid along a track, vertex by vertex.

    topotoolbox.longitudinal_swath: each vertex gathers the swath cells whose
    nearest track vertex lies within ``binning_distance`` of it along the track.
    """
    grid = _row_major(grid)
    plist = _percentile_list(percentiles)
    stats = ttb.longitudinal_swath(
        grid, *_track_indices(grid, track), _values(distance_map, np.float32), half_width,
        binning_distance, _values(nearest_point, np.intp), percentiles=plist or None,
        input_mode="indices2D", skip=skip,
    )
    xy = _cell_centres(grid, stats.track_x, stats.track_y)
    return _swath_table(stats, "distance", stats.along_track_distances, plist,
                        "Along-track swath", map_xy=xy)


@process(
    id="topotoolbox.longitudinal_swath_windowed",
    label="Along-track swath profile (windowed)",
    inputs=[Port("grid", "topotoolbox.GridObject", doc=_GRID_DOC),
            Port("track", "geolines", doc=_TRACK_DOC)],
    params=[
        Param("half_width", "float", min=0.0, doc=_HALF_WIDTH_DOC),
        Param("binning_distance", "float", min=0.0, doc=_BINNING_DOC),
        Param("n_points_regression", "int", default=5, min=1, doc=_REGRESSION_DOC),
        Param("percentiles", "topotoolbox.percentiles", default=""),
        Param("skip", "int", default=1, min=1, doc=_SKIP_DOC),
    ],
    outputs=[Output("profile", "datatable",
                    doc="Statistics per track vertex by distance along the track; x, y = the vertex.")],
    impl="library",
)
def longitudinal_swath_windowed(grid, track, half_width, binning_distance, n_points_regression=5,
                                percentiles="", skip=1):
    """Statistics of a grid along a track, in a window turned along it.

    topotoolbox.longitudinal_swath_windowed: each vertex gathers the cells in a
    rectangle (2 x binning_distance along, 2 x half_width across) oriented on
    the local track direction; no distance map needed.
    """
    grid = _row_major(grid)
    plist = _percentile_list(percentiles)
    stats = ttb.longitudinal_swath_windowed(
        grid, *_track_indices(grid, track), half_width, binning_distance,
        n_points_regression=n_points_regression, percentiles=plist or None,
        input_mode="indices2D", skip=skip,
    )
    xy = _cell_centres(grid, stats.track_x, stats.track_y)
    return _swath_table(stats, "distance", stats.along_track_distances, plist,
                        "Along-track swath (windowed)", map_xy=xy)


@process(
    id="topotoolbox.get_point_pixels",
    label="Cells of an along-track point",
    inputs=[
        Port("grid", "topotoolbox.GridObject", doc="Grid the swath is laid on."),
        Port("track", "geolines", doc="The track the distance map was computed from."),
        Port("distance_map", "topotoolbox.GridObject", doc=_DISTANCE_MAP_DOC),
        Port("nearest_point", "topotoolbox.GridObject", doc=_NEAREST_DOC),
    ],
    params=[
        Param("point_index", "int", min=0, doc="Track vertex (0 = first; counted before skip)."),
        Param("half_width", "float", min=0.0, doc=_HALF_WIDTH_DOC),
        Param("binning_distance", "float", min=0.0,
              doc=_BINNING_DOC + " 0 = only the cells nearest to it."),
    ],
    outputs=[Output("pixels", "geopoints", doc="Centres of the cells the point gathers.")],
    impl="library",
)
def get_point_pixels(grid, track, distance_map, nearest_point, point_index, half_width,
                     binning_distance):
    """The cells one point of the along-track profile gathers.

    topotoolbox.get_point_pixels (same selection as the along-track swath).
    """
    grid = _row_major(grid)
    rows, cols = ttb.get_point_pixels(
        grid, *_track_indices(grid, track), _values(distance_map, np.float32), point_index,
        half_width, binning_distance, _values(nearest_point, np.intp), input_mode="indices2D",
    )
    return _geovector(grid, "points", rows, cols)


@process(
    id="topotoolbox.get_windowed_point_samples",
    label="Cells of an along-track point (windowed)",
    inputs=[Port("grid", "topotoolbox.GridObject", doc="Grid the swath is laid on."),
            Port("track", "geolines", doc=_TRACK_DOC)],
    params=[
        Param("point_index", "int", min=0, doc="Track vertex (0 = first; counted before skip)."),
        Param("half_width", "float", min=0.0, doc=_HALF_WIDTH_DOC),
        Param("binning_distance", "float", min=0.0, doc=_BINNING_DOC),
        Param("n_points_regression", "int", default=5, min=1, doc=_REGRESSION_DOC),
    ],
    outputs=[Output("pixels", "geopoints", doc="Centres of the cells in the point's window.")],
    impl="library",
)
def get_windowed_point_samples(grid, track, point_index, half_width, binning_distance,
                               n_points_regression=5):
    """The cells in the window of one point of the windowed along-track profile.

    topotoolbox.get_windowed_point_samples (same selection as the windowed
    along-track swath).
    """
    grid = _row_major(grid)
    rows, cols = ttb.get_windowed_point_samples(
        grid, *_track_indices(grid, track), point_index, half_width, binning_distance,
        n_points_regression=n_points_regression, input_mode="indices2D",
    )
    return _geovector(grid, "points", rows, cols)


# ---- stream networks --------------------------------------------------------

register_type("topotoolbox.FlowObject", "graph", ttb.FlowObject)
register_type("topotoolbox.StreamObject", "graph", ttb.StreamObject)

_POINTS_DOC = "Points, in any CRS (reprojected to the grid's); each falls in one cell."


def _point_cells(obj, points, what):
    """(rows, cols) of the cells the GeoVector ``points`` fall in, on the grid
    of ``obj`` (GridObject, FlowObject or StreamObject: transform, shape,
    georef); ValueError for no point or a point off the grid."""
    if points.n_vertices == 0:
        raise ValueError(f"no {what} given")
    xy = points.in_crs(obj.georef.to_wkt() if obj.georef is not None else None).xy
    cols, rows = ~obj.transform * (xy[:, 0], xy[:, 1])
    rows, cols = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    off = (rows < 0) | (rows >= obj.shape[0]) | (cols < 0) | (cols >= obj.shape[1])
    if off.any():
        raise ValueError(f"{int(off.sum())} of the {what} fall outside the grid")
    return rows, cols


def _stream_point_nodes(stream, points, what):
    """Logical node attribute list of ``stream``: True at the nodes the points
    fall on; ValueError when none falls on the network."""
    rows, cols = _point_cells(stream, points, what)
    marked = np.zeros(stream.shape, dtype=bool)
    marked[rows, cols] = True
    nodes = stream.ezgetnal(marked)
    if not nodes.any():
        raise ValueError(f"none of the {what} falls on a stream cell")
    return nodes


@process(
    id="topotoolbox.flow_object",
    label="Flow directions (TopoToolbox)",
    inputs=[
        Port("dem", "topotoolbox.GridObject", arg="grid",
             doc="Elevation to route water over (sinks are resolved here)."),
        Port("bc", "topotoolbox.GridObject", optional=True,
             doc="Cells kept at their DEM elevation when sinks are filled (nonzero = kept)."),
    ],
    params=[
        Param("method", "enum", default="d8", choices=["d8"], choice_labels=["D8"],
              doc="Flow routing: each cell drains to its steepest neighbour."),
        Param("sink_resolution", "enum", default="carve", choices=["carve", "lcat"],
              choice_labels=["Carve", "Least-cost (lcat)"],
              doc="How flow leaves filled sinks and flats."),
    ],
    outputs=[Output("flow", "topotoolbox.FlowObject", doc="Flow directions, reusable by later runs.")],
    impl="library",
)
def flow_object(grid, bc=None, method="d8", sink_resolution="carve"):
    """Where each cell sends its water.

    topotoolbox.FlowObject: fills sinks, then routes flow.
    """
    if bc is not None:
        bc = (np.nan_to_num(np.asarray(bc.z)) != 0).astype(np.uint8)
    return ttb.FlowObject(grid, bc=bc, method=method, sink_resolution=sink_resolution)


@process(
    id="topotoolbox.stream_object",
    label="Stream network (TopoToolbox)",
    inputs=[
        Port("flow", "topotoolbox.FlowObject", doc="Flow directions to trace streams on."),
        Port("stream_pixels", "topotoolbox.GridObject", optional=True,
             doc="Cells that are streams (nonzero); replaces the threshold."),
        Port("channelheads", "geopoints", optional=True,
             doc="Channel heads: keeps the streams downstream of them; " + _POINTS_DOC),
    ],
    params=[
        Param("units", "enum", default="pixels", choices=["pixels", "mapunits", "m2", "km2"],
              choice_labels=["Cells", "Map units²", "m²", "km²"],
              doc="Units of the threshold."),
        Param("threshold", "float", default=0.0, min=0.0,
              doc="Upstream area needed to be a stream; 0 = automatic (1 % of the mean "
                  "grid side, squared, in cells)."),
    ],
    outputs=[Output("stream", "topotoolbox.StreamObject", doc="Stream network, reusable by later runs.")],
    impl="library",
)
def stream_object(flow, stream_pixels=None, channelheads=None, units="pixels", threshold=0.0):
    """Streams traced from an upstream-area threshold, a stream mask or heads.

    topotoolbox.StreamObject.
    """
    if stream_pixels is not None:
        stream_pixels = np.nan_to_num(np.asarray(stream_pixels.z)) != 0
    if channelheads is not None:
        channelheads = _point_cells(flow, channelheads, "channel heads")
    threshold = int(threshold) if units == "pixels" and float(threshold).is_integer() else threshold
    return ttb.StreamObject(flow, units=units, threshold=threshold,
                            stream_pixels=stream_pixels, channelheads=channelheads)


@process(
    id="topotoolbox.klargestconncomps",
    label="Largest stream networks",
    inputs=[Port("stream", "topotoolbox.StreamObject", doc="Stream network.")],
    params=[Param("k", "int", default=1, min=1, doc="Number of networks kept, largest first.")],
    outputs=[Output("largest", "topotoolbox.StreamObject", doc="The k largest networks.")],
    impl="library",
)
def klargestconncomps(stream, k=1):
    """Keep the k largest connected networks (by cell count).

    topotoolbox StreamObject.klargestconncomps.
    """
    return stream.klargestconncomps(k)


@process(
    id="topotoolbox.trunk",
    label="Trunk streams",
    inputs=[
        Port("stream", "topotoolbox.StreamObject", doc="Stream network."),
        Port("flow_accumulation", "topotoolbox.GridObject", optional=True,
             doc="Flow accumulation: the trunk follows the largest one instead of the longest path."),
    ],
    outputs=[Output("trunk", "topotoolbox.StreamObject", doc="The trunk stream of each network.")],
    impl="library",
)
def trunk(stream, flow_accumulation=None):
    """Keep only the main stream of each network.

    topotoolbox StreamObject.trunk: traced upstream along the longest path.
    """
    return stream.trunk(flow_accumulation=flow_accumulation)


@process(
    id="topotoolbox.upstreamto",
    label="Streams upstream of points",
    inputs=[Port("stream", "topotoolbox.StreamObject", doc="Stream network."),
            Port("points", "geopoints", doc="Points on stream cells (e.g. outlets); " + _POINTS_DOC)],
    outputs=[Output("upstream", "topotoolbox.StreamObject", doc="The streams upstream of the points.")],
    impl="library",
)
def upstreamto(stream, points):
    """Keep the streams upstream of the given points.

    topotoolbox StreamObject.upstreamto. Points must fall on stream cells.
    """
    return stream.upstreamto(_stream_point_nodes(stream, points, "points"))


@process(
    id="topotoolbox.downstreamto",
    label="Streams downstream of points",
    inputs=[Port("stream", "topotoolbox.StreamObject", doc="Stream network."),
            Port("points", "geopoints", doc="Points on stream cells; " + _POINTS_DOC)],
    outputs=[Output("downstream", "topotoolbox.StreamObject", doc="The streams downstream of the points.")],
    impl="library",
)
def downstreamto(stream, points):
    """Keep the streams downstream of the given points.

    topotoolbox StreamObject.downstreamto. Points must fall on stream cells.
    """
    return stream.downstreamto(_stream_point_nodes(stream, points, "points"))


@process(
    id="topotoolbox.stream_lines",
    label="Stream lines",
    inputs=[
        Port("stream", "topotoolbox.StreamObject", doc="Stream network."),
        Port("dem", "topotoolbox.GridObject", optional=True,
             doc="Elevation; needed for gradient and ksn."),
        Port("flow", "topotoolbox.FlowObject", optional=True,
             doc="Flow directions of the network; needed for drainage area, ksn and chi."),
    ],
    params=[
        Param("stream_order", "enum", default="strahler", choices=["none", "strahler", "shreve"],
              choice_labels=["None", "Strahler", "Shreve"], doc="Stream order of every vertex."),
        Param("drainage_area", "bool", default=True, doc="Add the drainage area (m², needs the flow)."),
        Param("upstream_distance", "bool", default=True,
              doc="Add the distance from the farthest channel head (map units)."),
        Param("downstream_distance", "bool", default=False,
              doc="Add the distance to the outlet (map units)."),
        Param("gradient", "bool", default=False, doc="Add the channel gradient (needs the DEM)."),
        Param("ksn", "bool", default=False,
              doc="Add the normalised steepness index (needs the DEM and the flow)."),
        Param("chi", "bool", default=False, doc="Add chi (needs the flow)."),
        Param("impose", "bool", default=False,
              doc="Gradient and ksn: impose downstream minima first (no negative slopes)."),
        Param("theta", "float", default=0.45, min=0.0, doc="ksn: reference concavity."),
        Param("mn", "float", default=0.45, min=0.0, doc="chi: m/n ratio."),
        Param("a0", "float", default=1e6, min=0.0, doc="chi: reference area (map units²)."),
    ],
    outputs=[Output("lines", "geolines",
                    doc="The streams as lines, one vertex per stream cell, with the chosen values "
                        "per vertex.")],
    impl="library",
)
def stream_lines(stream, dem=None, flow=None, stream_order="strahler", drainage_area=True,
                 upstream_distance=True, downstream_distance=False, gradient=False, ksn=False,
                 chi=False, impose=False, theta=0.45, mn=0.45, a0=1e6):
    """The stream network as lines carrying per-node values.

    topotoolbox StreamObject.to_geodataframe, streamorder, upstream_distance,
    downstream_distance, gradient, ksn, chitransform.
    """
    if (gradient or ksn) and dem is None:
        raise ValueError("gradient and ksn need the DEM")
    if (drainage_area or ksn or chi) and flow is None:
        raise ValueError("drainage area, ksn and chi need the flow directions")
    values = {}
    if stream_order != "none":
        values["stream_order"] = stream.streamorder(method=stream_order)
    acc = flow.flow_accumulation() if flow is not None else None  # cells
    if drainage_area:
        values["drainage_area"] = stream.ezgetnal(acc) * stream.cellsize ** 2
    if upstream_distance:
        values["upstream_distance"] = stream.upstream_distance()
    if downstream_distance:
        values["downstream_distance"] = stream.downstream_distance()
    if gradient:
        values["gradient"] = stream.gradient(dem, impose=impose)
    if ksn:
        values["ksn"] = stream.ksn(dem, acc, impose=impose, theta=theta)
    if chi:
        values["chi"] = stream.chitransform(acc, a0=a0, mn=mn)

    vec = CONVERTERS.convert(stream.to_geodataframe(), "geolines",
                             from_type="geopandas.GeoDataFrame")
    node = np.full(stream.shape, -1, dtype=np.int64)
    node[stream.node_indices] = np.arange(stream.stream.size)
    cols, rows = ~stream.transform * (vec.xy[:, 0], vec.xy[:, 1])
    vertex_node = node[np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)]
    if (vertex_node < 0).any():
        raise RuntimeError("a stream line vertex is on no stream node")
    attrs = {k: np.asarray(v, dtype=np.float64)[vertex_node] for k, v in values.items()}
    return dataclasses.replace(vec, vertex_attrs={**vec.vertex_attrs, **attrs})
