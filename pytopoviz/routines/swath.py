"""Swath profile routines (composite processes).

The two swath profiles, one per library, take the same inputs and the same
shared parameters under the same names (Topos fills them from one form):
the DEM, a line, optionally another grid to profile on the DEM's cells, the
half-width and the along-track bin (a window of that length along the line;
lsdtt3 cuts the line into bins of it, TopoToolbox slides it, a point per
track cell). Empty sizes default to HALF_WIDTH_CELLS / ALONG_BIN_CELLS /
CROSS_BIN_CELLS cells of the DEM. Outputs are named alike too: along_track,
cross_track (TopoToolbox only), outline, signed_distance, then what is
proper to each library.

Author: B.G.
"""

from __future__ import annotations

from shapely.geometry import LineString

from ..core import Output, Param, Port, process
from ..core.registries import CONVERTERS, PROCESSES
from ..geovector import GeoVector

_OUTLINE_DOC = "The swath corridor: the line widened by the half-width, round ends."
#: Default sizes, in cells of the DEM.
HALF_WIDTH_CELLS = 100.0
ALONG_BIN_CELLS = 10.0
CROSS_BIN_CELLS = 1.0
_LINE_DOC = "Line to profile along, in any CRS (reprojected to the DEM's)."
_VALUES_DOC = "Other grid to profile on the DEM's cells (e.g. slope); the DEM itself if empty."
_HALF_WIDTH_DOC = "Cells farther from the line are left out (map units). Empty: %g cells."
_ALONG_BIN_DOC = ("Length of the window each along-track point gathers (map units). "
                  "Empty: %g cells.")


def _size(value, cells, cell_size):
    """`value`, or `cells` cells when empty."""
    return float(value) if value is not None else cells * float(cell_size)


def _nth_line(lines, index):
    """Line `index` (a feature of one part) of the lines GeoVector `lines`,
    alone."""
    if not 0 <= index < lines.n_features:
        raise ValueError(f"no line {index}: the object holds {lines.n_features}")
    p0, p1 = lines.feature_offsets[index], lines.feature_offsets[index + 1]
    if p1 - p0 != 1:
        raise ValueError(f"line {index} has {p1 - p0} parts; a swath needs a single line")
    v0, v1 = lines.part_offsets[p0], lines.part_offsets[p0 + 1]
    return GeoVector("lines", lines.xy[v0:v1], epsg=lines.epsg, crs_wkt=lines.crs_wkt)


_LINE_INDEX = Param("line_index", "int", label="Which line", default=0, min=0,
                    doc="The line of the object to profile along (0 = the first), when it "
                        "holds several.")


def _outline(line, half_width):
    """The corridor of `line` (a single-line GeoVector) widened by `half_width`
    on both sides with round ends and joins, as a geopolygons GeoVector in the
    line's CRS; None for a half-width of 0 (every cell)."""
    if not half_width or half_width <= 0.0:
        return None
    import geopandas as gpd

    corridor = LineString(line.line_xy()).buffer(half_width)
    frame = gpd.GeoDataFrame(geometry=[corridor], crs=line.crs())
    return CONVERTERS.convert(frame, "geopolygons", from_type="geopandas.GeoDataFrame")


@process(
    id="pytopoviz.swath_profile_topotoolbox",
    category="Swath",
    label="Swath profile",
    inputs=[
        Port("dem", "topotoolbox.GridObject", label="DEM",
             doc="Grid the swath is laid on; profiled unless another grid is given."),
        Port("line", "geolines", label="Line", doc=_LINE_DOC),
        Port("values", "topotoolbox.GridObject", label="Grid to profile", optional=True,
             doc=_VALUES_DOC),
    ],
    params=[
        _LINE_INDEX,
        Param("half_width", "float", label="Half-width", optional=True, min=0.0,
              doc=(_HALF_WIDTH_DOC % HALF_WIDTH_CELLS) + " Must be above 0."),
        Param("along_bin", "float", label="Along-track bin", optional=True, min=0.0,
              doc=(_ALONG_BIN_DOC % ALONG_BIN_CELLS) + " A point per track cell, each "
                  "gathering the cells within half of it on either side (sliding window)."),
        Param("cross_bin", "float", label="Cross-track bin", optional=True, min=0.0,
              doc="Width of each distance bin of the cross-track profile (map units). "
                  "Empty: %g cell." % CROSS_BIN_CELLS),
        Param("method", "enum",
              label="Along-track method", default="nearest_point",
              choices=["nearest_point", "windowed"],
              choice_labels=["Nearest track vertex", "Oriented window"],
              doc="Each along-track point gathers the cells whose nearest track vertex is in "
                  "its window, or the cells in a rectangle turned along the track."),
        Param("n_points_regression", "int", label="Window: vertices to orient", default=5, min=1,
              doc="Oriented window: track vertices used to orient each window."),
        Param("percentiles", "topotoolbox.percentiles", label="Extra percentiles", default=""),
        Param("normalize", "bool", label="Cross-track relative to the line", default=False,
              doc="Cross-track profile relative to the mean of the cells within one bin of "
                  "the line."),
        Param("skip", "int", label="Keep every n-th point", default=1, min=1,
              doc="Along-track profile: keep every n-th track vertex as a point."),
    ],
    outputs=[
        Output("along_track", "datatable", doc="Along-track profile."),
        Output("cross_track", "datatable", doc="Cross-track profile."),
        Output("outline", "geopolygons", doc=_OUTLINE_DOC),
        Output("signed_distance", "topotoolbox.GridObject",
               doc="Distance of each swath cell to the line, + left, - right (map units)."),
        Output("track", "geolines", doc="The line laid on the grid (one vertex per cell)."),
        Output("swath_map", "topotoolbox.SwathMap",
               doc="The swath, for the single steps (Profiles › Steps)."),
    ],
    impl="composite",
)
def swath_profile_topotoolbox(
    dem, line, values=None, line_index=0, half_width=None, along_bin=None, cross_bin=None,
    method="nearest_point", n_points_regression=5, percentiles="", normalize=False, skip=1,
):
    """Profiles of a grid along and across a line.

    Lays the swath on the DEM (topotoolbox.swath_map: the line prepared, the
    distance map), then runs topotoolbox.transverse_swath and
    topotoolbox.longitudinal_swath (or longitudinal_swath_windowed) on the
    grid to profile, within the half-width.
    """
    line = _nth_line(line, line_index)
    hw = _size(half_width, HALF_WIDTH_CELLS, dem.cellsize)
    if hw <= 0.0:
        raise ValueError("the TopoToolbox swath needs a half-width above 0")
    window = _size(along_bin, ALONG_BIN_CELLS, dem.cellsize)
    grid = values if values is not None else dem
    laid = PROCESSES.get("topotoolbox.swath_map")(grid=dem, track=line, half_width=hw)
    sm = laid["swath_map"]
    out = {"cross_track": PROCESSES.get("topotoolbox.transverse_swath")(
        grid=grid, distance_map=sm.distance_map, half_width=hw,
        bin_resolution=_size(cross_bin, CROSS_BIN_CELLS, dem.cellsize), normalize=normalize,
        percentiles=percentiles)["profile"]}
    if method == "windowed":
        along = PROCESSES.get("topotoolbox.longitudinal_swath_windowed")(
            grid=grid, track=sm.track, half_width=hw, binning_distance=window / 2.0,
            n_points_regression=n_points_regression, percentiles=percentiles, skip=skip)
    else:
        along = PROCESSES.get("topotoolbox.longitudinal_swath")(
            grid=grid, track=sm.track, distance_map=sm.distance_map,
            nearest_point=sm.nearest_point, half_width=hw, binning_distance=window / 2.0,
            percentiles=percentiles, skip=skip)
    out.update(along_track=along["profile"], outline=laid["outline"],
               signed_distance=laid["distance_map"], track=laid["prepared"], swath_map=sm)
    return out


@process(
    id="pytopoviz.swath_profile_lsdtt3",
    category="Swath",
    label="Swath profile",
    inputs=[
        Port("dem", "lsdtt3.Raster", label="DEM",
             doc="Grid defining the cells and the valid area; profiled unless another grid "
                 "is given."),
        Port("line", "geolines", label="Line", doc=_LINE_DOC),
        Port("values", "lsdtt3.Raster", label="Grid to profile", optional=True, doc=_VALUES_DOC),
    ],
    params=[
        _LINE_INDEX,
        Param("half_width", "float", label="Half-width", optional=True, min=0.0,
              doc=(_HALF_WIDTH_DOC % HALF_WIDTH_CELLS) + " 0: every cell."),
        Param("along_bin", "float", label="Along-track bin", optional=True, min=0.0,
              doc=(_ALONG_BIN_DOC % ALONG_BIN_CELLS) + " The line is cut into bins of it, "
                  "a point per bin."),
    ],
    outputs=[
        Output("along_track", "datatable",
               doc="Mean and percentiles per bin along the line; x, y = the bin centre on the line."),
        Output("outline", "geopolygons", optional=True,
               doc=_OUTLINE_DOC + " Not produced for a half-width of 0."),
        Output("signed_distance", "lsdtt3.Raster",
               doc="Distance of each swath cell to the line, + left, - right (m)."),
        Output("distance", "lsdtt3.Raster", doc="Distance of each swath cell to the line (m)."),
        Output("along_distance", "lsdtt3.Raster",
               doc="Distance along the line of each swath cell (m)."),
    ],
    impl="composite",
)
def swath_profile_lsdtt3(dem, line, values=None, line_index=0, half_width=None, along_bin=None):
    """Profile of a grid along a line, bin by bin.

    Runs lsdtt3.swath_profile (lsdtt3 has no cross-track profile). The outline
    is drawn in the DEM's CRS.
    """
    line = _nth_line(line, line_index)
    cell = dem.metadata.cell_size
    hw = _size(half_width, HALF_WIDTH_CELLS, cell)
    res = PROCESSES.get("lsdtt3.swath_profile")(
        reference=dem, baseline=line, values=values, half_width_metres=hw,
        bin_width_metres=_size(along_bin, ALONG_BIN_CELLS, cell))
    out = {"along_track": res["profile"], "signed_distance": res["signed_perpendicular_distance"],
           "distance": res["perpendicular_distance"], "along_distance": res["along_axis"]}
    outline = _outline(line.in_crs(dem.metadata.crs_wkt), hw)
    if outline is not None:
        out["outline"] = outline
    return out
