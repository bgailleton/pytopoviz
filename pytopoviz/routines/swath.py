"""Swath profile routines (composite processes).

Author: B.G.
"""

from __future__ import annotations

from shapely.geometry import LineString

from ..core import Output, Param, Port, process
from ..core.registries import CONVERTERS, PROCESSES

_OUTLINE_DOC = "The swath corridor: the line widened by the half-width, round ends."


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
    label="Swath profile",
    inputs=[
        Port("dem", "topotoolbox.GridObject", doc="Grid to profile (e.g. the DEM)."),
        Port("track", "geolines", doc="Line to profile along, in any CRS."),
        Port("prepared_track", "geolines", optional=True,
             doc="Track already prepared (one vertex per cell); skips the preparation, "
                 "track is then unused."),
        Port("distance_map", "topotoolbox.GridObject", optional=True,
             doc="Distance map of the prepared track; with nearest_point, skips computing them."),
        Port("nearest_point", "topotoolbox.GridObject", optional=True,
             doc="Nearest-vertex grid of the prepared track, given with distance_map."),
        Port("mask", "topotoolbox.GridObject", optional=True,
             doc="Cells to use: nonzero = used."),
    ],
    params=[
        Param("half_width", "float", min=0.0,
              doc="Swath half-width: cells farther from the track are left out (map units)."),
        Param("bin_resolution", "float", default=10.0, min=0.0,
              doc="Cross-track profile: width of each distance bin (map units)."),
        Param("normalize", "bool", default=False,
              doc="Cross-track profile relative to the mean of the cells within one bin of "
                  "the track."),
        Param("method", "enum", default="nearest_point", choices=["nearest_point", "windowed"],
              choice_labels=["Nearest track vertex", "Oriented window"],
              doc="Along-track profile: each point gathers the cells whose nearest track vertex "
                  "is close to it, or the cells in a rectangle turned along the track."),
        Param("binning_distance", "float", min=0.0,
              doc="Along-track profile: each point gathers the cells within this distance of it "
                  "along the track (map units)."),
        Param("n_points_regression", "int", default=5, min=1,
              doc="Oriented window: track vertices used to orient each window."),
        Param("percentiles", "topotoolbox.percentiles", default=""),
        Param("skip", "int", default=1, min=1,
              doc="Along-track profile: keep every n-th track vertex as a point."),
        Param("return_centre_line", "bool", default=False,
              doc="Also find the swath's centreline."),
    ],
    outputs=[
        Output("transverse", "datatable", doc="Cross-track profile."),
        Output("longitudinal", "datatable", doc="Along-track profile."),
        Output("prepared", "geolines",
               doc="The track used, one vertex per cell (reusable as prepared_track)."),
        Output("distances", "topotoolbox.GridObject",
               doc="Signed distance from each cell to the track (reusable as distance_map)."),
        Output("nearest", "topotoolbox.GridObject",
               doc="Index of each cell's nearest track vertex (reusable as nearest_point)."),
        Output("centre_line", "geolines", optional=True,
               doc="Centreline of the swath (when asked and the distance map was computed)."),
        Output("dist_from_boundary", "topotoolbox.GridObject", optional=True,
               doc="Distance from each swath cell to the swath edge (with the centreline)."),
        Output("outline", "geopolygons", optional=True, doc=_OUTLINE_DOC),
    ],
    impl="composite",
)
def swath_profile_topotoolbox(
    dem, track, half_width, binning_distance, prepared_track=None, distance_map=None,
    nearest_point=None, mask=None, bin_resolution=10.0, normalize=False, method="nearest_point",
    n_points_regression=5, percentiles="", skip=1, return_centre_line=False,
):
    """Profiles of a grid across and along a track.

    Chains topotoolbox.prepare_track -> topotoolbox.compute_swath_distance_map
    -> topotoolbox.transverse_swath and topotoolbox.longitudinal_swath (or
    longitudinal_swath_windowed). ``prepared_track`` skips the preparation;
    ``distance_map`` with ``nearest_point`` skips the distance map. ``outline``
    is the prepared track widened by ``half_width``.
    """
    if prepared_track is None:
        prepared_track = PROCESSES.get("topotoolbox.prepare_track")(
            grid=dem, track=track)["prepared_track"]
    if distance_map is None or nearest_point is None:
        maps = PROCESSES.get("topotoolbox.compute_swath_distance_map")(
            grid=dem, track=prepared_track, mask=mask, half_width=half_width,
            return_centre_line=return_centre_line)
        distance_map, nearest_point = maps["distance_map"], maps["nearest_point"]
    else:
        maps = {}
    out = {"prepared": prepared_track, "distances": distance_map, "nearest": nearest_point,
           "centre_line": maps.get("centre_line"),
           "dist_from_boundary": maps.get("dist_from_boundary"),
           "outline": _outline(prepared_track, half_width)}

    out["transverse"] = PROCESSES.get("topotoolbox.transverse_swath")(
        grid=dem, distance_map=distance_map, half_width=half_width,
        bin_resolution=bin_resolution, normalize=normalize, percentiles=percentiles)["profile"]
    if method == "windowed":
        along = PROCESSES.get("topotoolbox.longitudinal_swath_windowed")(
            grid=dem, track=prepared_track, half_width=half_width,
            binning_distance=binning_distance, n_points_regression=n_points_regression,
            percentiles=percentiles, skip=skip)
    else:
        along = PROCESSES.get("topotoolbox.longitudinal_swath")(
            grid=dem, track=prepared_track, distance_map=distance_map,
            nearest_point=nearest_point, half_width=half_width,
            binning_distance=binning_distance, percentiles=percentiles, skip=skip)
    out["longitudinal"] = along["profile"]
    return out


@process(
    id="pytopoviz.swath_profile_lsdtt3",
    label="Swath profile",
    inputs=[
        Port("dem", "lsdtt3.Raster",
             doc="Grid defining the cells and the valid area; profiled unless values are given."),
        Port("baseline", "geolines", doc="Line to profile along, in any CRS."),
        Port("values", "lsdtt3.Raster", optional=True,
             doc="Other grid to profile on the same cells (e.g. slope)."),
    ],
    params=[
        Param("half_width_metres", "float", default=0.0, min=0.0,
              doc="Only cells within this distance of the line are used; 0 = every cell."),
        Param("bin_width_metres", "float", default=1000.0, min=0.0,
              doc="Length of each bin along the line (must be above 0)."),
    ],
    outputs=[
        Output("profile", "datatable", doc="Mean and percentiles per bin along the line; x, y = the bin centre on the line."),
        Output("along_axis", "lsdtt3.Raster", doc="Distance along the line of each swath cell (m)."),
        Output("perpendicular_distance", "lsdtt3.Raster",
               doc="Distance from each swath cell to the line (m)."),
        Output("signed_perpendicular_distance", "lsdtt3.Raster",
               doc="Same, positive left of the line, negative right (m)."),
        Output("outline", "geopolygons", optional=True,
               doc=_OUTLINE_DOC + " Not produced for a half-width of 0."),
    ],
    impl="composite",
)
def swath_profile_lsdtt3(dem, baseline, values=None, half_width_metres=0.0,
                         bin_width_metres=1000.0):
    """Profile of a grid along a line, bin by bin.

    Runs lsdtt3.swath_profile (lsdtt3 has no cross-track profile). The outline
    is drawn in the DEM's CRS.
    """
    out = PROCESSES.get("lsdtt3.swath_profile")(
        reference=dem, baseline=baseline, values=values,
        half_width_metres=half_width_metres, bin_width_metres=bin_width_metres)
    out["outline"] = _outline(baseline.in_crs(dem.metadata.crs_wkt), half_width_metres)
    return out
