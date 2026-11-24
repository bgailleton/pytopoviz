"""Ways of getting a line (composite processes): read from a file, or the
flow path down from a point, traced by one library or the other (each with
its own flow directions, computed from the DEM when none is given).

Author: B.G.
"""

from __future__ import annotations

import numpy as np

from ..core import Output, Param, Port, process
from ..core.registries import CONVERTERS, PROCESSES
from ..geovector import GeoVector

_POINT_DOC = "Where the path starts (its first point), in any CRS (reprojected to the DEM's)."


@process(
    id="pytopoviz.read_lines",
    category="Input",
    label="Lines from a file",
    params=[Param("path", "path", doc="Vector file to read (GeoJSON, Shapefile, GeoPackage, "
                                      "KML... anything geopandas reads).")],
    outputs=[Output("lines", "geolines", doc="The file's lines, attributes kept.")],
    impl="composite",
)
def read_lines(path):
    """The lines of a vector file.

    geopandas.read_file; the features that are not lines are left out (an
    error when none is).
    """
    import geopandas as gpd

    frame = gpd.read_file(path)
    lines = frame[frame.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if lines.empty:
        kinds = ", ".join(sorted(set(frame.geometry.geom_type.dropna()))) or "nothing"
        raise ValueError(f"{path} holds no lines ({kinds})")
    return CONVERTERS.convert(lines.reset_index(drop=True), "geolines",
                              from_type="geopandas.GeoDataFrame")


def _first_point(point, crs_wkt):
    return point.in_crs(crs_wkt).xy[0]


def _path_vector(x, y, epsg, crs_wkt):
    if len(x) < 2:
        raise ValueError("the point drains nowhere (an outlet or a no-data cell)")
    return GeoVector("lines", np.column_stack([x, y]), epsg=epsg, crs_wkt=crs_wkt)


@process(
    id="pytopoviz.flow_path_topotoolbox",
    category="Flow",
    label="Flow path from a point",
    inputs=[
        Port("dem", "topotoolbox.GridObject", label="DEM",
             doc="Elevation the flow directions are made from when none are given."),
        Port("point", "geopoints", label="Point", doc=_POINT_DOC),
        Port("flow", "topotoolbox.FlowObject", label="Flow object", optional=True,
             doc="Flow directions of the DEM; made from it (default settings) if empty."),
    ],
    outputs=[
        Output("path", "geolines", doc="The path water takes from the point to its outlet."),
        Output("flow_directions", "topotoolbox.FlowObject", optional=True,
               doc="The flow directions made (not produced when given)."),
    ],
    impl="composite",
)
def flow_path_topotoolbox(dem, point, flow=None):
    """The path water takes from a point, down to its outlet.

    topotoolbox.flow_object (when no flow directions are given) ->
    topotoolbox.flowpathextract.
    """
    out = {}
    if flow is None:
        flow = PROCESSES.get("topotoolbox.flow_object")(grid=dem)["flow"]
        out["flow_directions"] = flow
    first = GeoVector("points", point.xy[:1], epsg=point.epsg, crs_wkt=point.crs_wkt)
    out["path"] = PROCESSES.get("topotoolbox.flowpathextract")(flow=flow, points=first)["paths"]
    return out


@process(
    id="pytopoviz.flow_path_lsdtt3",
    category="Flow",
    label="Flow path from a point",
    inputs=[
        Port("dem", "lsdtt3.Raster", label="DEM",
             doc="Elevation the flow directions are made from when none are given."),
        Port("point", "geopoints", label="Point", doc=_POINT_DOC),
        Port("flow", "lsdtt3.FlowInfo", label="Flow object", optional=True,
             doc="Flow directions of the DEM; made from it (default settings) if empty."),
    ],
    outputs=[
        Output("path", "geolines", doc="The path water takes from the point to its outlet."),
        Output("flow_directions", "lsdtt3.FlowInfo", optional=True,
               doc="The flow directions made (not produced when given)."),
    ],
    impl="composite",
)
def flow_path_lsdtt3(dem, point, flow=None):
    """The path water takes from a point, down to its outlet.

    lsdtt3.fill_pits_boundary_aware -> lsdtt3.flow_info (when no flow
    directions are given), then lsdtt3's D8 receivers followed from the
    point's cell until one is its own receiver (an outlet).
    """
    out = {}
    if flow is None:
        filled = PROCESSES.get("lsdtt3.fill_pits_boundary_aware")(raster=dem)["filled"]
        flow = PROCESSES.get("lsdtt3.flow_info")(dem=filled)["flow"]
        out["flow_directions"] = flow
    meta = flow.metadata
    x, y = _first_point(point, meta.crs_wkt)
    row = int(np.floor((meta.y_max - y) / meta.cell_size))
    col = int(np.floor((x - meta.x_min) / meta.cell_size))
    if not (0 <= row < meta.n_rows and 0 <= col < meta.n_cols):
        raise ValueError("the point is off the grid")
    receiver = np.asarray(flow.receiver).reshape(-1)
    cells = [row * meta.n_cols + col]
    while len(cells) <= receiver.size:
        nxt = int(receiver[cells[-1]])
        if nxt < 0 or nxt == cells[-1]:
            break
        cells.append(nxt)
    rows, cols = np.divmod(np.asarray(cells, dtype=np.int64), meta.n_cols)
    out["path"] = _path_vector(meta.x_min + (cols + 0.5) * meta.cell_size,
                               meta.y_max - (rows + 0.5) * meta.cell_size,
                               meta.epsg_code or 0, meta.crs_wkt)
    return out
