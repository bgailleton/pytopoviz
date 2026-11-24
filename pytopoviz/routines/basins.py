"""The basin upstream of a point, from a precomputed flow (lsdtt3 FlowInfo or
TopoToolbox FlowObject).

Each point is first snapped to an outlet, chosen among candidates within a
search radius (neither library snaps):
- ``none``: the cell the point falls in;
- ``max_area``: the cell of largest drainage area;
- ``nearest_channel``: the nearest channel cell (drainage area >= threshold);
- every channel junction (a channel cell with 2 or more channel donors, a
  triple junction included): the junction cell itself, just below the
  confluence (all its branches), and the first cell of each branch, just
  above it (that branch only).
A snap method picks one of them (``auto_pick``); a pick by index overrides it.
``candidates_for`` lists them with, for the cells inside the circle, which
candidate each drains to first (a quick view of what goes in), for a front
end to show and pick from.

Then each library delineates the basins from the outlets: lsdtt3
FlowInfo.basins, topotoolbox FlowObject.drainagebasins.

Author: B.G.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..core import Output, Param, Port, process
from ..core.registries import CONVERTERS
from ..datatable import DataTable
from ..geovector import GeoVector

SNAPS = ["max_area", "nearest_channel", "junction", "none"]
SNAP_LABELS = ["Max drainage area", "Nearest channel cell", "Channel junction", "No snap"]
# Candidate kinds each snap method takes, the first one there is.
_AUTO = {
    "none": ["none"],
    "max_area": ["max_area"],
    "nearest_channel": ["nearest_channel", "max_area"],
    "junction": ["junction_below", "nearest_channel", "max_area"],
}
# Defaults in cells when the radius / threshold are left at 0.
RADIUS_CELLS = 10
THRESHOLD_CELLS = 100
# Junctions listed per point, the nearest first.
MAX_JUNCTIONS = 12


class _Net:
    """A flow network seen the same way from either library: the receiver of
    each cell (row-major flat index; itself at an outlet), its drainage area
    (map units²) and the grid geometry (north-up)."""

    def __init__(self, receiver, area, shape, x_min, y_max, cell, epsg, crs_wkt):
        self.receiver = receiver
        self.area = area
        self.shape = (int(shape[0]), int(shape[1]))
        self.x_min = float(x_min)
        self.y_max = float(y_max)
        self.cell = float(cell)
        self.epsg = int(epsg or 0)
        self.crs_wkt = crs_wkt or ""

    def crs(self):
        return self.crs_wkt or self.epsg

    def cells_of(self, points, what="points"):
        """Flat indices of the cells the GeoVector ``points`` fall in."""
        if points.n_vertices == 0:
            raise ValueError(f"no {what} given")
        xy = points.in_crs(self.crs()).xy
        rows = np.floor((self.y_max - xy[:, 1]) / self.cell).astype(np.int64)
        cols = np.floor((xy[:, 0] - self.x_min) / self.cell).astype(np.int64)
        off = (rows < 0) | (rows >= self.shape[0]) | (cols < 0) | (cols >= self.shape[1])
        if off.any():
            raise ValueError(f"{int(off.sum())} of the {what} fall outside the grid")
        return rows * self.shape[1] + cols

    def xy(self, cells):
        rows, cols = np.divmod(np.asarray(cells, dtype=np.int64), self.shape[1])
        return self.x_min + (cols + 0.5) * self.cell, self.y_max - (rows + 0.5) * self.cell


def _net_lsdtt3(flow):
    meta = flow.metadata
    rec = np.asarray(flow.receiver, dtype=np.int64).reshape(-1).copy()
    own = np.arange(rec.size)
    bad = (rec < 0) | (rec >= rec.size)
    rec[bad] = own[bad]
    area = np.nan_to_num(np.asarray(flow.drainage_area().to_numpy(), dtype=float)).reshape(-1)
    return _Net(rec, area, (meta.n_rows, meta.n_cols), meta.x_min, meta.y_max, meta.cell_size,
                meta.epsg_code, meta.crs_wkt)


def _ttb_order(flow):
    return getattr(flow, "order", None) or "F"


def _net_topotoolbox(flow):
    from ..adapters.topotoolbox import _crs_fields

    shape = tuple(flow.shape)
    n = shape[0] * shape[1]

    def c_index(lin):
        return np.ravel_multi_index(np.unravel_index(np.asarray(lin, dtype=np.int64), shape,
                                                     order=_ttb_order(flow)), shape)

    rec = np.arange(n, dtype=np.int64)
    rec[c_index(flow.source)] = c_index(flow.target)
    cell = float(flow.cellsize)
    area = np.nan_to_num(np.asarray(flow.flow_accumulation().z, dtype=float)).reshape(-1) * cell ** 2
    t = flow.transform
    epsg, crs_wkt = _crs_fields(flow.georef)
    return _Net(rec, area, shape, t.c, t.f, cell, epsg, crs_wkt)


def _net(flow, type_id):
    if type_id == "lsdtt3.FlowInfo":
        return _net_lsdtt3(flow)
    if type_id == "topotoolbox.FlowObject":
        return _net_topotoolbox(flow)
    raise ValueError(f"no basins from a {type_id}")


def _defaults(net, radius, threshold):
    radius = float(radius) if radius and radius > 0 else RADIUS_CELLS * net.cell
    threshold = float(threshold) if threshold and threshold > 0 else THRESHOLD_CELLS * net.cell ** 2
    return radius, threshold


def _circle(net, cell, radius):
    """Flat indices of the cells within ``radius`` of ``cell``'s centre, their
    distances, and the (row0, col0, n_rows, n_cols) box holding them."""
    nr, nc = net.shape
    r0, c0 = divmod(int(cell), nc)
    k = int(np.ceil(radius / net.cell))
    rows = np.arange(max(r0 - k, 0), min(r0 + k, nr - 1) + 1)
    cols = np.arange(max(c0 - k, 0), min(c0 + k, nc - 1) + 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    dist = np.hypot(rr - r0, cc - c0) * net.cell
    inside = dist <= radius
    return (rr * nc + cc)[inside], dist[inside], inside, (rows[0], cols[0])


def _donors(net, cells, channel):
    """Channel donors of each of ``cells`` ({cell: [donor, ...]}): the channel
    cells among their 8 neighbours whose receiver they are."""
    nr, nc = net.shape
    out = {}
    for c in cells:
        r, q = divmod(int(c), nc)
        found = []
        for dr in (-1, 0, 1):
            for dq in (-1, 0, 1):
                rr, qq = r + dr, q + dq
                if (dr or dq) and 0 <= rr < nr and 0 <= qq < nc:
                    d = rr * nc + qq
                    if channel[d] and net.receiver[d] == c:
                        found.append(d)
        out[int(c)] = found
    return out


def _candidates(net, cell, radius, threshold):
    """The outlets a point at ``cell`` can snap to (module doc): [{kind,
    label, cell, junction?}], and the circle (cells, distances, mask, box
    origin)."""
    circle, dist, inside, origin = _circle(net, cell, radius)
    channel = net.area >= threshold
    cands = [{"kind": "none", "label": "Clicked cell", "cell": int(cell)}]
    cands.append({"kind": "max_area", "label": "Max drainage area",
                  "cell": int(circle[np.argmax(net.area[circle])])})
    on = channel[circle]
    if on.any():
        cands.append({"kind": "nearest_channel", "label": "Nearest channel cell",
                      "cell": int(circle[on][np.argmin(dist[on])])})
        order = np.argsort(dist[on], kind="stable")
        chan = circle[on][order]
        donors = _donors(net, chan, channel)
        n_j = 0
        for j in chan:
            ds = donors[int(j)]
            if len(ds) < 2:
                continue
            n_j += 1
            if n_j > MAX_JUNCTIONS:
                break
            ds = sorted(ds, key=lambda d: -net.area[d])
            cands.append({"kind": "junction_below", "junction": n_j, "cell": int(j),
                          "label": f"Junction {n_j} ({len(ds)} branches): below"})
            for i, d in enumerate(ds):
                cands.append({"kind": "junction_branch", "junction": n_j, "cell": int(d),
                              "label": f"Junction {n_j}: branch {i + 1} of {len(ds)}"})
    return cands, (circle, dist, inside, origin)


def auto_pick(snap, cands):
    """Index of the candidate snap method ``snap`` takes."""
    kinds = [c["kind"] for c in cands]
    for kind in _AUTO.get(snap, ["none"]):
        if kind in kinds:
            return kinds.index(kind)
    return 0


def _picks(text):
    """"2,-1,0" -> [2, -1, 0]; blanks and junk are -1 (the snap's pick)."""
    out = []
    for part in str(text or "").split(","):
        try:
            out.append(int(part.strip()))
        except ValueError:
            out.append(-1)
    return out


def _owners(net, cells, cands):
    """For each of ``cells``, the index of the first candidate its flow
    reaches (-1: none before an outlet)."""
    targets = {}
    for i, c in enumerate(cands):
        targets.setdefault(c["cell"], i)
    memo = {}
    out = np.empty(len(cells), dtype=np.int64)
    for k, c in enumerate(cells):
        path = []
        x = int(c)
        while True:
            if x in targets:
                res = targets[x]
                break
            if x in memo:
                res = memo[x]
                break
            path.append(x)
            nxt = int(net.receiver[x])
            if nxt == x:
                res = -1
                break
            x = nxt
        for p in path:
            memo[p] = res
        out[k] = res
    return out


def _region_rings(net, owners, inside, origin):
    """{candidate index: [flat ring x0, y0, ...]}: outlines of the parts of
    the circle draining to each candidate."""
    from rasterio.features import shapes
    from affine import Affine

    grid = np.full(inside.shape, -1, dtype=np.int32)
    grid[inside] = owners
    row0, col0 = origin
    transform = Affine(net.cell, 0.0, net.x_min + col0 * net.cell,
                       0.0, -net.cell, net.y_max - row0 * net.cell)
    out = {}
    for geom, value in shapes(grid, mask=grid >= 0, transform=transform, connectivity=8):
        for ring in geom["coordinates"]:
            out.setdefault(int(value), []).append(np.asarray(ring, dtype=float).reshape(-1).tolist())
    return out


def candidates_for(flow, flow_type, points, radius=0.0, threshold=0.0, snap="max_area"):
    """What a front end shows before picking: for each point, its candidates
    ({index, kind, label, x, y, area}), the one ``snap`` takes (auto), the
    circle (x, y, radius) and the outlines of the parts of it draining to each
    candidate (regions: {candidate, rings})."""
    net = _net(flow, flow_type)
    radius, threshold = _defaults(net, radius, threshold)
    out = []
    for cell in net.cells_of(points):
        cands, (circle, _dist, inside, origin) = _candidates(net, cell, radius, threshold)
        xs, ys = net.xy([c["cell"] for c in cands])
        listed = []
        for i, c in enumerate(cands):
            listed.append({"index": i, "kind": c["kind"], "label": c["label"],
                           "x": float(xs[i]), "y": float(ys[i]), "area": float(net.area[c["cell"]]),
                           "junction": int(c.get("junction", 0))})
        cx, cy = net.xy([cell])
        rings = _region_rings(net, _owners(net, circle, cands), inside, origin)
        out.append({"x": float(cx[0]), "y": float(cy[0]), "radius": radius, "threshold": threshold,
                    "candidates": listed, "auto": auto_pick(snap, cands),
                    "regions": [{"candidate": k, "rings": v} for k, v in sorted(rings.items())]})
    return {"points": out}


def _outlets(net, points, snap, radius, threshold, picks):
    """Unique outlet cells, the candidate each came from, and per outlet the
    first point snapped to it (point cell)."""
    radius, threshold = _defaults(net, radius, threshold)
    chosen = _picks(picks)
    outlets, info, starts = [], [], []
    for i, cell in enumerate(net.cells_of(points)):
        cands, _ = _candidates(net, cell, radius, threshold)
        p = chosen[i] if i < len(chosen) and 0 <= chosen[i] < len(cands) else auto_pick(snap, cands)
        if cands[p]["cell"] in outlets:
            continue
        outlets.append(cands[p]["cell"])
        info.append(cands[p])
        starts.append(int(cell))
    return np.asarray(outlets, dtype=np.int64), info, starts


def _parents(net, labels, outlets):
    """Basin id each basin drains into (0: none): the label just below its
    outlet (the innermost basin holding that cell)."""
    flat = labels.reshape(-1)
    out = []
    for o in outlets:
        below = int(net.receiver[o])
        v = flat[below] if below != o else 0
        out.append(int(v) if np.isfinite(v) else 0)
    return out


def _full_polygons(net, labels, parents):
    """GeoVector polygons of each whole basin (its nested basins included),
    from the label grid (ids 1..n, innermost; 0/NaN outside)."""
    import geopandas as gpd
    from affine import Affine
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    n = len(parents)
    children = {k: [] for k in range(1, n + 1)}
    for k, p in enumerate(parents, start=1):
        if p in children and p != k:
            children[p].append(k)

    def family(k):
        out, todo = [], [k]
        while todo:
            x = todo.pop()
            out.append(x)
            todo.extend(children[x])
        return out

    ids = np.nan_to_num(labels, nan=0.0).astype(np.int32)
    transform = Affine(net.cell, 0.0, net.x_min, 0.0, -net.cell, net.y_max)
    geoms = []
    for k in range(1, n + 1):
        mask = np.isin(ids, family(k))
        parts = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=transform,
                                             connectivity=8) if v == 1]
        geoms.append(unary_union(parts) if parts else None)
    frame = gpd.GeoDataFrame(geometry=geoms, crs=net.crs() or None)
    return CONVERTERS.convert(frame, "geopolygons", from_type="geopandas.GeoDataFrame")


def _finish(net, points, info, starts, outlets, labels, outlines, parents, table):
    """The outputs but the basins grid: outlines (with attributes), outlet
    points, the table when asked."""
    n = len(outlets)
    ids = np.arange(1, n + 1)
    area = net.area[outlets]
    ox, oy = net.xy(outlets)
    px, py = net.xy(starts)
    snap = [c["label"] for c in info]
    attrs = {"basin": ids, "area_m2": area, "parent": np.asarray(parents, dtype=np.int64)}
    out = {"outlines": dataclasses.replace(outlines, feature_attrs={**attrs, "snap": snap}),
           "outlets": GeoVector("points", np.column_stack([ox, oy]),
                                feature_attrs={**attrs, "snap": snap},
                                epsg=net.epsg, crs_wkt=net.crs_wkt)}
    if table:
        out["table"] = DataTable(
            columns={"basin": ids, "outlet_x": ox, "outlet_y": oy, "point_x": px, "point_y": py,
                     "snap_distance": np.hypot(ox - px, oy - py), "area": area,
                     "parent": np.asarray(parents, dtype=float)},
            units={"outlet_x": "m", "outlet_y": "m", "point_x": "m", "point_y": "m",
                   "snap_distance": "m", "area": "m²"},
            title="Basins")
    return out


_POINTS_DOC = "Where the basins end (one per point), in any CRS (reprojected to the flow's)."


def _params():
    return [
        Param("snap", "enum", default="max_area", choices=SNAPS, choice_labels=SNAP_LABELS,
              label="Snap to",
              doc="Outlet each point moves to within the radius: the cell of largest drainage area, "
                  "the nearest channel cell, the nearest channel junction (just below it), or the "
                  "point's own cell. A candidate picked by hand overrides it."),
        Param("radius", "float", default=0.0, min=0.0, label="Search radius", unit="m",
              doc=f"How far a point may move; 0 = {RADIUS_CELLS} cells."),
        Param("threshold", "float", default=0.0, min=0.0, label="Channel threshold", unit="m²",
              doc=f"Drainage area from which a cell is a channel (nearest channel, junctions); "
                  f"0 = {THRESHOLD_CELLS} cells."),
        Param("picks", "string", default="", label="Picked candidates",
              doc="Candidate index per point, comma-separated (-1 or blank: the snap's pick)."),
        Param("make_table", "bool", default=False, label="Basin table",
              doc="Also make a table: outlet, point, snap distance, area and parent of each basin."),
    ]


def _outputs(grid_type):
    return [
        Output("basins", grid_type, doc="Basin id of each cell (the innermost basin; NaN outside)."),
        Output("outlines", "geopolygons", doc="Outline of each whole basin, nested ones included."),
        Output("outlets", "geopoints", doc="Where each basin ends (the snapped points)."),
        Output("table", "datatable", optional=True, doc="One row per basin (Basin table)."),
    ]


@process(
    id="pytopoviz.basins_from_points_lsdtt3",
    category="Flow",
    label="Basin from a point",
    inputs=[Port("flow", "lsdtt3.FlowInfo", label="Flow object", doc="Flow network (FlowInfo)."),
            Port("points", "geopoints", label="Points", doc=_POINTS_DOC)],
    params=_params(),
    outputs=_outputs("lsdtt3.Raster"),
    impl="composite",
)
def basins_from_points_lsdtt3(flow, points, snap="max_area", radius=0.0, threshold=0.0, picks="",
                              make_table=False):
    """The basin upstream of each point, snapped first (module doc).

    Snap in pytopoviz, then lsdtt3 FlowInfo.basins (labels, polygons).
    """
    net = _net_lsdtt3(flow)
    outlets, info, starts = _outlets(net, points, snap, radius, threshold, picks)
    rows, cols = np.divmod(outlets, net.shape[1])
    coll = flow.basins(outlets=np.column_stack([rows, cols]))
    # lsdtt3 ids (basin_id) -> ours (1.. in outlet order), through the outlet cell.
    ours = {int(o): k for k, o in enumerate(outlets, start=1)}
    by_id = {int(b): ours[int(o)] for b, o in zip(coll.table["basin_id"], coll.table["outlet_flat"])}
    raw = np.asarray(coll.labels.to_numpy(), dtype=float)
    labels = np.full(raw.shape, np.nan)
    for b, k in by_id.items():
        labels[raw == b + 1] = k
    parents = _parents(net, labels, outlets)
    polys = coll.polygons.copy()
    polys["_ours"] = [by_id[int(b)] for b in polys["basin_id"]]
    polys = polys.sort_values("_ours")[["geometry"]].reset_index(drop=True)
    if polys.crs is None and net.crs():
        polys = polys.set_crs(net.crs())
    outlines = CONVERTERS.convert(polys, "geopolygons", from_type="geopandas.GeoDataFrame")
    out = _finish(net, points, info, starts, outlets, labels, outlines, parents, make_table)
    from ..georaster import GeoRaster
    meta = flow.metadata
    geo = GeoRaster(z=labels, cell_size=meta.cell_size, x_min=meta.x_min, y_min=meta.y_min,
                    epsg=meta.epsg_code, crs_wkt=meta.crs_wkt)
    out["basins"] = CONVERTERS.convert(geo, "lsdtt3.Raster", from_type="georaster")
    return out


@process(
    id="pytopoviz.basins_from_points_topotoolbox",
    category="Flow",
    label="Basin from a point",
    inputs=[Port("flow", "topotoolbox.FlowObject", label="Flow object",
                 doc="Flow network (FlowObject)."),
            Port("points", "geopoints", label="Points", doc=_POINTS_DOC)],
    params=_params(),
    outputs=_outputs("topotoolbox.GridObject"),
    impl="composite",
)
def basins_from_points_topotoolbox(flow, points, snap="max_area", radius=0.0, threshold=0.0, picks="",
                                   make_table=False):
    """The basin upstream of each point, snapped first (module doc).

    Snap in pytopoviz, then topotoolbox FlowObject.drainagebasins (labels);
    the outlines are polygonised here (rasterio), each whole basin.
    """
    net = _net_topotoolbox(flow)
    outlets, info, starts = _outlets(net, points, snap, radius, threshold, picks)
    shape = net.shape
    lin = np.ravel_multi_index(np.unravel_index(outlets, shape), shape, order=_ttb_order(flow))
    grid = flow.drainagebasins(lin)
    raw = np.asarray(grid.z, dtype=float)
    # topotoolbox labels -> ours (1.. in outlet order), read at each outlet.
    labels = np.full(raw.shape, np.nan)
    flat = raw.reshape(-1)
    for k, o in enumerate(outlets, start=1):
        labels[raw == flat[o]] = k
    parents = _parents(net, labels, outlets)
    outlines = _full_polygons(net, labels, parents)
    out = _finish(net, points, info, starts, outlets, labels, outlines, parents, make_table)
    out["basins"] = grid.duplicate_with_new_data(labels.astype(np.float32))
    return out
