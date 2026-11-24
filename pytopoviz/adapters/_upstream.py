"""Discharge entering a flood domain from the land upstream of it, as Q = P * A.

The domain is the footprint of the flood's bed. IGN's RGE ALTI (WMS-R of the
Géoplateforme, float32 in Lambert-93) is read around it at a chosen cell and
routed by steepest descent with pyfastflow (Cordonnier carving through pits).
The domain's cells are outlets and only cells outside it carry area, so the
area arriving in each domain cell is the land outside that drains to it.

The box read grows while a catchment draining to the domain reaches its edge
(the divide is then outside it), up to an area cap. Small entries are added to
the nearest large one, so the total area is kept. Each entry is placed on the
lowest bed cell near the routed crossing.

Moved from nevertheLAS (python/upstream_entries.py) for pytopoviz.upstream_entries.

Author: B.G (11/2026)
"""

import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

WMS = ("https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
       "&LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES&STYLES=&CRS=EPSG:2154"
       "&BBOX={left!r},{bottom!r},{right!r},{top!r}&WIDTH={size}&HEIGHT={size}&FORMAT=image/x-bil;bits=32")
USER_AGENT = "pytopoviz"
TILE = 512 # Cells per side of a request (the service takes up to 4096): small, so a small box reads little past its edge.
NODATA_BELOW = -1000.0 # No-data is -99999, blended with heights near its edge.
REQUESTS = 8
EDGE_RING = 2 # Cells of a draining catchment this close to the box edge mean the box is too small.
SNAP_CELLS = 1.5 # The entry is moved to the lowest bed cell within this many routing cells.


class Tiles:
    """Elevation tiles on a lattice of TILE cells, kept across growths of the box."""

    def __init__(self, cell, report):
        self.cell = float(cell)
        self.report = report
        self.kept = {}

    def _fetch(self, key):
        span = TILE * self.cell
        left, bottom = key[0] * span, key[1] * span
        url = WMS.format(left=left, bottom=bottom, right=left + span, top=bottom + span, size=TILE)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        last = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as reply:
                    data = reply.read()
                if len(data) == TILE * TILE * 4:
                    tile = np.frombuffer(data, "<f4").reshape(TILE, TILE).copy()
                    tile[tile < NODATA_BELOW] = np.nan
                    return tile
                last = data[:300].decode("utf-8", "replace")
            except OSError as error:
                last = str(error)
        raise RuntimeError(f"IGN elevation request failed: {last}")

    def read(self, i0, j0, i1, j1):
        """Heights of lattice cells [i0, i1) x [j0, j1) (column, row from the south), north row first."""
        span = TILE
        keys = [(a, b) for a in range(i0 // span, (i1 - 1) // span + 1) for b in range(j0 // span, (j1 - 1) // span + 1)]
        missing = [key for key in keys if key not in self.kept]
        if missing:
            done = 0
            self.report(0, len(missing), "Reading IGN elevation")
            with ThreadPoolExecutor(REQUESTS) as pool:
                for key, tile in zip(missing, pool.map(self._fetch, missing)):
                    self.kept[key] = tile
                    done += 1
                    self.report(done, len(missing), "Reading IGN elevation")
        out = np.full((j1 - j0, i1 - i0), np.nan, np.float32)
        for a, b in keys:
            tile = self.kept[(a, b)]
            # Tile rows run from its north edge: lattice row b * span + span - 1 - r.
            c0, c1 = max(i0, a * span), min(i1, (a + 1) * span)
            r0, r1 = max(j0, b * span), min(j1, (b + 1) * span)
            rows = slice((b + 1) * span - r1, (b + 1) * span - r0)
            out[j1 - r1:j1 - r0, c0 - i0:c1 - i0] = tile[rows, c0 - a * span:c1 - a * span]
        return out


def _crs(wkt):
    from rasterio.crs import CRS

    lambert = CRS.from_epsg(2154)
    if not wkt:
        return lambert, lambert, True
    crs = CRS.from_wkt(wkt)
    return crs, lambert, False


def _route(z, domain, cell):
    """Area (m2) arriving in each domain cell from outside it, and each cell's outlet index."""
    import cupy as cp
    from pyfastflow.core import Backend
    from pyfastflow.flow import SFDFlowProgram

    rows, columns = z.shape
    missing = np.isnan(z)
    outlet = domain | missing
    outlet[0, :] = outlet[-1, :] = outlet[:, 0] = outlet[:, -1] = True
    source = np.where(domain | missing, 0.0, cell * cell).astype(np.float32)
    # No-data (the sea, or past the coverage) takes what flows to it.
    heights = np.where(missing, np.nanmin(z) - 1000.0 if (~missing).any() else 0.0, z).astype(np.float32)
    flow = SFDFlowProgram(Backend.from_name("cupy"), nx=columns, ny=rows, dx=float(cell), outlet="mask", source=source)
    try:
        flow.z.from_numpy(heights)
        flow.outlet_mask.from_numpy(outlet.astype(np.uint8))
        flow.route()
        flow.resolve_minima()
        flow.accumulate()
        drainage = flow.drainage.to_numpy().reshape(rows, columns)
        end = cp.asarray(flow.rec.array).reshape(-1).astype(cp.int64)
        while True:
            jumped = end[end]
            if bool((jumped == end).all()):
                break
            end = jumped
        end = cp.asnumpy(end).reshape(rows, columns)
    finally:
        flow.close()
    return drainage, end


def compute(bed, x_min, y_min, bed_cell, crs_wkt, cell, max_km2, min_km2, report):
    """Entries of the bed ``bed`` (array from the north, NaN off the footprint) from the land upstream.

    Returns ``{"entries": [{"x", "y", "z", "area", "own_area", "merged"}], ...}``: coordinates in
    the bed's CRS, ``z`` the bed there (NaN off it), areas in m2.
    """
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject, transform, transform_bounds

    crs, lambert, assumed = _crs(crs_wkt)
    valid = ~np.isnan(bed)
    if not valid.any():
        raise ValueError("The bed has no valid cell")
    rows_b, columns_b = bed.shape
    bed_transform = from_origin(x_min, y_min + rows_b * bed_cell, bed_cell, bed_cell)
    left, bottom, right, top = transform_bounds(crs, lambert, x_min, y_min, x_min + columns_b * bed_cell, y_min + rows_b * bed_cell, densify_pts=21)
    margin = max(1000.0, 0.25 * max(right - left, top - bottom))
    box = [math.floor((left - margin) / cell), math.floor((bottom - margin) / cell),
           math.ceil((right + margin) / cell), math.ceil((top + margin) / cell)]
    cap = float(max_km2) * 1.0e6
    tiles = Tiles(cell, report)
    while True:
        i0, j0, i1, j1 = box
        z = tiles.read(i0, j0, i1, j1)
        rows, columns = z.shape
        grid = from_origin(i0 * cell, j1 * cell, cell, cell)
        placed = np.zeros((rows, columns), np.uint8)
        reproject(valid.astype(np.uint8), placed, src_transform=bed_transform, src_crs=crs, dst_transform=grid, dst_crs=lambert,
                  resampling=Resampling.nearest, src_nodata=None, dst_nodata=0)
        domain = placed > 0
        if not domain.any():
            raise ValueError("The bed is narrower than the routing cell: choose a finer cell")
        report(0, 1, "Routing")
        drainage, end = _route(z, domain, cell)
        drains = domain.reshape(-1)[end.reshape(-1)].reshape(rows, columns) & ~domain
        ring = np.zeros((rows, columns), bool)
        ring[:EDGE_RING + 1, :] = ring[-EDGE_RING - 1:, :] = ring[:, :EDGE_RING + 1] = ring[:, -EDGE_RING - 1:] = True
        cut = drains & ring & ~np.isnan(z)
        area = (i1 - i0) * (j1 - j0) * cell * cell
        if not cut.any() or area >= cap:
            break
        # Grow the sides the cut catchments touch, by half the box on that axis, less if over the cap.
        grow_x, grow_y = max(1, (i1 - i0) // 2), max(1, (j1 - j0) // 2)
        sides = [cut[:, :EDGE_RING + 1].any(), cut[-EDGE_RING - 1:, :].any(), cut[:, -EDGE_RING - 1:].any(), cut[:EDGE_RING + 1, :].any()]
        grown = box
        for share in (1.0, 0.5, 0.25, 0.125):
            gx, gy = int(grow_x * share), int(grow_y * share)
            trial = [i0 - gx * sides[0], j0 - gy * sides[1], i1 + gx * sides[2], j1 + gy * sides[3]]
            if (trial[2] - trial[0]) * (trial[3] - trial[1]) * cell * cell <= cap:
                grown = trial
                break
        if grown == box:
            break
        box = grown
    entering = domain & (drainage > 0.0)
    er, ec = np.nonzero(entering)
    areas = drainage[er, ec].astype(np.float64)
    # Routed crossings, in Lambert-93 then in the bed's CRS.
    lx = (i0 + ec + 0.5) * cell
    ly = (j1 - er - 0.5) * cell
    bx, by = (lx, ly) if crs == lambert else transform(lambert, crs, lx.tolist(), ly.tolist())
    bx, by = np.asarray(bx), np.asarray(by)
    kept = areas >= float(min_km2) * 1.0e6
    if not kept.any() and areas.size:
        kept[np.argmax(areas)] = True
    total = areas.copy()
    merged = np.zeros(areas.size, np.int64)
    k = np.nonzero(kept)[0]
    for s in np.nonzero(~kept)[0]:
        nearest = k[np.argmin((bx[k] - bx[s]) ** 2 + (by[k] - by[s]) ** 2)]
        total[nearest] += areas[s]
        merged[nearest] += 1
    reach = max(SNAP_CELLS * cell, bed_cell)
    entries = []
    for e in k:
        x, y, z_bed = _snap(bed, x_min, y_min, bed_cell, bx[e], by[e], reach)
        entries.append({"x": x, "y": y, "z": z_bed, "area": float(total[e]), "own_area": float(areas[e]), "merged": int(merged[e])})
    entries.sort(key=lambda entry: -entry["area"])
    return {"entries": entries, "total_area": float(areas.sum()), "box_area": float(area), "cut_area": float(drainage[_reached(end, cut, domain)].sum()),
            "capped": bool(cut.any()), "assumed_crs": assumed, "cell": float(cell), "crossings": int(areas.size)}


def _reached(end, cut, domain):
    """Domain cells some cut cell drains to: the area they receive may be short."""
    hit = np.zeros(end.size, bool)
    hit[end.reshape(-1)[cut.reshape(-1)]] = True
    return hit.reshape(end.shape) & domain


def _snap(bed, x_min, y_min, bed_cell, x, y, reach):
    """``(x, y, bed)`` of the lowest valid bed cell within ``reach`` of (x, y); (x, y, NaN) if none."""
    rows, columns = bed.shape
    north = y_min + rows * bed_cell
    col, row = (x - x_min) / bed_cell, (north - y) / bed_cell
    span = reach / bed_cell
    r0, r1 = max(int(math.floor(row - span)), 0), min(int(math.ceil(row + span)) + 1, rows)
    c0, c1 = max(int(math.floor(col - span)), 0), min(int(math.ceil(col + span)) + 1, columns)
    if r0 >= r1 or c0 >= c1:
        return float(x), float(y), float("nan")
    window = bed[r0:r1, c0:c1]
    ii, jj = np.mgrid[r0:r1, c0:c1]
    near = ((ii + 0.5 - row) ** 2 + (jj + 0.5 - col) ** 2 <= span * span) & ~np.isnan(window)
    if not near.any():
        return float(x), float(y), float("nan")
    index = np.argmin(np.where(near, window, np.inf))
    i, j = ii.reshape(-1)[index], jj.reshape(-1)[index]
    return float(x_min + (j + 0.5) * bed_cell), float(north - (i + 0.5) * bed_cell), float(bed[i, j])
