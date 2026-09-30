"""Online DEM sources: download a lon/lat box and reproject it to UTM.

Only open data behind plain HTTPS, no account or API key:

- ``copernicus30`` / ``copernicus90``: Copernicus DEM GLO-30 / GLO-90, 1x1 deg
  Cloud Optimised GeoTIFFs on the public AWS buckets. Only the blocks covering
  the box are read. A missing tile (open ocean) is left as nodata.
- ``srtm30``: SRTM 1 arc-second, void-filled, from the AWS "elevation-tiles-prod"
  (Mapzen/Joerd) ``skadi`` tiles. Whole tiles are downloaded and cached under
  ``~/.cache/pytopoviz/skadi``.

The output grid is the UTM envelope of the box (zone of the box centre), so it
has no nodata wedges; the geographic area read is that envelope's lon/lat
envelope. ``plan_download`` (the process's ``describe``) reports that grid, the
tiles read and the download size without downloading anything.

Author: B.G.
"""

from __future__ import annotations

import gzip
import math
import os
import shutil
import tempfile
import urllib.error
import urllib.request

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds

from ..core import Output, Param, process
from ..georaster import GeoRaster

# Per source: display label, arc-seconds per pixel (latitude), default output
# cell size (m), licence, and for the download estimate: ``block`` the COG block
# side read at once (0 = whole tiles are downloaded), ``bytes_per_px`` the
# measured compressed size of a COG pixel, ``tile_bytes`` a whole tile's size,
# ``coverage`` the latitude band of the source's own data.
SOURCES = {
    "copernicus30": {
        "label": "Copernicus GLO-30 (30 m)", "arcsec": 1, "cell_size": 30.0,
        "block": 1024, "bytes_per_px": 3.2,
        "licence": "Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and "
                   "Space GmbH 2014-2018, provided under COPERNICUS by the European Union "
                   "and ESA. Free to use, including commercially, with this attribution. "
                   "A few countries' tiles are not public; open ocean has no tiles.",
    },
    "copernicus90": {
        "label": "Copernicus GLO-90 (90 m)", "arcsec": 3, "cell_size": 90.0,
        "block": 2048, "bytes_per_px": 3.6,
        "licence": "Copernicus DEM GLO-90 © DLR e.V. 2010-2014 and © Airbus Defence and "
                   "Space GmbH 2014-2018, provided under COPERNICUS by the European Union "
                   "and ESA. Free to use, including commercially, with this attribution.",
    },
    "srtm30": {
        "label": "SRTM 1\u2033 void-filled (30 m)", "arcsec": 1, "cell_size": 30.0,
        "block": 0, "tile_bytes": 14.0e6, "coverage": (-56.0, 60.0),
        "licence": "SRTM 1 arc-second (NASA/USGS, public domain), voids filled from other "
                   "open datasets; tiles from the AWS Terrain Tiles open dataset "
                   "(Mapzen/Joerd), attribution per its source list. Tiles are cached in "
                   "~/.cache/pytopoviz/skadi and reused.",
    },
}

# Copernicus tiles have fewer columns towards the poles: from |latitude| (deg),
# the share num/den of a full tile's columns.
_COP_COLUMN_BANDS = ((85, 1, 10), (80, 1, 5), (70, 1, 3), (60, 1, 2), (50, 2, 3))
# UTM is defined over this latitude band.
_UTM_LATS = (-80.0, 84.0)

_COP_URL = ("https://copernicus-dem-{res}m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_{code}_{lat}_00_{lon}_00_DEM/"
            "Copernicus_DSM_COG_{code}_{lat}_00_{lon}_00_DEM.tif")
_SKADI_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat}/{lat}{lon}.hgt.gz"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pytopoviz", "skadi")
_USER_AGENT = "pytopoviz (https://github.com/TopoToolbox)"

# Ranged reads of the COGs without GDAL listing the bucket first.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "60",
}


def utm_epsg(lon, lat):
    """EPSG code of the standard UTM zone at (lon, lat) (no Norway/Svalbard
    exceptions)."""
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def _lat_name(lat):
    return "%s%02d" % ("N" if lat >= 0 else "S", abs(lat))


def _lon_name(lon):
    return "%s%03d" % ("E" if lon >= 0 else "W", abs(lon))


def _url_exists(url):
    """True on 200, False on 403/404 (the buckets answer either for a missing
    key); anything else (network down, 5xx) raises."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return False
        raise


def _copernicus_path(source, lat, lon):
    res, code = ("30", "10") if source == "copernicus30" else ("90", "30")
    url = _COP_URL.format(res=res, code=code, lat=_lat_name(lat), lon=_lon_name(lon))
    return "/vsicurl/" + url if _url_exists(url) else None


def _skadi_path(lat, lon):
    """Local .hgt of a skadi tile, downloading it on first use. None if the
    tile doesn't exist."""
    name = _lat_name(lat) + _lon_name(lon)
    path = os.path.join(_CACHE_DIR, name + ".hgt")
    if os.path.exists(path):
        return path
    url = _SKADI_URL.format(lat=_lat_name(lat), lon=_lon_name(lon))
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None
        raise
    os.makedirs(_CACHE_DIR, exist_ok=True)
    # Written aside then renamed, so an interrupted download never looks cached.
    fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".part")
    try:
        with resp, os.fdopen(fd, "wb") as out, gzip.GzipFile(fileobj=resp) as gz:
            shutil.copyfileobj(gz, out)
        os.replace(tmp, path)
    except BaseException:
        os.remove(tmp)
        raise
    return path


def _check_box(source, west, south, east, north):
    if source not in SOURCES:
        raise ValueError("unknown source %r (one of %s)" % (source, ", ".join(SOURCES)))
    if not (west < east and south < north):
        raise ValueError("empty box: west < east and south < north are required")
    if not (-180.0 <= west and east <= 180.0 and -90.0 <= south and north <= 90.0):
        raise ValueError("box outside -180..180, -90..90")


def _output_grid(west, south, east, north, cell):
    """The output grid of a box: UTM zone of the box centre, the box's UTM
    envelope snapped outwards to whole cells, and the lon/lat envelope of that
    rectangle (the area read). Returns (epsg, (x0, y0, x1, y1), n_cols, n_rows,
    (west, south, east, north))."""
    epsg = utm_epsg(0.5 * (west + east), 0.5 * (south + north))
    utm, wgs84 = CRS.from_epsg(epsg), CRS.from_epsg(4326)
    ux0, uy0, ux1, uy1 = transform_bounds(wgs84, utm, west, south, east, north, densify_pts=41)
    ux0, uy0 = math.floor(ux0 / cell) * cell, math.floor(uy0 / cell) * cell
    ux1, uy1 = math.ceil(ux1 / cell) * cell, math.ceil(uy1 / cell) * cell
    n_cols, n_rows = int(round((ux1 - ux0) / cell)), int(round((uy1 - uy0) / cell))
    read = transform_bounds(utm, wgs84, ux0, uy0, ux1, uy1, densify_pts=41)
    return epsg, (ux0, uy0, ux1, uy1), n_cols, n_rows, read


def _lattice(source, west, south, east, north):
    """The mosaic over a lon/lat area: the source's pixel lattice (edges at
    k/3600 deg - half a pixel) padded by two pixels. Returns (transform,
    n_rows, n_cols, (x0, y0, x1, y1))."""
    d = SOURCES[source]["arcsec"] / 3600.0
    pad = 2 * d
    x0 = math.floor((west - pad + 0.5 * d) / d) * d - 0.5 * d
    y1 = math.ceil((north + pad + 0.5 * d) / d) * d - 0.5 * d
    n_cols = int(math.ceil((east + pad - x0) / d))
    n_rows = int(math.ceil((y1 - (south - pad)) / d))
    x1, y0 = x0 + n_cols * d, y1 - n_rows * d
    return Affine(d, 0.0, x0, 0.0, -d, y1), n_rows, n_cols, (x0, y0, x1, y1)


def _tiles(bounds):
    """(lat, lon) of the lower-left corner of every 1 deg tile over bounds."""
    x0, y0, x1, y1 = bounds
    return [(lat, lon)
            for lat in range(int(math.floor(y0)), int(math.ceil(y1)))
            for lon in range(int(math.floor(x0)), int(math.ceil(x1)))]


def _mosaic(source, west, south, east, north):
    """Every tile over the area pasted into one lon/lat grid on the source's
    pixel lattice. Returns (z, transform)."""
    transform, n_rows, n_cols, (x0, y0, x1, y1) = _lattice(source, west, south, east, north)
    d = transform.a
    z = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    wgs84 = CRS.from_epsg(4326)

    found = 0
    for lat, lon in _tiles((x0, y0, x1, y1)):
        if source == "srtm30":
            path = _skadi_path(lat, lon)
        else:
            path = _copernicus_path(source, lat, lon)
        if path is None:
            continue
        found += 1
        with rasterio.open(path) as src:
            # Only the part of the tile inside the mosaic, a pixel wider.
            win = window_from_bounds(max(x0, lon) - d, max(y0, lat) - d,
                                     min(x1, lon + 1) + d, min(y1, lat + 1) + d,
                                     src.transform)
            c0, r0 = math.floor(win.col_off), math.floor(win.row_off)
            c1 = math.ceil(win.col_off + win.width)
            r1 = math.ceil(win.row_off + win.height)
            win = Window(c0, r0, c1 - c0, r1 - r0).intersection(
                Window(0, 0, src.width, src.height))
            data = src.read(1, window=win, out_dtype="float32")
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            # Same lattice as the mosaic for 1"/3" tiles, so nearest is a
            # copy; high-latitude Copernicus tiles (coarser in longitude)
            # get repeated columns.
            reproject(data, z, src_transform=src.window_transform(win), src_crs=wgs84,
                      dst_transform=transform, dst_crs=wgs84,
                      src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=Resampling.nearest, init_dest_nodata=False)
    if found == 0:
        raise ValueError("no %s data in this box (open ocean?)" % source)
    return z, transform


def _cols_per_deg(source, lat):
    """Columns per degree of longitude of a Copernicus tile at latitude lat."""
    full = 3600 // SOURCES[source]["arcsec"]
    band = min(abs(lat), abs(lat + 1))
    for from_lat, num, den in _COP_COLUMN_BANDS:
        if band >= from_lat:
            return full * num // den
    return full


def _download_bytes(source, bounds):
    """Worst case (every tile exists, i.e. no ocean) bytes read for the mosaic
    over bounds: COG sources read only the blocks covering it, the others
    download whole tiles."""
    src = SOURCES[source]
    x0, y0, x1, y1 = bounds
    total = 0.0
    for lat, lon in _tiles(bounds):
        if src["block"] == 0:
            total += src["tile_bytes"]
            continue
        rows, cols, block = 3600 // src["arcsec"], _cols_per_deg(source, lat), src["block"]
        # Part of the area inside this tile, in tile pixels (row 0 = north edge).
        c0 = min(max(math.floor((x0 - lon) * cols), 0), cols - 1)
        c1 = min(max(math.ceil((x1 - lon) * cols), 1), cols)
        r0 = min(max(math.floor((lat + 1 - y1) * rows), 0), rows - 1)
        r1 = min(max(math.ceil((lat + 1 - y0) * rows), 1), rows)
        for br in range(r0 // block, (r1 - 1) // block + 1):
            for bc in range(c0 // block, (c1 - 1) // block + 1):
                h = min(block, rows - br * block)
                w = min(block, cols - bc * block)
                total += h * w * src["bytes_per_px"]
    return total


def plan_download(source="copernicus30", west=0.0, south=0.0, east=0.0, north=0.0,
                  cell_size=None):
    """What ``download_dem`` will produce for these arguments, without
    downloading: source label and licence, UTM zone, grid shape and extent,
    tiles read, worst-case download size, and warnings."""
    _check_box(source, west, south, east, north)
    src = SOURCES[source]
    cell = float(cell_size) if cell_size else src["cell_size"]
    epsg, (ux0, uy0, ux1, uy1), n_cols, n_rows, read = _output_grid(west, south, east, north, cell)
    bounds = _lattice(source, *read)[3]

    warnings = []
    lo, hi = src.get("coverage", (-90.0, 90.0))
    if south < lo or north > hi:
        warnings.append("%s covers %g° to %g° latitude; outside that it is filled from "
                        "other datasets or empty." % (src["label"], lo, hi))
    if south < _UTM_LATS[0] or north > _UTM_LATS[1]:
        warnings.append("UTM is defined for 80°S to 84°N; the grid is distorted beyond.")
    if east - west > 6.0:
        warnings.append("Box wider than one UTM zone (6°): scale distortion grows away "
                        "from zone %d." % (epsg % 100))
    return {
        "label": src["label"],
        "licence": src["licence"],
        "epsg": epsg,
        "zone": epsg % 100,
        "north": epsg < 32700,
        "cell_size": cell,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "width_m": ux1 - ux0,
        "height_m": uy1 - uy0,
        "tiles": len(_tiles(bounds)),
        "download_bytes": _download_bytes(source, bounds),
        "warnings": warnings,
    }


@process(
    id="dem_sources.download_dem",
    label="Download DEM",
    params=[
        Param("source", "string", default="copernicus30",
              choices=list(SOURCES), choice_labels=[s["label"] for s in SOURCES.values()],
              doc="Where the elevations come from: Copernicus DEM GLO-30 / GLO-90, or "
                  "SRTM 1 arc-second (void-filled, AWS elevation tiles)."),
        Param("west", "float", doc="Western edge, degrees longitude."),
        Param("south", "float", doc="Southern edge, degrees latitude."),
        Param("east", "float", doc="Eastern edge, degrees longitude."),
        Param("north", "float", doc="Northern edge, degrees latitude."),
        Param("cell_size", "float", optional=True, min=0.0,
              doc="Output cell size in metres; unset or 0: the source's (30 or 90)."),
    ],
    outputs=[Output("grid", "georaster")],
    impl="library",
    describe=plan_download,
)
def download_dem(source="copernicus30", west=0.0, south=0.0, east=0.0, north=0.0, cell_size=None):
    """Download a DEM for a lon/lat box and reproject it to UTM (zone of the
    box centre), bilinear, nodata as NaN."""
    _check_box(source, west, south, east, north)
    cell = float(cell_size) if cell_size else SOURCES[source]["cell_size"]
    epsg, (ux0, uy0, ux1, uy1), n_cols, n_rows, read = _output_grid(west, south, east, north, cell)

    with rasterio.Env(**_GDAL_ENV):
        z, src_transform = _mosaic(source, *read)

    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    reproject(z, out, src_transform=src_transform, src_crs=CRS.from_epsg(4326),
              dst_transform=from_bounds(ux0, uy0, ux1, uy1, n_cols, n_rows),
              dst_crs=CRS.from_epsg(epsg),
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return GeoRaster(z=out, cell_size=cell, x_min=ux0, y_min=uy0, epsg=epsg)
