"""Online raster sources: a lon/lat box read straight into one grid of a target
CRS (the scene's; by default the UTM zone of the box centre, the polar
stereographic one for ArcticDEM / REMA).

Only open data behind plain HTTPS, no account or API key (SOURCES: label,
licence, coverage; ``nc`` marks non-commercial licences). Kinds:

- ``cog``: Cloud Optimised GeoTIFF tiles (Copernicus GLO-30/90, NASADEM,
  ArcticDEM, REMA, swissALTI3D, ESA WorldCover, Sentinel-2 L2A bands; FABDEM's
  1 deg GeoTIFFs inside its 10 deg zips, copied whole once). The overview not
  coarser than the cell is read, in the file's own blocks, several at once;
  every block read is kept on disk (``cache_dir``) and reused.
- ``skadi``: SRTM 1" void-filled, whole tiles (AWS elevation-tiles-prod).
- ``terrarium``: AWS Terrain Tiles (web mercator PNGs, land and bathymetry),
  the zoom not coarser than the cell.
- ``wms_bil`` / ``wcs`` / ``arcgis``: national services (IGN RGE ALTI and
  LiDAR HD grids, AHN4, Environment Agency, PNOA, EMODnet, USGS 3DEP), read in
  tiles of a fixed lattice per rung of LADDER (the finest rung not coarser
  than the cell, not finer than the source), kept on disk, so nearby cell
  sizes share their tiles.

All tiles of a source are pasted on the source's own lattice, then warped
once into the output grid (bilinear, or average when the cell is coarser;
nearest / mode for classes). ``plan_download`` (the process's ``describe``)
reports the grid, the requests and an estimate of the bytes read, without
network.

Progress (pytopoviz.core.progress): "downloading" in requests, then
"reprojecting" (no measure).

Author: B.G.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window

from ..core import Output, Param, process, progress
from ..georaster import GeoRaster

_USER_AGENT = "pytopoviz (https://github.com/TopoToolbox)"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pytopoviz")
#: Requests at once.
WORKERS = 8
#: Cell sizes (m) of the service lattices.
LADDER = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
#: Largest lattice mosaic read at once (pixels).
MAX_MOSAIC = 600_000_000
#: Metres per degree of latitude.
DEG = 111320.0
# UTM is defined over this latitude band.
_UTM_LATS = (-80.0, 84.0)

_IGN = "IGN Géoplateforme, Licence Ouverte Etalab 2.0"
_FRANCE = (-5.5, 41.0, 10.0, 51.5)

# Per source: label, kind, licence, nc (non-commercial), coverage (lon/lat
# west, south, east, north; None: world), cell (default output, m), native (m),
# overviews (factors of the native read), bpp (bytes read per native pixel, for
# the estimate), classes (categorical), and the kind's own keys.
SOURCES = {
    "copernicus30": {
        "label": "Copernicus GLO-30 (30 m)", "kind": "cog", "cell": 30.0, "native": 30.0,
        "overviews": (1, 2, 4, 8), "bpp": 3.2, "coverage": None,
        "licence": "Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, "
                   "provided under COPERNICUS by the European Union and ESA. Free to use, including commercially, "
                   "with this attribution. A few countries' tiles are not public; open ocean has no tiles.",
    },
    "copernicus90": {
        "label": "Copernicus GLO-90 (90 m)", "kind": "cog", "cell": 90.0, "native": 90.0,
        "overviews": (1, 2, 4, 8), "bpp": 3.6, "coverage": None,
        "licence": "Copernicus DEM GLO-90 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, "
                   "provided under COPERNICUS by the European Union and ESA. Free to use, including commercially, "
                   "with this attribution.",
    },
    "srtm30": {
        "label": "SRTM 1″ void-filled (30 m)", "kind": "skadi", "cell": 30.0, "native": 30.0,
        "bpp": 1.1, "coverage": (-180.0, -56.0, 180.0, 60.0),
        "licence": "SRTM 1 arc-second (NASA/USGS, public domain), voids filled from other open datasets; tiles "
                   "from the AWS Terrain Tiles open dataset (Mapzen/Joerd), attribution per its source list.",
    },
    "nasadem": {
        "label": "NASADEM (30 m)", "kind": "cog", "cell": 30.0, "native": 30.0, "overviews": (1,),
        "bpp": 2.0, "coverage": (-180.0, -56.0, 180.0, 60.0),
        "licence": "NASADEM (NASA JPL), public domain; Cloud Optimised GeoTIFFs from Microsoft Planetary Computer.",
    },
    "fabdem": {
        "label": "FABDEM V1-2, forests and buildings removed (30 m)", "kind": "cog", "cell": 30.0,
        "native": 30.0, "overviews": (1,), "bpp": 2.0, "whole": True, "nc": True, "coverage": (-180.0, -60.0, 180.0, 80.0),
        "licence": "FABDEM V1-2 (Hawker et al. 2022, University of Bristol), CC BY-NC-SA 4.0: non-commercial use "
                   "only. Each 1° tile is copied whole from its 10° zip once.",
    },
    "terrain_tiles": {
        "label": "Terrain Tiles, land and bathymetry", "kind": "terrarium", "cell": 100.0, "native": 5.0,
        "max_zoom": 15, "bpp": 1.5, "coverage": None,
        "licence": "AWS Terrain Tiles (Mapzen/Joerd): SRTM, GMTED2010, ETOPO1, GEBCO and national datasets; "
                   "attribution per its source list (github.com/tilezen/joerd).",
    },
    "arcticdem32": {
        "label": "ArcticDEM v4.1 mosaic (32 m)", "kind": "cog", "cell": 32.0, "native": 32.0,
        "overviews": (1, 2, 4, 8), "bpp": 2.0, "epsg": 3413, "pgc": ("arcticdem", "v4.1", "32m", 41),
        "coverage": (-180.0, 50.0, 180.0, 90.0),
        "licence": "ArcticDEM v4.1, Polar Geospatial Center (University of Minnesota), from Maxar imagery; CC BY 4.0.",
    },
    "arcticdem10": {
        "label": "ArcticDEM v4.1 mosaic (10 m)", "kind": "cog", "cell": 10.0, "native": 10.0,
        "overviews": (1, 2, 4, 8, 16, 32), "bpp": 2.0, "epsg": 3413, "pgc": ("arcticdem", "v4.1", "10m", 41),
        "coverage": (-180.0, 50.0, 180.0, 90.0),
        "licence": "ArcticDEM v4.1, Polar Geospatial Center (University of Minnesota), from Maxar imagery; CC BY 4.0.",
    },
    "rema32": {
        "label": "REMA v2.0 mosaic (32 m)", "kind": "cog", "cell": 32.0, "native": 32.0,
        "overviews": (1, 2, 4, 8), "bpp": 2.0, "epsg": 3031, "pgc": ("rema", "v2.0", "32m", 31),
        "coverage": (-180.0, -90.0, 180.0, -60.0),
        "licence": "REMA v2.0, Polar Geospatial Center (University of Minnesota), from Maxar imagery; CC BY 4.0.",
    },
    "rema10": {
        "label": "REMA v2.0 mosaic (10 m)", "kind": "cog", "cell": 10.0, "native": 10.0,
        "overviews": (1, 2, 4, 8, 16, 32), "bpp": 2.0, "epsg": 3031, "pgc": ("rema", "v2.0", "10m", 31),
        "coverage": (-180.0, -90.0, 180.0, -60.0),
        "licence": "REMA v2.0, Polar Geospatial Center (University of Minnesota), from Maxar imagery; CC BY 4.0.",
    },
    "ign_rgealti": {
        "label": "IGN RGE ALTI (1 m)", "kind": "wms_bil", "cell": 5.0, "native": 1.0, "bpp": 4.0,
        "epsg": 2154, "layer": "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES", "tile": 1024, "coverage": _FRANCE,
        "licence": _IGN,
    },
    "lidarhd_mnt": {
        "label": "IGN LiDAR HD terrain, MNT (50 cm)", "kind": "wms_bil", "cell": 1.0, "native": 0.5, "bpp": 4.0,
        "epsg": 2154, "layer": "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", "tile": 1024,
        "coverage": _FRANCE, "licence": _IGN + "; only where LiDAR HD is published.",
    },
    "lidarhd_mns": {
        "label": "IGN LiDAR HD surface, MNS (50 cm)", "kind": "wms_bil", "cell": 1.0, "native": 0.5, "bpp": 4.0,
        "epsg": 2154, "layer": "IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", "tile": 1024,
        "coverage": _FRANCE, "licence": _IGN + "; only where LiDAR HD is published.",
    },
    "lidarhd_mnh": {
        "label": "IGN LiDAR HD canopy height, MNH (50 cm)", "kind": "wms_bil", "cell": 1.0, "native": 0.5,
        "bpp": 4.0, "epsg": 2154, "layer": "IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93",
        "tile": 1024, "coverage": _FRANCE, "licence": _IGN + "; only where LiDAR HD is published.",
    },
    "swissalti3d": {
        "label": "swissALTI3D (0.5 / 2 m)", "kind": "cog", "cell": 2.0, "native": 0.5, "overviews": (1, 4),
        "bpp": 2.0, "epsg": 2056, "coverage": (5.9, 45.8, 10.5, 47.9),
        "licence": "swissALTI3D © swisstopo, open government data: free use with the attribution "
                   "\"Source: Federal Office of Topography swisstopo\".",
    },
    "ahn4_dtm": {
        "label": "AHN4 terrain, DTM (50 cm)", "kind": "wcs", "cell": 1.0, "native": 0.5, "bpp": 4.0,
        "epsg": 28992, "tile": 1000, "url": "https://service.pdok.nl/rws/ahn/wcs/v1_0", "coverage_id": "dtm_05m",
        "axes": ("x", "y"), "scale_axes": ("x", "y"), "coverage": (3.2, 50.7, 7.3, 53.6),
        "licence": "AHN4 (Actueel Hoogtebestand Nederland), Rijkswaterstaat / PDOK, CC0 1.0.",
    },
    "ahn4_dsm": {
        "label": "AHN4 surface, DSM (50 cm)", "kind": "wcs", "cell": 1.0, "native": 0.5, "bpp": 4.0,
        "epsg": 28992, "tile": 1000, "url": "https://service.pdok.nl/rws/ahn/wcs/v1_0", "coverage_id": "dsm_05m",
        "axes": ("x", "y"), "scale_axes": ("x", "y"), "coverage": (3.2, 50.7, 7.3, 53.6),
        "licence": "AHN4 (Actueel Hoogtebestand Nederland), Rijkswaterstaat / PDOK, CC0 1.0.",
    },
    "ea_dtm": {
        "label": "England LIDAR composite DTM (1 m)", "kind": "wcs", "cell": 2.0, "native": 1.0, "bpp": 4.0,
        "epsg": 27700, "tile": 1000,
        "url": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
        "coverage_id": "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
        "axes": ("E", "N"), "scale_axes": ("i", "j"), "coverage": (-6.5, 49.8, 2.0, 55.9),
        "licence": "Environment Agency LIDAR Composite DTM 1 m, © Environment Agency, Open Government Licence v3.0.",
    },
    "pnoa_mdt": {
        "label": "Spain MDT05, PNOA (5 m)", "kind": "wcs", "cell": 5.0, "native": 5.0, "bpp": 2.0,
        "epsg": 25830, "tile": 1000, "url": "https://servicios.idee.es/wcs-inspire/mdt",
        "coverage_id": "Elevacion25830_5", "axes": ("x", "y"), "scale_axes": ("x", "y"),
        "coverage": (-9.5, 35.9, 4.4, 43.9),
        "licence": "MDT05 © Instituto Geográfico Nacional / CNIG (PNOA), CC BY 4.0.",
    },
    "usgs_3dep": {
        "label": "USGS 3DEP (1 m where flown, else 1/3″)", "kind": "arcgis", "cell": 10.0, "native": 1.0,
        "bpp": 4.0, "epsg": 4326, "tile": 1000,
        "url": "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage",
        "coverage": (-180.0, 17.5, -64.0, 72.0),
        "licence": "USGS 3D Elevation Program (3DEP), public domain.",
    },
    "emodnet": {
        "label": "EMODnet Bathymetry DTM (1/16′, ~115 m)", "kind": "wcs", "cell": 115.0, "native": 115.0,
        "bpp": 4.0, "epsg": 4326, "tile": 1000, "url": "https://ows.emodnet-bathymetry.eu/wcs",
        "coverage_id": "emodnet__mean", "axes": ("Long", "Lat"), "scale_axes": ("i", "j"), "lat_first": True,
        "coverage": (-36.0, 15.0, 43.0, 90.0),
        "licence": "EMODnet Bathymetry DTM, EMODnet Bathymetry Consortium; free use with attribution.",
    },
    "worldcover": {
        "label": "ESA WorldCover 2021 (10 m classes)", "kind": "cog", "cell": 10.0, "native": 10.0,
        "overviews": (1, 2, 4, 8, 16, 32, 64), "bpp": 0.3, "classes": True, "coverage": None,
        "licence": "ESA WorldCover 10 m 2021 v200, © ESA WorldCover project / Contains modified Copernicus "
                   "Sentinel data (2021) processed by the WorldCover consortium, CC BY 4.0.",
    },
    "sentinel2_l2a": {
        "label": "Sentinel-2 L2A band (10-60 m)", "kind": "cog", "cell": 10.0, "native": 10.0,
        "overviews": (1, 2, 4, 8, 16), "bpp": 1.5, "coverage": None, "hrefs_given": True,
        "licence": "Contains modified Copernicus Sentinel data (free, full and open); scenes from Element 84 "
                   "Earth Search.",
    },
}

#: Typical Manning n of the WorldCover classes (open channel and floodplain
#: tables: Chow 1959, Arcement & Schneider 1989).
MANNING = {10: 0.10, 20: 0.05, 30: 0.035, 40: 0.04, 50: 0.02, 60: 0.025, 70: 0.02, 80: 0.03, 90: 0.07,
           95: 0.12, 100: 0.035}

_COP_URL = ("https://copernicus-dem-{res}m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_{code}_{lat}_00_{lon}_00_DEM/"
            "Copernicus_DSM_COG_{code}_{lat}_00_{lon}_00_DEM.tif")
_SKADI_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat}/{lat}{lon}.hgt.gz"
_TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_NASADEM_URL = "https://nasademeuwest.blob.core.windows.net/nasadem-cog/v001/NASADEM_HGT_{lat}{lon}.tif"
_NASADEM_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/nasademeuwest/nasadem-cog"
_FABDEM_URL = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/{block}_FABDEM_V1-2.zip"
_PGC_URL = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/{dem}/mosaics/{v}/{res}/{t}/{t}_{res}_{v}_dem.tif"
_SWISS_STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
_WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                   "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
_IGN_WMS = ("https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer}&STYLES="
            "&CRS=EPSG:2154&BBOX={left!r},{bottom!r},{right!r},{top!r}&WIDTH={size}&HEIGHT={size}"
            "&FORMAT=image/x-bil;bits=32")
_HALF = 20037508.342789244

# Ranged reads of the COGs without GDAL listing the bucket first.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.zip",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_USERAGENT": _USER_AGENT,
}
# Bytes per chunk of a whole-file download.
_CHUNK = 1 << 20


def utm_epsg(lon, lat):
    """EPSG code of the standard UTM zone at (lon, lat) (no Norway/Svalbard
    exceptions)."""
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def _lat_name(lat, upper=True):
    s = "%s%02d" % ("N" if lat >= 0 else "S", abs(lat))
    return s if upper else s.lower()


def _lon_name(lon, upper=True):
    s = "%s%03d" % ("E" if lon >= 0 else "W", abs(lon))
    return s if upper else s.lower()


def _get(url, timeout=60):
    """Body of ``url``; None on 403/404 (a missing tile); raises otherwise."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None
        raise


def _missing(exc):
    text = str(exc)
    return any(s in text for s in ("404", "403", "does not exist", "No such file", "not recognized"))


def _atomic_write(path, write):
    """Calls ``write(file)`` on a temporary file then renames it to ``path``,
    so an interrupted write never looks cached."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _load(path):
    """A cached array (touched for the cache's LRU), or None."""
    try:
        a = np.load(path)
    except (OSError, ValueError):
        return None
    os.utime(path)
    return a


def _save(path, array):
    _atomic_write(path, lambda f: np.save(f, array))


# ---------------------------------------------------------------------------
# The output grid
# ---------------------------------------------------------------------------

def _check_box(source, west, south, east, north):
    if source not in SOURCES:
        raise ValueError("unknown source %r (one of %s)" % (source, ", ".join(SOURCES)))
    if not (west < east and south < north):
        raise ValueError("empty box: west < east and south < north are required")
    if not (-180.0 <= west and east <= 180.0 and -90.0 <= south and north <= 90.0):
        raise ValueError("box outside -180..180, -90..90")


def _target_crs(source, west, south, east, north, epsg=None, crs_wkt=None):
    if epsg:
        return CRS.from_epsg(int(epsg))
    if crs_wkt:
        return CRS.from_wkt(crs_wkt)
    if "pgc" in SOURCES[source]:
        return CRS.from_epsg(SOURCES[source]["epsg"])
    return CRS.from_epsg(utm_epsg(0.5 * (west + east), 0.5 * (south + north)))


def _output_grid(source, west, south, east, north, cell, epsg=None, crs_wkt=None):
    """The output grid of a box: the box's envelope in the target CRS snapped
    outwards to whole cells, and the lon/lat envelope of that rectangle (the
    area read). Returns (crs, (x0, y0, x1, y1), n_cols, n_rows, (west, south,
    east, north))."""
    crs = _target_crs(source, west, south, east, north, epsg, crs_wkt)
    wgs84 = CRS.from_epsg(4326)
    ux0, uy0, ux1, uy1 = transform_bounds(wgs84, crs, west, south, east, north, densify_pts=41)
    ux0, uy0 = math.floor(ux0 / cell) * cell, math.floor(uy0 / cell) * cell
    ux1, uy1 = math.ceil(ux1 / cell) * cell, math.ceil(uy1 / cell) * cell
    n_cols, n_rows = int(round((ux1 - ux0) / cell)), int(round((uy1 - uy0) / cell))
    read = transform_bounds(crs, wgs84, ux0, uy0, ux1, uy1, densify_pts=41)
    return crs, (ux0, uy0, ux1, uy1), n_cols, n_rows, read


def _read_metres(source, cell):
    """The resolution read for an output cell (m): the coarsest overview (COG),
    rung (services) or zoom (tiles) not coarser than the cell, never finer
    than the source."""
    src = SOURCES[source]
    native = src["native"]
    kind = src["kind"]
    if kind == "cog":
        f = max([f for f in src["overviews"] if native * f <= max(cell, native)] or [1])
        return native * f
    if kind in ("wms_bil", "wcs", "arcgis"):
        return _rung(source, cell)
    if kind == "terrarium":
        return 2.0 * _HALF / (256.0 * 2 ** _zoom(source, cell, 0.0))
    return native


def _rung(source, cell):
    native = SOURCES[source]["native"]
    fits = [r for r in LADDER if native <= r <= max(cell, native)]
    return fits[-1] if fits else native


def _zoom(source, cell, lat):
    px = 2.0 * _HALF * math.cos(math.radians(min(abs(lat), 85.0))) / 256.0
    return int(np.clip(math.ceil(math.log2(px / cell)), 0, SOURCES[source]["max_zoom"]))


# ---------------------------------------------------------------------------
# Readers: each returns [(array float32 NaN nodata, transform, crs)] on one
# lattice, and reports "downloading".
# ---------------------------------------------------------------------------

def _hrefs(source, read, cell):
    """The COG hrefs of a cog source over lon/lat ``read``."""
    west, south, east, north = read
    src = SOURCES[source]
    tiles = [(lat, lon) for lat in range(int(math.floor(south)), int(math.ceil(north)))
             for lon in range(int(math.floor(west)), int(math.ceil(east)))]
    if source in ("copernicus30", "copernicus90"):
        res, code = ("30", "10") if source == "copernicus30" else ("90", "30")
        return ["/vsicurl/" + _COP_URL.format(res=res, code=code, lat=_lat_name(lat), lon=_lon_name(lon))
                for lat, lon in tiles]
    if source == "nasadem":
        token = _nasadem_token()
        return ["/vsicurl/" + _NASADEM_URL.format(lat=_lat_name(lat, False), lon=_lon_name(lon, False))
                + "?" + token for lat, lon in tiles]
    if source == "fabdem":
        out = []
        for lat, lon in tiles:
            b0, l0 = (lat // 10) * 10, (lon // 10) * 10
            block = "%s%s-%s%s" % (_lat_name(b0), _lon_name(l0), _lat_name(b0 + 10), _lon_name(l0 + 10))
            out.append("/vsizip//vsicurl/%s/%s%s_FABDEM_V1-2.tif"
                       % (_FABDEM_URL.format(block=block), _lat_name(lat), _lon_name(lon)))
        return out
    if source == "worldcover":
        names = sorted({"%s%s" % (_lat_name(lat // 3 * 3), _lon_name(lon // 3 * 3)) for lat, lon in tiles})
        return ["/vsicurl/" + _WORLDCOVER_URL.format(tile=t) for t in names]
    if "pgc" in src:
        dem, v, res, off = src["pgc"]
        x0, y0, x1, y1 = transform_bounds(CRS.from_epsg(4326), CRS.from_epsg(src["epsg"]), *read, densify_pts=41)
        out = []
        for row in range(int(math.floor(y0 / 1e5)) + off, int(math.floor(y1 / 1e5)) + off + 1):
            for col in range(int(math.floor(x0 / 1e5)) + off, int(math.floor(x1 / 1e5)) + off + 1):
                t = "%02d_%02d" % (row, col)
                out.append("/vsicurl/" + _PGC_URL.format(dem=dem, v=v, res=res, t=t))
        return out
    if source == "swissalti3d":
        return _swiss_hrefs(read, cell)
    raise ValueError("%s needs its hrefs" % source)


_token = {"value": "", "until": 0.0}
_token_lock = threading.Lock()


def _nasadem_token():
    with _token_lock:
        if time.time() < _token["until"]:
            return _token["value"]
        body = _get(_NASADEM_TOKEN)
        if body is None:
            raise ValueError("Planetary Computer refused the NASADEM token")
        data = json.loads(body)
        _token["value"] = data["token"]
        _token["until"] = time.time() + 1800.0
        return _token["value"]


def _swiss_hrefs(read, cell):
    """The latest swissALTI3D tile of every km in ``read``: the 2 m file when
    the cell allows, else the 0.5 m one."""
    want = "_2_2056_" if cell >= 2.0 else "_0.5_2056_"
    url = _SWISS_STAC + "?" + urllib.parse.urlencode({"bbox": "%.6f,%.6f,%.6f,%.6f" % tuple(read), "limit": 100})
    latest = {}
    while url:
        body = _get(url)
        if body is None:
            break
        data = json.loads(body)
        for f in data.get("features", []):
            parts = f["id"].split("_")  # swissalti3d_<year>_<tile>
            year, tile = int(parts[1]), parts[2]
            for asset in f["assets"].values():
                href = asset["href"]
                if want in href and href.endswith(".tif") and latest.get(tile, (0,))[0] < year:
                    latest[tile] = (year, href)
        url = next((l["href"] for l in data.get("links", []) if l.get("rel") == "next"), None)
    return ["/vsicurl/" + h for _y, h in latest.values()]


def _key(href):
    return hashlib.sha1(href.split("?", 1)[0].encode()).hexdigest()[:20]


class _Datasets(threading.local):
    """One dataset per thread, href and overview (GDAL handles are not shared
    between threads); ``close_all`` closes every thread's."""

    def __init__(self):
        self.sets = {}

    def open(self, href, level, every, lock):
        ds = self.sets.get((href, level))
        if ds is None:
            ds = rasterio.open(href, overview_level=level) if level >= 0 else rasterio.open(href)
            self.sets[(href, level)] = ds
            with lock:
                every.append(ds)
        return ds


def _materialize(href, cache):
    """A whole copy of ``href`` under the cache (FABDEM's GeoTIFFs inside
    zips: random reads in a deflated member restart its decompression)."""
    with rasterio.Env(**_GDAL_ENV):  # GDAL options are per thread
        return _materialize_here(href, cache)


def _materialize_here(href, cache):
    path = os.path.join(cache, "whole", _key(href) + ".tif")
    if os.path.exists(path):
        os.utime(path)
        return path
    try:
        with rasterio.open(href) as src:
            profile = src.profile
            data = src.read()
    except rasterio.errors.RasterioIOError as exc:
        if _missing(exc):
            return None
        raise
    profile.update(driver="GTiff", tiled=True, blockxsize=256, blockysize=256, compress="deflate")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
    os.replace(tmp, path)
    return path


def _cog_plan(href, factor, read):
    """The blocks of ``href`` covering lon/lat ``read`` at the overview not
    coarser than ``factor`` (None: no such file, or nothing inside)."""
    try:
        with rasterio.open(href) as ds:
            ovr = ds.overviews(1)
            level = max([i for i, f in enumerate(ovr) if f <= factor] or [-1])
    except rasterio.errors.RasterioIOError as exc:
        if _missing(exc):
            return None
        raise
    with (rasterio.open(href, overview_level=level) if level >= 0 else rasterio.open(href)) as ds:
        x0, y0, x1, y1 = transform_bounds(CRS.from_epsg(4326), ds.crs, *read, densify_pts=41)
        win = ds.window(x0, y0, x1, y1)
        c0 = max(int(math.floor(win.col_off)) - 1, 0)
        r0 = max(int(math.floor(win.row_off)) - 1, 0)
        c1 = min(int(math.ceil(win.col_off + win.width)) + 1, ds.width)
        r1 = min(int(math.ceil(win.row_off + win.height)) + 1, ds.height)
        if c1 <= c0 or r1 <= r0:
            return None
        bh, bw = ds.block_shapes[0]
        blocks = [(bi, bj) for bi in range(r0 // bh, (r1 - 1) // bh + 1)
                  for bj in range(c0 // bw, (c1 - 1) // bw + 1)]
        return {"href": href, "level": level, "bh": bh, "bw": bw, "blocks": blocks,
                "win": (r0 // bh * bh, c0 // bw * bw, min(((r1 - 1) // bh + 1) * bh, ds.height),
                        min(((c1 - 1) // bw + 1) * bw, ds.width)),
                "crs": ds.crs, "transform": ds.transform, "nodata": ds.nodata, "size": (ds.height, ds.width)}


def _read_cog(source, read, cell, cache, hrefs=None):
    """Every COG over ``read`` at the overview for ``cell``: blocks read (or
    taken from the cache) several at once, assembled per file."""
    src = SOURCES[source]
    hrefs = hrefs if hrefs is not None else _hrefs(source, read, cell)
    if src.get("whole"):
        paths = []
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(_materialize, h, cache) for h in hrefs]
            for n, fut in enumerate(as_completed(futures), 1):
                progress.report(n, len(futures) + 1, "downloading")
                if fut.result() is not None:
                    paths.append(fut.result())
        hrefs = paths
    factor = int(round(_read_metres(source, cell) / src["native"]))

    def plan_of(href):
        with rasterio.Env(**_GDAL_ENV):  # GDAL options are per thread
            return _cog_plan(href, factor, read)

    plans = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for plan in pool.map(plan_of, hrefs):
            if plan is not None:
                plans.append(plan)
    if not plans:
        raise ValueError("no %s data in this area" % src["label"])
    datasets = _Datasets()
    every, lock = [], threading.Lock()

    def block(plan, bi, bj):
        path = os.path.join(cache, "blocks", _key(plan["href"]), str(plan["level"]), "%d_%d.npy" % (bi, bj))
        a = _load(path)
        if a is None:
            h, w = plan["size"]
            r, c = bi * plan["bh"], bj * plan["bw"]
            win = Window(c, r, min(plan["bw"], w - c), min(plan["bh"], h - r))
            with rasterio.Env(**_GDAL_ENV):  # GDAL options are per thread
                a = datasets.open(plan["href"], plan["level"], every, lock).read(1, window=win)
            if plan["href"].startswith("/vsi"):
                _save(path, a)  # a local file (a whole copy) is not kept twice
        return plan, bi, bj, a

    pieces = []
    arrays = {id(p): np.full((p["win"][2] - p["win"][0], p["win"][3] - p["win"][1]), np.nan, np.float32)
              for p in plans}
    total = sum(len(p["blocks"]) for p in plans)
    futures = []
    with ThreadPoolExecutor(WORKERS) as pool:
        try:
            futures = [pool.submit(block, p, bi, bj) for p in plans for bi, bj in p["blocks"]]
            for n, fut in enumerate(as_completed(futures), 1):
                plan, bi, bj, a = fut.result()
                progress.report(n, total, "downloading")
                a = a.astype(np.float32)
                if plan["nodata"] is not None:
                    a[a == plan["nodata"]] = np.nan
                r = bi * plan["bh"] - plan["win"][0]
                c = bj * plan["bw"] - plan["win"][1]
                arrays[id(plan)][r:r + a.shape[0], c:c + a.shape[1]] = a
        finally:
            for f in futures:
                f.cancel()
    for ds in every:
        ds.close()
    for p in plans:
        t = p["transform"]
        pieces.append((arrays[id(p)], t * Affine.translation(p["win"][1], p["win"][0]), p["crs"]))
    return pieces


def _skadi_path(lat, lon, cache):
    """Local .hgt of a skadi tile, downloading it on first use. None if the
    tile doesn't exist."""
    name = _lat_name(lat) + _lon_name(lon)
    path = os.path.join(cache, "skadi", name + ".hgt")
    if os.path.exists(path):
        os.utime(path)
        return path
    body = _get(_SKADI_URL.format(lat=_lat_name(lat), lon=_lon_name(lon)))
    if body is None:
        return None
    _atomic_write(path, lambda f: f.write(gzip.decompress(body)))
    return path


def _read_skadi(source, read, cell, cache, hrefs=None):
    west, south, east, north = read
    tiles = [(lat, lon) for lat in range(int(math.floor(south)), int(math.ceil(north)))
             for lon in range(int(math.floor(west)), int(math.ceil(east)))]
    pieces = []
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(_skadi_path, lat, lon, cache) for lat, lon in tiles]
        for n, fut in enumerate(as_completed(futures), 1):
            progress.report(n, len(futures), "downloading")
            path = fut.result()
            if path is None:
                continue
            with rasterio.open(path) as ds:
                a = ds.read(1, out_dtype="float32")
                if ds.nodata is not None:
                    a[a == ds.nodata] = np.nan
                pieces.append((a, ds.transform, ds.crs))
    return pieces


def _terrarium_tile(z, x, y, cache):
    path = os.path.join(cache, "terrarium", str(z), "%d_%d.npy" % (x, y))
    a = _load(path)
    if a is not None:
        return a
    body = _get(_TERRARIUM_URL.format(z=z, x=x, y=y))
    if body is None:
        a = np.full((256, 256), np.nan, np.float32)
    else:
        with MemoryFile(body) as mem, mem.open() as ds:
            rgb = ds.read([1, 2, 3]).astype(np.float32)
        a = (rgb[0] * 256.0 + rgb[1] + rgb[2] / 256.0 - 32768.0).astype(np.float32)
    _save(path, a)
    return a


#: Terrain Tiles have their bathymetry up to this zoom only (finer tiles are 0
#: at sea).
_BATHY_ZOOM = 9


def _terrarium_mosaic(read, z, cache, done, total):
    west, south, east, north = read
    n = 2 ** z
    x0, y1 = _merc_tile(west, south, z)
    x1, y0 = _merc_tile(east, north, z)
    px = 2.0 * _HALF / (256.0 * n)
    mosaic = np.full(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.nan, np.float32)
    tiles = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(_terrarium_tile, z, x, y, cache): (x, y) for x, y in tiles}
        for k, fut in enumerate(as_completed(futures), 1):
            progress.report(done + k, total, "downloading")
            x, y = futures[fut]
            mosaic[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = fut.result()
    return mosaic, Affine(px, 0.0, -_HALF + x0 * 256 * px, 0.0, -px, _HALF - y0 * 256 * px)


def _read_terrarium(source, read, cell, cache, hrefs=None):
    """Tiles of the zoom for ``cell``; above _BATHY_ZOOM, the sea (0 there)
    is filled from that zoom's bathymetry."""
    z = _zoom(source, cell, 0.5 * (read[1] + read[3]))
    total = _requests(source, read, cell) + (_terrarium_count(read, _BATHY_ZOOM) if z > _BATHY_ZOOM else 0)
    mosaic, transform = _terrarium_mosaic(read, z, cache, 0, total)
    merc = CRS.from_epsg(3857)
    if z > _BATHY_ZOOM:
        base, base_t = _terrarium_mosaic(read, _BATHY_ZOOM, cache, total - _terrarium_count(read, _BATHY_ZOOM), total)
        under = np.full(mosaic.shape, np.nan, np.float32)
        reproject(base, under, src_transform=base_t, src_crs=merc, dst_transform=transform, dst_crs=merc,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
        sea = (mosaic == 0.0) & (under < 0.0)
        mosaic[sea] = under[sea]
    return [(mosaic, transform, merc)]


def _terrarium_count(read, z):
    x0, y1 = _merc_tile(read[0], read[1], z)
    x1, y0 = _merc_tile(read[2], read[3], z)
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def _merc_tile(lon, lat, z):
    lat = max(min(lat, 85.0511), -85.0511)
    n = 2 ** z
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    r = math.radians(lat)
    y = int(math.floor((1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n))
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _service_lattice(source, read, cell):
    """(crs, step in CRS units, tile px, [(ix, iy)]) of the service tiles over
    ``read``."""
    src = SOURCES[source]
    crs = CRS.from_epsg(src["epsg"])
    step = _rung(source, cell) / (DEG if crs.is_geographic else 1.0)
    x0, y0, x1, y1 = transform_bounds(CRS.from_epsg(4326), crs, *read, densify_pts=41)
    span = src["tile"] * step
    tiles = [(ix, iy) for iy in range(int(math.floor(y0 / span)), int(math.floor(y1 / span)) + 1)
             for ix in range(int(math.floor(x0 / span)), int(math.floor(x1 / span)) + 1)]
    return crs, step, src["tile"], tiles


def _service_tile(source, step, size, ix, iy, cache):
    """One service tile (size x size, row 0 north) of the lattice, cached."""
    src = SOURCES[source]
    path = os.path.join(cache, "services", source, "%g" % (step * 1e6), "%d_%d.npy" % (ix, iy))
    a = _load(path)
    if a is not None:
        return a
    left, bottom = ix * size * step, iy * size * step
    right, top = left + size * step, bottom + size * step
    kind = src["kind"]
    if kind == "wms_bil":
        body = _get(_IGN_WMS.format(layer=src["layer"], left=left, bottom=bottom, right=right, top=top, size=size))
        if body is None or len(body) != size * size * 4:
            raise ValueError("%s: unexpected answer (%s)" % (src["label"], body[:200] if body else "none"))
        a = np.frombuffer(body, "<f4").reshape(size, size).copy()
        a[a < -1000.0] = np.nan
    else:
        if kind == "wcs":
            ax, ay = src["axes"]
            sx, sy = src["scale_axes"]
            sub = [("SUBSET", "%s(%r,%r)" % (ay, bottom, top)), ("SUBSET", "%s(%r,%r)" % (ax, left, right))] \
                if src.get("lat_first") else \
                [("SUBSET", "%s(%r,%r)" % (ax, left, right)), ("SUBSET", "%s(%r,%r)" % (ay, bottom, top))]
            query = [("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
                     ("COVERAGEID", src["coverage_id"])] + sub + \
                    [("SCALESIZE", "%s(%d),%s(%d)" % (sx, size, sy, size)), ("FORMAT", "image/tiff")]
        else:  # arcgis exportImage
            query = [("bbox", "%r,%r,%r,%r" % (left, bottom, right, top)), ("bboxSR", src["epsg"]),
                     ("imageSR", src["epsg"]), ("size", "%d,%d" % (size, size)), ("format", "tiff"),
                     ("pixelType", "F32"), ("interpolation", "RSP_BilinearInterpolation"), ("f", "image")]
        body = _get(src["url"] + "?" + urllib.parse.urlencode(query, safe="(),:"), timeout=120)
        a = np.full((size, size), np.nan, np.float32)
        if body is not None and body[:2] in (b"II", b"MM"):
            with MemoryFile(body) as mem, mem.open() as ds:
                data = ds.read(1, out_dtype="float32")
                if ds.nodata is not None:
                    data[data == np.float32(ds.nodata)] = np.nan
                data[np.abs(data) > 1e30] = np.nan
                reproject(data, a, src_transform=ds.transform, src_crs=ds.crs or CRS.from_epsg(src["epsg"]),
                          dst_transform=Affine(step, 0.0, left, 0.0, -step, top), dst_crs=CRS.from_epsg(src["epsg"]),
                          src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
        elif body is not None:
            raise ValueError("%s: %s" % (src["label"], body[:300].decode("utf-8", "replace")))
    _save(path, a.astype(np.float32))
    return a


def _read_service(source, read, cell, cache, hrefs=None):
    crs, step, size, tiles = _service_lattice(source, read, cell)
    ix0, iy0 = min(t[0] for t in tiles), min(t[1] for t in tiles)
    ix1, iy1 = max(t[0] for t in tiles), max(t[1] for t in tiles)
    nx, ny = (ix1 - ix0 + 1) * size, (iy1 - iy0 + 1) * size
    if nx * ny > MAX_MOSAIC:
        raise ValueError("%d x %d source cells: too many, take a larger cell or a smaller area" % (nx, ny))
    mosaic = np.full((ny, nx), np.nan, np.float32)
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(_service_tile, source, step, size, ix, iy, cache): (ix, iy) for ix, iy in tiles}
        for k, fut in enumerate(as_completed(futures), 1):
            progress.report(k, len(futures), "downloading")
            ix, iy = futures[fut]
            r = (iy1 - iy) * size
            c = (ix - ix0) * size
            mosaic[r:r + size, c:c + size] = fut.result()
    top = (iy1 + 1) * size * step
    return [(mosaic, Affine(step, 0.0, ix0 * size * step, 0.0, -step, top), crs)]


_READERS = {"cog": _read_cog, "skadi": _read_skadi, "terrarium": _read_terrarium,
            "wms_bil": _read_service, "wcs": _read_service, "arcgis": _read_service}


def _paste(pieces):
    """The pieces of one source pasted on one lattice (the first piece's pixel
    grid, the finest resolution): (z, transform, crs)."""
    crs = pieces[0][2]
    d = min(abs(p[1].a) for p in pieces)
    t0 = pieces[0][1]
    lefts, rights, tops, bottoms = [], [], [], []
    for a, t, _c in pieces:
        lefts.append(t.c)
        rights.append(t.c + a.shape[1] * t.a)
        tops.append(t.f)
        bottoms.append(t.f + a.shape[0] * t.e)
    x0 = t0.c + math.floor((min(lefts) - t0.c) / d) * d
    y1 = t0.f + math.ceil((max(tops) - t0.f) / d) * d
    nx = int(math.ceil((max(rights) - x0) / d))
    ny = int(math.ceil((y1 - min(bottoms)) / d))
    if nx * ny > MAX_MOSAIC:
        raise ValueError("%d x %d source cells: too many, take a larger cell or a smaller area" % (nx, ny))
    if len(pieces) == 1:
        return pieces[0]
    transform = Affine(d, 0.0, x0, 0.0, -d, y1)
    z = np.full((ny, nx), np.nan, np.float32)
    for a, t, c in pieces:
        reproject(a, z, src_transform=t, src_crs=c, dst_transform=transform, dst_crs=crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest, init_dest_nodata=False)
    return z, transform, crs


def fetch_grid(source="copernicus30", west=0.0, south=0.0, east=0.0, north=0.0, cell_size=None, epsg=None,
               crs_wkt=None, cache_dir=None, hrefs=None, manning=False):
    """The source over a lon/lat box as a GeoRaster in the target CRS (see the
    module doc). ``hrefs``: the COGs of a source that needs them (Sentinel-2
    bands); ``manning``: WorldCover classes as Manning n."""
    _check_box(source, west, south, east, north)
    src = SOURCES[source]
    cell = float(cell_size) if cell_size else src["cell"]
    crs, (ux0, uy0, ux1, uy1), n_cols, n_rows, read = _output_grid(source, west, south, east, north, cell,
                                                                     epsg, crs_wkt)
    cache = cache_dir or _CACHE_DIR
    with rasterio.Env(**_GDAL_ENV):
        pieces = _READERS[src["kind"]](source, read, cell, cache, hrefs)
    if not pieces:
        raise ValueError("no %s data in this box" % src["label"])
    z, transform, src_crs = _paste(pieces)
    read_m = abs(transform.a) * (DEG if src_crs.is_geographic else 1.0)
    coarser = cell > 1.5 * read_m
    if src.get("classes"):
        resampling = Resampling.mode if coarser else Resampling.nearest
    else:
        resampling = Resampling.average if coarser else Resampling.bilinear
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    progress.report(0, 0, "reprojecting")
    reproject(z, out, src_transform=transform, src_crs=src_crs,
              dst_transform=from_bounds(ux0, uy0, ux1, uy1, n_cols, n_rows), dst_crs=crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
    if manning and source == "worldcover":
        out = manning_of(out)
    epsg_out = crs.to_epsg()
    return GeoRaster(z=out, cell_size=cell, x_min=ux0, y_min=uy0, epsg=epsg_out,
                     **({} if epsg_out else {"crs_wkt": crs.to_wkt()}))


def manning_of(classes):
    """Manning n of WorldCover class codes (MANNING; NaN elsewhere)."""
    n = np.full(classes.shape, np.nan, np.float32)
    for code, value in MANNING.items():
        n[classes == code] = value
    return n


def _requests(source, read, cell):
    """Requests (tiles, blocks or files) an area takes, without network (an
    estimate for STAC sources)."""
    src = SOURCES[source]
    west, south, east, north = read
    kind = src["kind"]
    if kind in ("wms_bil", "wcs", "arcgis"):
        return len(_service_lattice(source, read, cell)[3])
    if kind == "terrarium":
        return _terrarium_count(read, _zoom(source, cell, 0.5 * (south + north)))
    if kind == "skadi" or source in ("copernicus30", "copernicus90", "nasadem", "fabdem"):
        return (int(math.ceil(north)) - int(math.floor(south))) * (int(math.ceil(east)) - int(math.floor(west)))
    lat = math.radians(0.5 * (south + north))
    km2 = (east - west) * DEG * math.cos(lat) * (north - south) * DEG / 1e6
    return max(1, int(math.ceil(km2 / {"swissalti3d": 1.0, "worldcover": 333.0 ** 2}.get(source, 1e4))))


def plan_download(source="copernicus30", west=0.0, south=0.0, east=0.0, north=0.0, cell_size=None, epsg=None,
                  crs_wkt=None, cache_dir=None, manning=False):
    """What ``download_dem`` will produce for these arguments, without
    downloading: source label, licence and coverage, the target CRS, grid shape
    and extent, the resolution read, requests, an estimate of the bytes read,
    and warnings."""
    _check_box(source, west, south, east, north)
    src = SOURCES[source]
    cell = float(cell_size) if cell_size else src["cell"]
    crs, (ux0, uy0, ux1, uy1), n_cols, n_rows, read = _output_grid(source, west, south, east, north, cell,
                                                                     epsg, crs_wkt)
    read_m = _read_metres(source, cell)
    lat = math.radians(0.5 * (read[1] + read[3]))
    area = (read[2] - read[0]) * DEG * max(math.cos(lat), 0.05) * (read[3] - read[1]) * DEG
    warnings = []
    cov = src.get("coverage")
    if cov is not None and (west < cov[0] or south < cov[1] or east > cov[2] or north > cov[3]):
        warnings.append("%s covers %g° to %g° E, %g° to %g° N: cells outside are empty."
                        % (src["label"], cov[0], cov[2], cov[1], cov[3]))
    epsg_out = crs.to_epsg() or 0
    utm = 32601 <= epsg_out <= 32760
    if utm and (south < _UTM_LATS[0] or north > _UTM_LATS[1]):
        warnings.append("UTM is defined for 80°S to 84°N; the grid is distorted beyond.")
    if utm and east - west > 6.0:
        warnings.append("Box wider than one UTM zone (6°): scale distortion grows away from zone %d."
                        % (epsg_out % 100))
    if src.get("nc"):
        warnings.append("Non-commercial licence.")
    return {
        "label": src["label"],
        "licence": src["licence"],
        "nc": bool(src.get("nc")),
        "coverage": list(cov) if cov else None,
        "epsg": epsg_out,
        "zone": epsg_out % 100 if utm else 0,
        "north": epsg_out < 32700 if utm else True,
        "cell_size": cell,
        "read_cell": read_m,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "width_m": ux1 - ux0,
        "height_m": uy1 - uy0,
        "tiles": _requests(source, read, cell),
        "download_bytes": area / (read_m * read_m) * src["bpp"],
        "warnings": warnings,
    }


def catalogue():
    """id -> {label, licence, nc, coverage, cell, native} of every source."""
    return {k: {"label": v["label"], "licence": v["licence"], "nc": bool(v.get("nc")),
                "coverage": list(v["coverage"]) if v.get("coverage") else None, "cell": v["cell"],
                "native": v["native"]} for k, v in SOURCES.items() if not v.get("hrefs_given")}


@process(
    id="dem_sources.download_dem",
    category="Input",
    label="Download DEM",
    params=[
        Param("source", "string", default="copernicus30",
              choices=[k for k, v in SOURCES.items() if not v.get("hrefs_given")],
              choice_labels=[v["label"] for v in SOURCES.values() if not v.get("hrefs_given")],
              doc="Where the values come from (see the module's SOURCES: world, polar and national elevation, "
                  "bathymetry, land cover)."),
        Param("west", "float", doc="Western edge, degrees longitude."),
        Param("south", "float", doc="Southern edge, degrees latitude."),
        Param("east", "float", doc="Eastern edge, degrees longitude."),
        Param("north", "float", doc="Northern edge, degrees latitude."),
        Param("cell_size", "float",
              label="Cell size (0 = the source's)", unit="m", optional=True, min=0.0,
              doc="Output cell size in metres; unset or 0: the source's default."),
        Param("epsg", "int", optional=True, label="Target EPSG",
              doc="CRS of the output grid; unset: the UTM zone of the box centre (polar stereographic for "
                  "ArcticDEM / REMA)."),
        Param("crs_wkt", "string", optional=True, label="Target CRS (WKT)",
              doc="CRS of the output grid when it has no EPSG code."),
        Param("cache_dir", "string", optional=True, label="Cache folder",
              doc="Where tiles and blocks read are kept (default ~/.cache/pytopoviz)."),
        Param("manning", "bool", default=False, label="As Manning n",
              doc="WorldCover only: the classes turned into Manning n (MANNING)."),
    ],
    outputs=[Output("grid", "georaster")],
    impl="library",
    describe=plan_download,
)
def download_dem(source="copernicus30", west=0.0, south=0.0, east=0.0, north=0.0, cell_size=None, epsg=None,
                 crs_wkt=None, cache_dir=None, manning=False):
    """Download a source over a lon/lat box into one grid of the target CRS."""
    return fetch_grid(source, west, south, east, north, cell_size, epsg, crs_wkt, cache_dir, manning=manning)
