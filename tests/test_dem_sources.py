"""Online DEM sources without network: the plan in the target CRS, the
service ladder, the COG reader's overview choice and block cache (a COG in
GDAL memory given as hrefs), Manning n, Hub'Eau stations as GeoJSON.

Author: B.G.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
import rasterio.shutil  # noqa: E402
import rasterio.warp  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from pytopoviz.adapters import dem_sources as ds  # noqa: E402

TOPOS_PYTHON = os.path.join(os.path.dirname(__file__), "..", "..", "Topos", "python")


def test_plan_in_target_crs():
    paris = dict(west=2.30, south=48.80, east=2.40, north=48.86)
    p = ds.plan_download("copernicus30", cell_size=30.0, **paris)
    assert p["epsg"] == 32631 and p["zone"] == 31 and p["north"]
    assert p["n_cols"] * 30.0 == pytest.approx(p["width_m"]) and 6000 < p["width_m"] < 8500
    assert p["tiles"] == 1 and p["read_cell"] == 30.0 and p["download_bytes"] > 0
    l93 = ds.plan_download("ign_rgealti", cell_size=5.0, epsg=2154, **paris)
    assert l93["epsg"] == 2154 and l93["zone"] == 0 and l93["read_cell"] == 5.0
    polar = ds.plan_download("arcticdem32", west=-50.0, south=70.0, east=-49.9, north=70.05)
    assert polar["epsg"] == 3413
    assert any("Non-commercial" in w for w in ds.plan_download("fabdem", **paris)["warnings"])
    assert ds.plan_download("ign_rgealti", west=20.0, south=48.8, east=20.1, north=48.9)["warnings"]


def test_ladder_and_overviews():
    assert ds._rung("lidarhd_mnt", 0.5) == 0.5
    assert ds._rung("lidarhd_mnt", 3.0) == 2.0
    assert ds._rung("ign_rgealti", 0.2) == 5.0  # never finer than the source
    assert ds._rung("ign_rgealti", 30.0) == 25.0
    assert ds._read_metres("sentinel2_l2a", 10.0) == 10.0
    assert ds._read_metres("sentinel2_l2a", 45.0) == 40.0


def test_cog_hrefs_block_cache(tmp_path):
    # A 10 m UTM 31N COG (a ramp) with overviews, around Paris.
    href = "/vsimem/test_dem_sources_ramp.tif"
    n = 1024
    x0, y0 = 445000.0, 5412000.0
    cols = np.arange(n, dtype=np.float32)
    z = np.tile(cols, (n, 1)) * 10.0
    with rasterio.open(href, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32",
                       crs="EPSG:32631", transform=from_origin(x0, y0, 10.0, 10.0), tiled=True,
                       blockxsize=256, blockysize=256, nodata=np.nan) as dst:
        dst.write(z, 1)
        dst.build_overviews([2, 4], rasterio.enums.Resampling.average)
    try:
        lon_lat = rasterio.warp.transform_bounds("EPSG:32631", "EPSG:4326", x0 + 2000, y0 - 6000, x0 + 4000, y0 - 4000)
        args = dict(zip(("west", "south", "east", "north"), lon_lat))
        cache = str(tmp_path)
        g = ds.fetch_grid("sentinel2_l2a", cell_size=10.0, epsg=32631, cache_dir=cache, hrefs=[href], **args)
        assert g.epsg == 32631 and g.cell_size == 10.0
        # The ramp: z = 10 m per column of 10 m, so z = x - x0 at cell centres.
        xs = g.x_min + (np.arange(g.z.shape[1]) + 0.5) * 10.0
        inner = g.z[2:-2, 2:-2]
        assert np.nanmax(np.abs(inner - (xs[2:-2] - x0 - 5.0))) < 1e-3
        blocks = [f for _, _, files in os.walk(os.path.join(cache, "blocks")) for f in files]
        assert blocks and all(f.endswith(".npy") for f in blocks)
        again = ds.fetch_grid("sentinel2_l2a", cell_size=10.0, epsg=32631, cache_dir=cache, hrefs=[href], **args)
        np.testing.assert_array_equal(again.z, g.z)
        coarse = ds.fetch_grid("sentinel2_l2a", cell_size=40.0, epsg=32631, cache_dir=cache, hrefs=[href], **args)
        assert coarse.z.shape[1] * 4 == pytest.approx(g.z.shape[1], abs=4)
        assert os.path.isdir(os.path.join(cache, "blocks", ds._key(href), "1"))  # overview 4 read
    finally:
        rasterio.shutil.delete(href)


def test_manning_and_stations():
    n = ds.manning_of(np.array([[10, 80], [0, 50]], np.float32))
    assert n[0, 0] == pytest.approx(0.10) and n[0, 1] == pytest.approx(0.03) and np.isnan(n[1, 0])
    sys.path.insert(0, os.path.abspath(TOPOS_PYTHON))
    sys.path.insert(0, os.path.abspath(os.path.join(TOPOS_PYTHON, "..", "..", "parselgodot", "core", "python")))
    sys.path.insert(0, os.path.abspath(os.path.join(TOPOS_PYTHON, "..", "..", "parselgodot", "pytopoviz", "python")))
    tiles_service = pytest.importorskip("tiles_service")
    stations = [{"code_station": "A1", "libelle_station": "La Seine à Paris", "longitude_station": 2.36,
                 "latitude_station": 48.84}, {"code_station": "B2", "longitude_station": 2.0, "latitude_station": 48.0}]
    out = tiles_service.stations_geojson(stations, {"A1": {"resultat_obs": 250000.0, "date_obs": "2026-11-09T10:00:00Z"}},
                                         {"A1": (310000.0, 2100000.0)})
    a, b = out["features"]
    assert a["geometry"]["coordinates"] == [2.36, 48.84]
    assert a["properties"]["q"] == 250.0 and a["properties"]["q_mean"] == 310.0 and a["properties"]["q_max"] == 2100.0
    assert b["properties"]["q"] is None and b["properties"]["name"] == ""
