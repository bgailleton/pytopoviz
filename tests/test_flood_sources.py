"""Discharge sources on the floods, the quadtree flood and the upstream entries
routing (GPU runs skipped without a usable CuPy GPU; no network).

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyfastflow")

import pytopoviz.adapters  # noqa: F401,E402  (registers library adapters)
from pytopoviz.adapters.pyfastflow import _source_discharge, _source_stamps  # noqa: E402
from pytopoviz.core import CONVERTERS, PROCESSES, TYPES, build_contract  # noqa: E402
from pytopoviz.georaster import GeoRaster  # noqa: E402
from pytopoviz.geovector import GeoVector  # noqa: E402


def _gpu():
    try:
        import cupy as cp

        cp.zeros(1).sum()
    except Exception:  # noqa: BLE001
        pytest.skip("no usable CuPy GPU")


def _dem(n=64, cell=5.0, nodata=False):
    """A tilted valley draining south (row 0 north), lowest at the south edge."""
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float64)
    z = 0.02 * cell * (n - rows) + 0.05 * cell * np.abs(cols - n / 2)
    if nodata:
        z[:4, :4] = np.nan
    return GeoRaster(z=z.astype(np.float32), cell_size=cell, x_min=1000.0, y_min=2000.0, epsg=2154)


def _sources(dem, rows_cols, **attrs):
    xy = np.array([[dem.x_min + (c + 0.5) * dem.cell_size, dem.y_max - (r + 0.5) * dem.cell_size]
                   for r, c in rows_cols], np.float64)
    return GeoVector(geometry="points", xy=xy, feature_attrs={k: np.asarray(v, np.float64) for k, v in attrs.items()},
                     epsg=dem.epsg)


def test_contract_has_sources_ports_and_new_processes():
    procs = {p["id"]: p for p in build_contract(TYPES, CONVERTERS, PROCESSES)["processes"]}
    for pid in ("pyfastflow.graphflood", "pyfastflow.graphflood_particle", "pyfastflow.inertial_flood",
                "pyfastflow.fusw_flood", "pyfastflow.quadtree_flood"):
        ports = {i["name"]: i for i in procs[pid]["inputs"]}
        assert ports["sources"]["type"] == ["geopoints"] and ports["sources"]["optional"], pid
    assert {o["name"] for o in procs["pyfastflow.upstream_entries"]["outputs"]} == {"sources"}
    assert procs["pyfastflow.quadtree_flood"]["runner"]["axis"] == "step"


def test_source_stamps_conserve_discharge_and_follow_the_rain():
    dem = _dem(nodata=True)
    valid = dem.valid
    # A fixed point, a bump of 12 m, a linked one, one off the grid's valid cells.
    pts = _sources(dem, [(30, 30), (40, 20), (50, 40), (1, 1)],
                   q=[2.0, 3.0, 0.0, 5.0], radius=[0.0, 12.0, 0.0, 0.0],
                   area=[0.0, 0.0, 1.0e6, 0.0], linked=[0.0, 0.0, 1.0, 0.0])
    stamps = _source_stamps(pts, dem, valid, "test")
    assert len(stamps) == 3  # the nodata one is skipped
    assert stamps[0][0].size == 1 and stamps[1][0].size > 1
    for cells, weights, *_ in stamps:
        assert np.isclose(weights.sum(), 1.0) and valid.reshape(-1)[cells].all()
    rain = 50.0 / 3.6e6  # m/s
    q = _source_discharge(stamps, rain, valid.shape)
    assert np.isclose(q.sum(), 2.0 + 3.0 + rain * 1.0e6)
    # The bump is largest at its centre cell.
    cells, weights = stamps[1][0], stamps[1][1]
    assert cells[np.argmax(weights)] == 40 * valid.shape[1] + 20
    # A rain field: the linked source takes the rain under it.
    field = np.zeros(valid.shape)
    field[50, 40] = 2.0 * rain
    assert np.isclose(_source_discharge(stamps[2:], field, valid.shape).sum(), 2.0 * rain * 1.0e6)
    with pytest.raises(ValueError, match="valid cells"):
        _source_stamps(_sources(dem, [(1, 1)], q=[1.0]), dem, valid, "test")


def test_floods_take_sources():
    _gpu()
    dem = _dem()
    pts = _sources(dem, [(10, 32)], q=[5.0], radius=[0.0])
    dry = PROCESSES.get("pyfastflow.inertial_flood")(dem=dem, duration=300.0, rainfall=0.0)
    wet = PROCESSES.get("pyfastflow.inertial_flood")(dem=dem, sources=pts, duration=300.0, rainfall=0.0)
    assert np.nanmax(dry["water_depth"].z) == 0.0
    assert np.nanmax(wet["water_depth"].z) > 0.0
    fusw = PROCESSES.get("pyfastflow.fusw_flood")(dem=dem, sources=pts, duration=120.0, rainfall=0.0)
    assert np.nanmax(fusw["water_depth"].z) > 0.0
    gf = PROCESSES.get("pyfastflow.graphflood")(dem=dem, sources=pts, rainfall=1.0, steps=50)
    base = PROCESSES.get("pyfastflow.graphflood")(dem=dem, rainfall=1.0, steps=50)
    assert np.nansum(gf["water_depth"].z) > np.nansum(base["water_depth"].z)


@pytest.mark.parametrize("scheme", ["inertial", "fusw"])
def test_quadtree_flood_runs_on_the_dem_lattice(scheme):
    _gpu()
    dem = _dem(nodata=True)
    pts = _sources(dem, [(10, 32)], q=[5.0], radius=[0.0])
    out = PROCESSES.get("pyfastflow.quadtree_flood")(
        dem=dem, sources=pts, scheme=scheme, rainfall=20.0, duration=200.0, levels=3,
        refine_every=50.0, hold_sources=True, return_slope=True)
    assert set(out) == {"water_depth", "unit_discharge", "leaf_size", "slope"}
    h = out["water_depth"]
    assert h.z.shape == dem.z.shape and h.x_min == dem.x_min and h.cell_size == dem.cell_size
    assert np.isnan(h.z[:4, :4]).all() and np.nanmax(h.z) > 0.0
    size = out["leaf_size"].z
    # Held source: its leaf is the DEM cell; some leaves are coarser.
    assert size[10, 32] == dem.cell_size and np.nanmax(size) > dem.cell_size


def test_quadtree_runner_advances_and_stops():
    _gpu()
    from pytopoviz.core import RUNNERS

    spec = PROCESSES.get("pyfastflow.quadtree_flood").spec
    p = {q.name: q.default_value for q in spec.params if q.has_default}
    p.update(levels=3, duration=100.0, chunk_seconds=0.2, refine_every=20.0)
    with RUNNERS["pyfastflow.quadtree_flood"]({"dem": _dem()}, p) as run:
        for _ in range(500):
            out = run.advance(p)
            if run.status()["stopped"]:
                break
        assert run.status()["stopped"] and "duration" in run.status()["text"]
        assert run.coordinate()["step"] > 0 and np.nanmax(out["water_depth"].z) > 0.0
    with pytest.raises(ValueError, match="needs a duration"):
        PROCESSES.get("pyfastflow.quadtree_flood")(dem=_dem(), duration=0.0)


def test_upstream_routing_gives_the_area_outside():
    _gpu()
    from pytopoviz.adapters import _upstream

    # A plane draining south; the domain a band across the south half.
    n, cell = 40, 10.0
    rows, cols = np.mgrid[0:n, 0:n]
    z = (n - rows).astype(np.float32) * 1.0
    domain = np.zeros((n, n), bool)
    domain[25:30, 5:35] = True
    drainage, _end = _upstream._route(z, domain, cell)
    entering = domain & (drainage > 0.0)
    # Every cell above the band (rows 1..24 of the band's columns, inside the
    # edge ring) drains into its top row.
    assert entering[25, 5:35].all() and not entering[26:30].any()
    assert np.isclose(drainage[25, 10], 24 * cell * cell)


def test_fusw_linked_sources_follow_the_rain_factor():
    _gpu()
    dem = _dem()
    pts = _sources(dem, [(10, 32)], q=[0.0], radius=[0.0], area=[1.0e7], linked=[1.0])
    run = PROCESSES.get("pyfastflow.fusw_flood")
    wet = run(dem=dem, sources=pts, duration=120.0, rainfall=20.0, source_series="0:0")
    dry = run(dem=dem, sources=pts, duration=120.0, rainfall=20.0, rain_series="0:0")
    assert np.nanmax(wet["water_depth"].z) > 0.0  # the source factor leaves them
    assert np.nanmax(dry["water_depth"].z) == 0.0  # the rain factor stops them


def test_quadtree_actions_levels_and_output_cell():
    _gpu()
    from pytopoviz.core import RUNNERS

    procs = {p["id"]: p for p in build_contract(TYPES, CONVERTERS, PROCESSES)["processes"]}
    actions = {a["name"]: a for a in procs["pyfastflow.quadtree_flood"]["runner"]["actions"]}
    assert set(actions) == {"refine_now", "dt_limit", "inspect"} and actions["inspect"]["point"]
    assert not actions["refine_now"]["point"]
    dem = _dem(nodata=True)
    levels = GeoRaster(z=np.full(dem.z.shape, 2.0, np.float32), cell_size=dem.cell_size, x_min=dem.x_min,
                       y_min=dem.y_min, epsg=dem.epsg)
    spec = PROCESSES.get("pyfastflow.quadtree_flood").spec
    p = {q.name: q.default_value for q in spec.params if q.has_default}
    p.update(duration=100.0, chunk_seconds=0.2, rainfall=50.0, output_cell=4 * dem.cell_size)
    pts = _sources(dem, [(10, 32)], q=[5.0], radius=[0.0])
    with RUNNERS["pyfastflow.quadtree_flood"]({"dem": dem, "sources": pts, "level_field": levels}, p) as run:
        out = run.advance(p)
        h = out["water_depth"]
        assert h.cell_size == 4 * dem.cell_size and h.z.shape == (16, 16) and h.x_min == dem.x_min
        assert np.nanmax(run._grid("leaf_size", False).z) <= 4 * dem.cell_size  # started at level 2
        limit = run.dt_limit()
        assert "dt" in limit["text"] and len(limit["at"]) == 2
        x, y = dem.x_min + 32.5 * dem.cell_size, dem.y_max - 10.5 * dem.cell_size
        cell = run.inspect(x, y)
        assert cell["text"].startswith("Cell 32, 10") and "Size:" in cell["text"]
        assert run.inspect(dem.x_min - 10.0, y)["text"].startswith("That point is off")
        again = run.refine_now()
        assert "water_depth" in again["outputs"] and again["text"]
