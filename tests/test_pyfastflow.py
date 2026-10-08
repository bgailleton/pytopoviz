"""pyfastflow adapter tests: registration, and small GPU runs of the Perlin
surface, Salève and GOLEM processes (skipped without pyfastflow or a usable GPU).

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import pytest

import pytopoviz.adapters  # noqa: F401  (registers library adapters)
from pytopoviz.core import PROCESSES, Session, Workflow, CONVERTERS, TYPES, build_contract
from pytopoviz.georaster import GeoRaster

# The adapter skips GraphFlood when pyfastflow's GraphFlood API does not import.
needs_graphflood = pytest.mark.skipif(
    "pyfastflow.graphflood" not in PROCESSES, reason="pyfastflow GraphFlood API not available")

pytest.importorskip("pyfastflow")


def _gpu():
    try:
        import cupy as cp

        cp.zeros(1).sum()
    except Exception:  # noqa: BLE001 - any CuPy / driver failure means no GPU
        pytest.skip("no usable CuPy GPU")


def test_processes_registered_in_contract():
    ids = {p["id"] for p in build_contract(TYPES, CONVERTERS, PROCESSES)["processes"]}
    assert {"pyfastflow.perlin_surface", "pyfastflow.golem_nosed",
            "pyfastflow.saleve_steady"} <= ids


def test_perlin_surface_shape_and_slope():
    _gpu()
    out = PROCESSES.get("pyfastflow.perlin_surface")(nx=96, ny=64, dx=10.0, slope=0.01)
    grid = out["grid"]
    assert grid.z.shape == (64, 96)
    assert grid.cell_size == 10.0
    # Southward slope: the north row is higher on average.
    assert grid.z[0].mean() > grid.z[-1].mean()


def test_golem_nosed_runs_and_keeps_nodata():
    _gpu()
    rng = np.random.default_rng(0)
    z = rng.random((64, 64)) * 10.0 + np.linspace(50.0, 0.0, 64)[:, None]
    z[:5, :5] = np.nan
    dem = GeoRaster(z=z, cell_size=30.0, x_min=100.0, y_min=200.0)
    out = PROCESSES.get("pyfastflow.golem_nosed")(
        dem=dem, steps=20, return_drainage_area=True, return_erosion_rate=True)
    assert set(out) == {"grid", "drainage_area", "erosion_rate"}
    grid = out["grid"]
    assert grid.z.shape == z.shape and grid.x_min == 100.0
    assert np.isnan(grid.z[:5, :5]).all()
    assert np.isfinite(grid.z[5:, 5:]).all()
    assert not np.allclose(grid.z[5:, 5:], z[5:, 5:])
    assert np.nanmax(out["drainage_area"].z) > 30.0 * 30.0
    assert np.nanmin(out["erosion_rate"].z) >= 0.0


def test_golem_nosed_refuses_field_of_another_shape():
    _gpu()
    dem = GeoRaster(z=np.ones((32, 32)), cell_size=30.0)
    with pytest.raises(ValueError):
        PROCESSES.get("pyfastflow.golem_nosed")(
            dem=dem, steps=1, uplift_field=GeoRaster(z=np.ones((16, 16)), cell_size=30.0))


def test_perlin_into_golem_workflow():
    _gpu()
    doc = {
        "version": 1,
        "inputs": {},
        "nodes": [
            {"id": "start", "process": "pyfastflow.perlin_surface",
             "params": {"nx": 64, "ny": 64}},
            {"id": "run", "process": "pyfastflow.golem_nosed",
             "params": {"steps": 5, "outlet_edges": "south"},
             "inputs": {"dem": "start.grid"}},
        ],
        "outputs": {"result": "run.grid"},
    }
    session = Session(TYPES, CONVERTERS)
    handles = Workflow.from_json(doc).run({}, session=session)
    assert session.get(handles["result"]).z.shape == (64, 64)


def test_saleve_noise_stack_fields():
    _gpu()
    stack = ('[{"type": "ridged"}, {"type": "worley", "cells": 8, "contrast": 0.5},'
             ' {"type": "gradient", "azimuth": 90}]')
    out = PROCESSES.get("pyfastflow.saleve_steady")(
        n=64, uplift=1e-3, uplift_noise=stack, erodibility_noise="[]", return_fields=True)
    assert out["grid"].z.shape == (64, 64)
    u = out["uplift_field"].z
    # Contrasts 1 + 0.5 + 1: within 2^(±2.5) of the mean value.
    assert u.min() >= 1e-3 * 2 ** -2.5 * 0.999 and u.max() <= 1e-3 * 2 ** 2.5 * 1.001
    assert np.allclose(out["erodibility_field"].z, 1e-5)
    with pytest.raises(ValueError, match="unknown noise type"):
        PROCESSES.get("pyfastflow.saleve_steady")(n=64, uplift_noise='[{"type": "foo"}]')


def test_saleve_multiscale_erosion_post_process():
    _gpu()
    run = PROCESSES.get("pyfastflow.saleve_steady")
    base = dict(n=64, dx=50.0, uplift_noise="[]", erodibility_noise="[]", return_fields=True)
    out = run(**base, mse_stages=2)
    z = out["grid"].z
    # Stage 2 doubles the vertices over the same extent.
    assert z.shape == (128, 128)
    assert np.isfinite(z).all()
    assert out["grid"].cell_size == pytest.approx(50.0 * 63 / 127)
    assert out["uplift_field"].z.shape == (128, 128)
    with pytest.raises(ValueError):
        run(**base, mse_stages=9)


def _flood_dem(nodata=False):
    rng = np.random.default_rng(1)
    z = rng.random((48, 64)) * 0.5 + np.linspace(20.0, 0.0, 48)[:, None]
    z[20:24, 30:34] -= 3.0  # a depression
    if nodata:
        z[:4, :4] = np.nan
    return GeoRaster(z=z, cell_size=10.0, x_min=0.0, y_min=0.0)


@needs_graphflood
@pytest.mark.parametrize("solver", ["analytical", "explicit", "transient"])
def test_graphflood_solvers(solver):
    _gpu()
    out = PROCESSES.get("pyfastflow.graphflood")(
        dem=_flood_dem(nodata=True), solver=solver, steps=20, check_every=10,
        return_discharge=True, return_outflow=True, return_imbalance=True,
        return_report=True)
    assert set(out) == {"water_depth", "discharge", "outflow", "imbalance", "report"}
    h = out["water_depth"].z
    assert np.isnan(h[:4, :4]).all() and np.isfinite(h[4:, 4:]).all()
    assert np.nanmin(h) >= 0.0
    assert list(out["report"].columns["step"]) == [10.0, 20.0]
    cols = out["report"].columns
    assert {"dh_p99", "residual", "flips", "stop", "h_max"} <= set(cols)
    assert list(cols["stop"]) == [0.0, 0.0]


@needs_graphflood
def test_graphflood_stops_on_convergence():
    _gpu()
    gf = PROCESSES.get("pyfastflow.graphflood")
    dem = _flood_dem()
    # A tolerance met at the first check stops at once.
    out = gf(dem=dem, steps=500, stop="converged", check_every=10, convergence_tol=1e3,
             return_report=True)
    cols = out["report"].columns
    assert list(cols["step"]) == [10.0] and list(cols["stop"]) == [1.0]
    # An unreachable one runs to the step count.
    out = gf(dem=dem, steps=30, stop="converged", check_every=10, convergence_tol=0.0,
             convergence_window=50, return_report=True)
    assert list(out["report"].columns["step"]) == [10.0, 20.0, 30.0]
    with pytest.raises(ValueError):
        gf(dem=dem, steps=30, stop="converged", check_every=0)


@needs_graphflood
def test_graphflood_flatten_lakes_depth_is_above_input_dem_and_chains():
    _gpu()
    dem = _flood_dem()
    gf = PROCESSES.get("pyfastflow.graphflood")
    out = gf(dem=dem, steps=5, initial_water="flatten_lakes")
    # The flattened depression's water counts in the depth.
    assert out["water_depth"].z[21, 31] > 1.0
    again = gf(dem=dem, h_init=out["water_depth"], steps=5, initial_water="flatten_lakes")
    assert again["water_depth"].z[21, 31] > 1.0


@needs_graphflood
def test_graphflood_particle_runs():
    _gpu()
    out = PROCESSES.get("pyfastflow.graphflood_particle")(
        dem=_flood_dem(nodata=True), particles=20000, launches=2,
        return_active_area=True, return_report=True)
    assert set(out) == {"water_depth", "active_area", "report"}
    assert np.isnan(out["water_depth"].z[:4, :4]).all()
    assert np.nansum(out["active_area"].z) > 0
    assert list(out["report"].columns["launch"]) == [1.0, 2.0]
    assert {"staleness", "coverage", "exit", "stop"} <= set(out["report"].columns)


@pytest.mark.skipif("pyfastflow.inertial_flood" not in PROCESSES,
                    reason="pyfastflow InertialFloodProgram not available")
@pytest.mark.parametrize("topology", ["D4", "D8"])
def test_inertial_flood_runs_and_chains_exactly(topology):
    _gpu()
    dem = _flood_dem(nodata=True)
    run = PROCESSES.get("pyfastflow.inertial_flood")
    out = run(dem=dem, duration=600.0, storm_duration=300.0, topology=topology,
              report_every=200.0, return_discharge=True, return_velocity=True,
              return_max_depth=True, return_state=True, return_report=True)
    assert set(out) == {"water_depth", "discharge", "velocity", "max_depth", "state", "report"}
    assert np.isnan(out["water_depth"].z[:4, :4]).all()
    assert list(out["report"].columns["time"]) == [200.0, 400.0, 600.0]
    assert np.nanmin(out["max_depth"].z - out["water_depth"].z) >= -1e-6
    # Fixed dt: two chained halves equal one run.
    whole = run(dem=dem, duration=200.0, time_step="fixed", dt=0.5, topology=topology)
    half = run(dem=dem, duration=100.0, time_step="fixed", dt=0.5, topology=topology,
               return_state=True)
    rest = run(dem=dem, previous_state=half["state"], duration=100.0, time_step="fixed",
               dt=0.5, topology=topology)
    assert np.allclose(whole["water_depth"].z, rest["water_depth"].z, equal_nan=True)


@needs_graphflood
def test_graphflood_runner_continues_across_advances():
    _gpu()
    from pytopoviz.core import RUNNERS

    spec = PROCESSES.get("pyfastflow.graphflood").spec
    params = {p.name: p.default_value for p in spec.params if p.has_default}
    params.update(solver="explicit", dt=0.5, steps=10, check_every=5, return_report=True)
    dem = _flood_dem()
    with RUNNERS["pyfastflow.graphflood"]({"dem": dem}, params) as live:
        live.advance(params)
        out = live.advance(params)
    assert list(out["report"].columns["step"]) == [5.0, 10.0, 15.0, 20.0]
    once = PROCESSES.get("pyfastflow.graphflood")(dem=dem, solver="explicit", dt=0.5, steps=20)
    assert np.allclose(out["water_depth"].z, once["water_depth"].z, atol=1e-6, equal_nan=True)
