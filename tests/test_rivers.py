"""River network tests: lsdtt3 and topotoolbox network processes and the
River network routines, on topotoolbox's bigtujunga sample DEM (skipped
without either library or the cached sample).

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lsdtt3")
ttb = pytest.importorskip("topotoolbox")

import pytopoviz.adapters  # noqa: F401,E402  (registers library adapters)
import pytopoviz.routines  # noqa: F401,E402  (registers composite routines)
from pytopoviz.core import CONVERTERS, PROCESSES, TYPES, Session, Workflow, build_contract  # noqa: E402
from pytopoviz.geovector import GeoVector  # noqa: E402

THRESHOLD = 1111


@pytest.fixture(scope="module")
def dem():
    try:
        return ttb.load_dem("bigtujunga")
    except Exception as exc:  # noqa: BLE001 - offline without the cached sample
        pytest.skip(f"bigtujunga sample unavailable: {exc}")


@pytest.fixture(scope="module")
def lsdtt3_flow(dem):
    raster = CONVERTERS.convert(dem, "lsdtt3.Raster", from_type="topotoolbox.GridObject")
    filled = PROCESSES.get("lsdtt3.fill_pits")(raster=raster)["filled"]
    return filled, PROCESSES.get("lsdtt3.flow_info")(dem=filled)["flow"]


def _all_valid(lines):
    return {k: bool(np.isfinite(v).all()) for k, v in lines.vertex_attrs.items()}


def test_processes_registered_in_contract():
    ids = {p["id"] for p in build_contract(TYPES, CONVERTERS, PROCESSES)["processes"]}
    assert {"lsdtt3.channel_heads", "lsdtt3.channel_network", "lsdtt3.network_lines",
            "topotoolbox.flow_object", "topotoolbox.stream_object",
            "topotoolbox.klargestconncomps", "topotoolbox.trunk", "topotoolbox.upstreamto",
            "topotoolbox.downstreamto", "topotoolbox.stream_lines",
            "pytopoviz.river_network_lsdtt3", "pytopoviz.river_network_topotoolbox"} <= ids


def test_lsdtt3_network_lines_carry_values(lsdtt3_flow):
    filled, flow = lsdtt3_flow
    network = PROCESSES.get("lsdtt3.channel_network")(
        flow=flow, threshold_contributing_pixels=THRESHOLD)["network"]
    out = PROCESSES.get("lsdtt3.network_lines")(network=network, dem=filled, ksn=True,
                                                 return_channel_mask=True, return_junctions=True)
    lines = out["lines"]
    assert lines.geometry == "lines" and lines.n_features > 0
    assert {"stream_order", "drainage_area"} <= set(lines.feature_attrs)
    assert _all_valid(lines) == {"drainage_area": True, "flow_distance": True, "chi": True,
                                 "ksn": True}
    # Every channel cell is on a line.
    assert np.nansum(out["channel_mask"].to_numpy()) == network.n_channel_cells
    assert out["junctions"].n_features == network.n_junctions
    assert out["segments"].n_rows > 0


def test_lsdtt3_network_from_heads_and_empty_sources(lsdtt3_flow):
    filled, flow = lsdtt3_flow
    heads = PROCESSES.get("lsdtt3.channel_heads")(
        dem=filled, flow=flow, threshold_contributing_pixels=THRESHOLD)["heads"]
    assert heads.geometry == "points" and heads.n_features > 0
    network = PROCESSES.get("lsdtt3.channel_network")(
        flow=flow, sources=heads, threshold_contributing_pixels=THRESHOLD)["network"]
    assert network.n_channel_cells > 0
    empty = GeoVector("points", np.zeros((0, 2)), epsg=heads.epsg, crs_wkt=heads.crs_wkt)
    with pytest.raises(ValueError, match="no sources"):
        PROCESSES.get("lsdtt3.channel_network")(flow=flow, sources=empty)


def test_topotoolbox_stream_lines_and_filters(dem):
    flow = PROCESSES.get("topotoolbox.flow_object")(grid=dem)["flow"]
    stream = PROCESSES.get("topotoolbox.stream_object")(flow=flow, threshold=THRESHOLD)["stream"]
    lines = PROCESSES.get("topotoolbox.stream_lines")(
        stream=stream, dem=dem, flow=flow, downstream_distance=True, gradient=True,
        ksn=True, chi=True, impose=True)["lines"]
    assert lines.n_features > 0
    assert all(_all_valid(lines).values()) and len(lines.vertex_attrs) == 7

    largest = PROCESSES.get("topotoolbox.klargestconncomps")(stream=stream, k=1)["largest"]
    trunk = PROCESSES.get("topotoolbox.trunk")(stream=largest)["trunk"]
    assert trunk.stream.size < largest.stream.size < stream.stream.size

    rows, cols = stream.node_indices
    x, y = stream.transform * (cols[[500]] + 0.5, rows[[500]] + 0.5)
    point = GeoVector("points", np.column_stack([x, y]), epsg=stream.georef.to_epsg())
    up = PROCESSES.get("topotoolbox.upstreamto")(stream=stream, points=point)["upstream"]
    assert 0 < up.stream.size < stream.stream.size
    off = GeoVector("points", np.column_stack([x + 1e7, y]), epsg=stream.georef.to_epsg())
    with pytest.raises(ValueError, match="outside the grid"):
        PROCESSES.get("topotoolbox.upstreamto")(stream=stream, points=off)


@pytest.mark.parametrize("routine", ["pytopoviz.river_network_lsdtt3",
                                     "pytopoviz.river_network_topotoolbox"])
def test_river_network_routine_workflow(dem, routine):
    threshold = ({"threshold_contributing_pixels": THRESHOLD}
                 if routine.endswith("lsdtt3") else {"threshold": float(THRESHOLD)})
    doc = {
        "version": 1,
        "inputs": {},
        "nodes": [
            {"id": "load", "process": "topotoolbox.load_dem", "params": {"name": "bigtujunga"}},
            {"id": "rivers", "process": routine, "params": threshold,
             "inputs": {"dem": "load.grid"}},
        ],
        "outputs": {"lines": "rivers.lines", "flow": "rivers.flow"},
    }
    session = Session(TYPES, CONVERTERS)
    handles = Workflow.from_json(doc).run({}, session=session)
    lines = session.get(handles["lines"])
    assert isinstance(lines, GeoVector) and lines.n_features > 0
    assert "drainage_area" in lines.vertex_attrs
