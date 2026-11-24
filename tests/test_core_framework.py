"""Framework machinery tests: direct process call, headless workflow,
JSON round-trip, contract generation, and the core-import boundary.

Author: B.G.
"""

from __future__ import annotations

import ast
import json
import os

import numpy as np
import pytest

import pytopoviz.adapters  # noqa: F401  (registers library adapters)
import pytopoviz.routines  # noqa: F401  (registers composite routines)
from pytopoviz.datatable import DataTable
from pytopoviz.georaster import GeoRaster
from pytopoviz.geovector import GeoVector
from pytopoviz.core import (
    CONVERTERS,
    PROCESSES,
    TYPES,
    ConversionError,
    DataHandle,
    Session,
    Workflow,
    build_contract,
)

HERE = os.path.dirname(__file__)
CORE_DIR = os.path.join(os.path.dirname(HERE), "pytopoviz", "core")


# ---- direct process call ----------------------------------------------------

def test_process_direct_call_returns_named_outputs():
    load = PROCESSES.get("topotoolbox.load_dem")
    smooth = PROCESSES.get("topotoolbox.gaussian_smooth")

    loaded = load(name="bigtujunga")
    assert set(loaded) == {"grid"}

    out = smooth(dem=loaded["grid"], sigma=2.0)
    assert set(out) == {"smoothed"}
    smoothed = out["smoothed"]
    assert smoothed.z.shape == loaded["grid"].z.shape
    # smoothing reduces variance
    assert np.nanstd(smoothed.z) < np.nanstd(loaded["grid"].z)


# ---- headless workflow ------------------------------------------------------

WORKFLOW_DOC = {
    "version": 1,
    "inputs": {
        "dem_name": {"type": "string", "default": "bigtujunga"},
        "sigma": {"type": "float", "default": 3.0},
    },
    "nodes": [
        {
            "id": "load",
            "process": "topotoolbox.load_dem",
            "params": {"name": {"$ref": "dem_name"}},
        },
        {
            "id": "smooth",
            "process": "topotoolbox.gaussian_smooth",
            "params": {"sigma": {"$ref": "sigma"}},
            "inputs": {"dem": "load.grid"},
        },
    ],
    "outputs": {"result": "smooth.smoothed"},
}


def test_headless_workflow_runs_and_produces_handle():
    wf = Workflow.from_json(WORKFLOW_DOC)
    session = Session(TYPES, CONVERTERS)
    results = wf.run({"sigma": 2.0}, session=session)

    assert set(results) == {"result"}
    handle = results["result"]
    assert isinstance(handle, DataHandle)
    assert handle.type_id == "topotoolbox.GridObject"
    assert handle.provenance == "smooth"

    grid = session.get(handle)
    assert grid.z.ndim == 2


def test_workflow_uses_defaults_when_args_missing():
    wf = Workflow.from_json(WORKFLOW_DOC)
    results = wf.run()  # no args -> sigma default 3.0
    assert set(results) == {"result"}


# ---- JSON round-trip --------------------------------------------------------

def test_workflow_json_round_trip():
    text = json.dumps(WORKFLOW_DOC)
    doc = json.loads(text)
    wf = Workflow.from_json(doc)
    results = wf.run({"sigma": 1.5})
    assert set(results) == {"result"}


# ---- port conversion in a workflow -----------------------------------------

from pytopoviz.core import Output, Port, process  # noqa: E402


@process(
    id="test.abs_field",
    label="Absolute value field",
    inputs=[Port("field", "field2d")],
    outputs=[Output("out", "field2d")],
    impl="composite",
)
def _abs_field(field):
    return np.abs(field)


def test_workflow_converts_grid_into_field_port():
    """A GridObject output feeds a port accepting only field2d; the registered
    GridObject->field2d converter must fire automatically at binding."""
    doc = {
        "version": 1,
        "nodes": [
            {"id": "load", "process": "topotoolbox.load_dem",
             "params": {"name": "bigtujunga"}},
            {"id": "absf", "process": "test.abs_field",
             "inputs": {"field": "load.grid"}},
        ],
        "outputs": {"field": "absf.out"},
    }
    wf = Workflow.from_json(doc)
    session = Session(TYPES, CONVERTERS)
    results = wf.run(session=session)
    value = session.get(results["field"])
    assert isinstance(value, np.ndarray) and value.ndim == 2


# ---- composite routine ------------------------------------------------------

def test_composite_routine_runs_in_workflow():
    doc = {
        "version": 1,
        "inputs": {"sigma": {"type": "float", "default": 3.0}},
        "nodes": [
            {"id": "prep", "process": "pytopoviz.load_and_smooth",
             "params": {"sigma": {"$ref": "sigma"}}},
        ],
        "outputs": {"dem": "prep.smoothed"},
    }
    wf = Workflow.from_json(doc)
    session = Session(TYPES, CONVERTERS)
    results = wf.run({"sigma": 1.5}, session=session)
    handle = results["dem"]
    assert handle.type_id == "topotoolbox.GridObject"
    assert session.get(handle).z.ndim == 2
    # the routine's contract advertises itself as composite
    spec = PROCESSES.get("pytopoviz.load_and_smooth").spec
    assert spec.impl == "composite"


# ---- cross-library workflow -------------------------------------------------

def test_cross_library_workflow_grid_to_flow():
    """topotoolbox DEM -> (auto GridObject->Raster convert) -> lsdtt3 fill -> flow -> area."""
    if not PROCESSES.has("lsdtt3.flow_info"):
        pytest.skip("lsdtt3 adapter not loaded")
    doc = {
        "version": 1,
        "nodes": [
            {"id": "load", "process": "topotoolbox.load_dem",
             "params": {"name": "bigtujunga"}},
            {"id": "fill", "process": "lsdtt3.fill_pits",
             "inputs": {"dem": "load.grid"}},
            {"id": "flow", "process": "lsdtt3.flow_info",
             "inputs": {"dem": "fill.filled"}},
            {"id": "da", "process": "lsdtt3.drainage_area",
             "inputs": {"flow": "flow.flow"}},
        ],
        "outputs": {"area": "da.area"},
    }
    wf = Workflow.from_json(doc)
    session = Session(TYPES, CONVERTERS)
    results = wf.run({}, session=session)
    area = results["area"]
    assert area.type_id == "lsdtt3.Raster"
    values = np.asarray(session.get(area).to_numpy())
    assert values.ndim == 2 and np.nanmax(values) > 0


# ---- georaster hub -----------------------------------------------------------

def test_georaster_round_trips_through_library_types():

    grid = PROCESSES.get("topotoolbox.load_dem")(name="bigtujunga")["grid"]
    geo = CONVERTERS.convert(grid, "georaster")
    assert isinstance(geo, GeoRaster)
    assert geo.shape == grid.z.shape
    assert geo.cell_size == grid.cellsize
    assert geo.x_min == grid.bounds.left and geo.y_max == grid.bounds.top
    assert geo.epsg == 32611

    back = CONVERTERS.convert(geo, "topotoolbox.GridObject")
    assert back.bounds == grid.bounds
    assert back.transform == grid.transform
    np.testing.assert_array_equal(back.z, grid.z)

    if not TYPES.has("lsdtt3.Raster"):
        return
    z = geo.z.copy()
    z[0, 0] = np.nan
    geo_nan = GeoRaster(z, geo.cell_size, geo.x_min, geo.y_min, geo.epsg)
    raster = CONVERTERS.convert(geo_nan, "lsdtt3.Raster")
    m = raster.metadata
    assert (m.x_min, m.y_min, m.cell_size) == (geo.x_min, geo.y_min, geo.cell_size)
    assert m.epsg_code == 32611
    again = CONVERTERS.convert(raster, "georaster")
    assert np.isnan(again.z[0, 0])  # sentinel nodata comes back as NaN
    np.testing.assert_allclose(again.z[1:], geo.z[1:].astype(np.float32))


def test_georaster_meta_round_trip():

    geo = GeoRaster(np.arange(6.0).reshape(2, 3), 10.0, 100.0, 200.0, 32611)
    again = GeoRaster.from_meta(geo.z.ravel(), json.loads(json.dumps(geo.meta())))
    assert again.meta() == geo.meta()
    np.testing.assert_array_equal(again.z, geo.z)


# ---- vector and table hubs --------------------------------------------------

def test_geovector_geodataframe_round_trip():
    gpd = pytest.importorskip("geopandas")
    import pandas as pd
    from shapely.geometry import (
        LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
    )

    square = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
    hole = [(1, 1), (2, 1), (2, 2), (1, 1)]
    frames = {
        "geopoints": gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[Point(0, 0), MultiPoint([(1, 1), (2, 2)]), None], crs=32611),
        "geolines": gpd.GeoDataFrame(
            {"order": [1.0, np.nan, 3.0], "name": ["a", None, "c"], "main": [True, False, True]},
            geometry=[LineString([(0, 0, 5), (1, 1, 6)]),
                      MultiLineString([[(0, 0, 1), (1, 0, 1)], [(2, 2, 0), (3, 3, 0), (4, 4, 0)]]),
                      MultiLineString([[(9, 9, 9), (8, 8, 8)]])], crs=4326),
        "geopolygons": gpd.GeoDataFrame(
            {"area": [15.5, 0.5]},
            geometry=[Polygon(square, [hole]),
                      MultiPolygon([Polygon([(10, 10), (11, 10), (11, 11), (10, 10)])])],
            crs=32611),
    }
    for hub, frame in frames.items():
        vec = CONVERTERS.convert(frame, hub)
        assert TYPES.identify(vec) == hub
        back = CONVERTERS.convert(vec, "geopandas.GeoDataFrame")
        assert back.crs == frame.crs
        assert list(back.columns) == list(frame.columns)
        for a, b in zip(frame.geometry, back.geometry):
            assert (a is None and b is None) or a.wkt == b.wkt  # type, parts, holes, z
        for col in frame.columns.drop("geometry"):
            pd.testing.assert_series_equal(back[col], frame[col])

    mixed = gpd.GeoDataFrame(geometry=[Point(0, 0), LineString([(0, 0), (1, 1)])], crs=32611)
    with pytest.raises(ConversionError, match="LINESTRING"):
        CONVERTERS.convert(mixed, "geopoints")


def test_geovector_and_datatable_meta_round_trip():
    lines = GeoVector(
        "lines", [[0, 0], [1, 1], [5, 5], [6, 5], [7, 7]],
        feature_offsets=[0, 1, 2], part_offsets=[0, 2, 5],
        feature_attrs={"width": np.array([1.5, np.nan]), "n": np.array([3, 4]), "name": ["a", None]},
        vertex_attrs={"z": [1, 2, 3, 4, 5]}, epsg=32611)
    polygon = GeoVector("polygons", [[0, 0], [1, 0], [1, 1], [0, 0]], crs_wkt="LOCAL_CS[\"x\"]")
    for vec in (lines, polygon):
        meta = json.loads(json.dumps(vec.meta(), allow_nan=False))  # NaN goes as null
        again = GeoVector.from_meta(vec.to_array(), meta)
        assert again.meta() == vec.meta()
        np.testing.assert_array_equal(again.to_array(), vec.to_array())
    assert again.n_features == 1 and not again.multi[0]
    back = GeoVector.from_meta(lines.to_array(), lines.meta())
    assert back.feature_attrs["n"].dtype == np.int64 and np.isnan(back.feature_attrs["width"][1])
    with pytest.raises(ValueError):
        GeoVector("polygons", [[0, 0], [1, 0], [1, 1], [0, 1]])  # ring not closed

    table = DataTable(
        {"d": [0, 1, 2], "mean": [5, 6, 7], "p25": [4, 5, 6], "p75": [6, 7, 8], "n": [9, 9, 8]},
        units={"d": "m", "mean": "m"},
        roles={"d": "x", "p25": "band_lo", "p75": "band_hi", "n": "count"},
        bands=[("p25", "p75")], title="profile")
    again = DataTable.from_meta(table.to_array(), json.loads(json.dumps(table.meta())))
    assert again.meta() == table.meta()
    np.testing.assert_array_equal(again.to_array(), table.to_array())
    with pytest.raises(ValueError):
        DataTable({"a": [1.0], "b": [1.0, 2.0]})  # unequal lengths
    with pytest.raises(ValueError):
        DataTable({"lo": [1.0]}, roles={"lo": "band_lo"})  # band column outside a band
    with pytest.raises(ValueError):
        DataTable({"e": [1.0]}, roles={"e": "map_x"})  # map_x without map_y


def test_geovector_to_crs():
    from pyproj import Transformer

    vec = GeoVector("points", [[6.8652, 45.8326], [7.0, 46.0]], epsg=4326)
    utm = vec.to_crs(32632)
    x, y = Transformer.from_crs(4326, 32632, always_xy=True).transform(vec.xy[:, 0], vec.xy[:, 1])
    assert utm.epsg == 32632 and utm.crs_wkt == ""
    np.testing.assert_allclose(utm.xy, np.column_stack([x, y]))
    np.testing.assert_allclose(utm.to_crs(4326).xy, vec.xy, atol=1e-9)
    assert vec.to_crs("EPSG:4326") is vec


def test_flow_accumulation_routine_from_georaster_input():
    if not PROCESSES.has("lsdtt3.flow_info"):
        pytest.skip("lsdtt3 adapter not loaded")
    grid = PROCESSES.get("topotoolbox.load_dem")(name="bigtujunga")["grid"]
    session = Session(TYPES, CONVERTERS)
    dem = session.put(CONVERTERS.convert(grid, "georaster"))
    assert dem.type_id == "georaster"
    doc = {
        "version": 1,
        "inputs": {"dem": {"type": "georaster"}},
        "nodes": [{"id": "acc", "process": "pytopoviz.flow_accumulation_lsdtt3",
                   "inputs": {"dem": {"$ref": "dem"}},
                   "params": {"units": "pixels"}}],
        "outputs": {"acc": "acc.accumulation"},
    }
    out = Workflow.from_json(doc).run({"dem": session.get(dem)}, session=session)
    acc = CONVERTERS.convert(session.get(out["acc"]), "georaster")
    assert acc.shape == grid.z.shape
    assert np.nanmax(acc.z) > 1


def test_flow_accumulation_routine_stage_overrides():
    """Each stage can be supplied instead of computed; intermediates are returned."""
    if not PROCESSES.has("lsdtt3.flow_info"):
        pytest.skip("lsdtt3 adapter not loaded")
    grid = PROCESSES.get("topotoolbox.load_dem")(name="bigtujunga")["grid"]
    dem = CONVERTERS.convert(grid, "lsdtt3.Raster")
    routine = PROCESSES.get("pytopoviz.flow_accumulation_lsdtt3")

    full = routine(dem=dem)
    assert set(full) == {"accumulation", "filled", "flow"}
    ref = CONVERTERS.convert(full["accumulation"], "field2d")

    # own filled DEM: same result, filled passed through
    own = routine(dem=dem, filled_dem=full["filled"])
    np.testing.assert_allclose(CONVERTERS.convert(own["accumulation"], "field2d"), ref)
    assert own["filled"] is full["filled"]

    # precomputed flow: no filled output (optional, not produced)
    reuse = routine(dem=dem, flow_directions=full["flow"])
    assert "filled" not in reuse
    np.testing.assert_allclose(CONVERTERS.convert(reuse["accumulation"], "field2d"), ref)

    # unit weights == contributing pixels
    geo = CONVERTERS.convert(grid, "georaster")
    ones = CONVERTERS.convert(GeoRaster.from_meta(np.ones(geo.shape), geo.meta()), "lsdtt3.Raster")
    weighted = routine(dem=dem, flow_directions=full["flow"], weights=ones)
    pixels = routine(dem=dem, flow_directions=full["flow"], units="pixels")
    np.testing.assert_allclose(
        CONVERTERS.convert(weighted["accumulation"], "field2d"),
        CONVERTERS.convert(pixels["accumulation"], "field2d"),
    )


def test_optional_output_not_produced_in_workflow():
    if not PROCESSES.has("lsdtt3.flow_info"):
        pytest.skip("lsdtt3 adapter not loaded")
    grid = PROCESSES.get("topotoolbox.load_dem")(name="bigtujunga")["grid"]
    flow = PROCESSES.get("pytopoviz.flow_accumulation_lsdtt3")(
        dem=CONVERTERS.convert(grid, "lsdtt3.Raster"))["flow"]
    doc = {
        "version": 1,
        "inputs": {"dem": {"type": "topotoolbox.GridObject"},
                   "flow": {"type": "lsdtt3.FlowInfo"}},
        "nodes": [{"id": "acc", "process": "pytopoviz.flow_accumulation_lsdtt3",
                   "inputs": {"dem": {"$ref": "dem"}, "flow_directions": {"$ref": "flow"}}}],
        "outputs": {"acc": "acc.accumulation", "filled": "acc.filled"},
    }
    out = Workflow.from_json(doc).run({"dem": grid, "flow": flow})
    assert set(out) == {"acc"}


def test_bad_boundary_conditions_rejected_at_validation():
    if not PROCESSES.has("lsdtt3.flow_info"):
        pytest.skip("lsdtt3 adapter not loaded")
    from pytopoviz.core import ValidationError

    doc = {
        "version": 1,
        "inputs": {"dem": {"type": "georaster"}},
        "nodes": [{"id": "f", "process": "lsdtt3.flow_info",
                   "inputs": {"dem": {"$ref": "dem"}},
                   "params": {"boundary_conditions": "oxoo"}}],
        "outputs": {},
    }
    with pytest.raises(ValidationError):
        Workflow.from_json(doc)


# ---- swath profiles ---------------------------------------------------------

def _swath_setup():
    """bigtujunga and a north-to-south line through the centres of column 400,
    rows 100 to 300."""
    grid = PROCESSES.get("topotoolbox.load_dem")(name="bigtujunga")["grid"]
    cs, t = grid.cellsize, grid.transform
    x = t.c + 400.5 * cs
    track = GeoVector("lines", [[x, t.f - 100.5 * cs], [x, t.f - 300.5 * cs]], epsg=32611)
    return grid, track, x


def test_topotoolbox_swath_processes():
    grid, track, x = _swath_setup()
    run = lambda pid, **kw: PROCESSES.get(pid)(**kw)
    cs = grid.cellsize

    # cell-centre convention: the line's column is at distance 0, positive to
    # the left (east of a southward track); the same line in lon/lat agrees
    d = run("topotoolbox.compute_swath_distance_map", grid=grid, track=track,
            half_width=300.0)["distance_map"].z
    np.testing.assert_allclose(d[200, 399:402], [-cs, 0.0, cs])
    d_ll = run("topotoolbox.compute_swath_distance_map", grid=grid, track=track.to_crs(4326),
               half_width=300.0)["distance_map"].z
    inside = np.isfinite(d)
    np.testing.assert_allclose(d_ll[inside], d[inside], atol=1e-3)

    prepared = run("topotoolbox.prepare_track", grid=grid, track=track)["prepared_track"]
    assert prepared.n_vertices == 201 and np.allclose(prepared.xy[:, 0], x)
    assert run("topotoolbox.rasterize_path", grid=grid, track=track)["path"].n_vertices == 201
    simple = run("topotoolbox.simplify_line", grid=grid, track=prepared, tolerance=2.0)["simplified"]
    np.testing.assert_allclose(simple.xy, prepared.xy[[0, -1]])

    maps = run("topotoolbox.compute_swath_distance_map", grid=grid, track=prepared,
               half_width=300.0, return_centre_line=True)
    assert "half_width" in maps["centre_line"].vertex_attrs

    # cross-track: bands and roles; normalize shifts every statistic but std
    kw = dict(grid=grid, distance_map=maps["distance_map"], half_width=300.0,
              bin_resolution=30.0, percentiles="10, 90, 50")
    raw = run("topotoolbox.transverse_swath", **kw)["profile"]
    rel = run("topotoolbox.transverse_swath", normalize=True, **kw)["profile"]
    assert raw.roles["cross_track"] == "x" and raw.roles["p50"] == "series"
    assert raw.bands == [("min", "max"), ("q1", "q3"), ("p10", "p90")]
    assert raw.roles["std"] == "aux" and raw.roles["count"] == "count"
    ref = np.nanmedian(raw.columns["mean"] - rel.columns["mean"])
    for c in ("mean", "median", "min", "max", "q1", "q3", "p10", "p50", "p90"):
        assert np.nanmax(np.abs(raw.columns[c] - rel.columns[c] - ref)) < 1e-2, c
    np.testing.assert_allclose(rel.columns["std"], raw.columns["std"])

    # along-track: one point per prepared vertex, x/y on the track
    along = run("topotoolbox.longitudinal_swath", grid=grid, track=prepared,
                distance_map=maps["distance_map"], nearest_point=maps["nearest_point"],
                half_width=300.0, binning_distance=60.0)["profile"]
    assert along.n_rows == 201 and along.roles["x"] == "map_x" and along.roles["y"] == "map_y"
    np.testing.assert_allclose(along.columns["x"], x)
    assert along.columns["along_track"][-1] == pytest.approx(200 * cs)
    pixels = run("topotoolbox.get_point_pixels", grid=grid, track=prepared,
                 distance_map=maps["distance_map"], nearest_point=maps["nearest_point"],
                 point_index=100, half_width=300.0, binning_distance=60.0)["pixels"]
    assert pixels.geometry == "points" and pixels.n_features == along.columns["count"][100]

    windowed = run("topotoolbox.longitudinal_swath_windowed", grid=grid, track=prepared,
                   half_width=300.0, binning_distance=60.0, skip=10)["profile"]
    assert windowed.n_rows == 21
    window = run("topotoolbox.get_windowed_point_samples", grid=grid, track=prepared,
                 point_index=100, half_width=300.0, binning_distance=60.0)["pixels"]
    assert window.n_features > 0

    ptype = TYPES.get("topotoolbox.percentiles")
    assert ptype.matches("") and ptype.matches(" 5, 95 ")
    assert not ptype.matches("5, 101") and not ptype.matches("5; 95")


def test_lsdtt3_swath_processes():
    if not PROCESSES.has("lsdtt3.swath_profile"):
        pytest.skip("lsdtt3 adapter not loaded")
    grid, track, x = _swath_setup()
    raster = CONVERTERS.convert(grid, "lsdtt3.Raster")
    res = PROCESSES.get("lsdtt3.swath_profile")(
        reference=raster, baseline=track.to_crs(4326), half_width_metres=300.0,
        bin_width_metres=600.0)
    table = res["profile"]
    assert table.n_rows == 10 and table.roles["along_track"] == "x"
    assert table.bands == [("p0", "p100"), ("p25", "p75")]
    assert table.roles["p50"] == "series" and table.roles["p5"] == "aux"
    # map point of each bin centre: on the line, that far from its start
    assert table.roles["x"] == "map_x" and table.roles["y"] == "map_y"
    y0 = grid.transform.f - 100.5 * grid.cellsize
    np.testing.assert_allclose(table.columns["x"], x, atol=1e-3)
    np.testing.assert_allclose(y0 - table.columns["y"], table.columns["along_track"], atol=1e-3)
    signed = CONVERTERS.convert(res["signed_perpendicular_distance"], "field2d")
    np.testing.assert_allclose(signed[200, 399:402], [-grid.cellsize, 0.0, grid.cellsize],
                               atol=1e-6)  # the line went through lon/lat
    dist = PROCESSES.get("lsdtt3.swath_distances")(raster=raster, baseline=track)
    assert set(dist) == {"along_axis", "perpendicular", "signed_perpendicular"}


def test_swath_workflows_run():
    from pytopoviz.core import ValidationError

    grid, track, _ = _swath_setup()
    session = Session(TYPES, CONVERTERS)
    cs = grid.cellsize

    def doc(process, params, dem_type="topotoolbox.GridObject"):
        """The workflow as Topos runs it: same ports and params for both libraries."""
        return {
            "version": 1,
            "inputs": {"dem": {"type": dem_type}, "line": {"type": "geolines"}},
            "nodes": [{"id": "s", "process": process, "params": params,
                       "inputs": {"dem": {"$ref": "dem"}, "line": {"$ref": "line"}}}],
            "outputs": {"out": "s.along_track"},
        }

    # One step: the swath map is laid inside. Along-track bin 120 = +-60.
    params = {"half_width": 300.0, "along_bin": 120.0, "percentiles": "10, 90",
              "method": "windowed", "skip": 10}
    out = Workflow.from_json(doc("pytopoviz.swath_profile_topotoolbox", params)).run(
        {"dem": grid, "line": track}, session=session)
    assert session.get(out["out"]).n_rows == 21

    with pytest.raises(ValidationError):
        Workflow.from_json(doc("pytopoviz.swath_profile_topotoolbox",
                               dict(params, percentiles="10, 200")))

    run = PROCESSES.get("pytopoviz.swath_profile_topotoolbox")
    res = run(dem=grid, line=track)  # empty sizes: 100 / 10 / 1 cells
    assert set(res) == {"along_track", "cross_track", "outline", "signed_distance", "track",
                        "swath_map"}
    assert res["swath_map"].half_width == pytest.approx(100 * cs)
    cross = res["cross_track"].columns["cross_track"]
    assert np.nanmax(np.diff(cross)) == pytest.approx(cs)
    with pytest.raises(ValueError):
        run(dem=grid, line=track, half_width=0.0)

    # The corridor outline: the track widened by the half-width, round ends.
    outline = PROCESSES.get("topotoolbox.swath_map")(grid=grid, track=track, half_width=300.0)["outline"]
    assert TYPES.identify(outline) == "geopolygons" and outline.n_features == 1
    length = 200 * cs  # rows 100 to 300 of one column
    area = CONVERTERS.convert(outline, "geopandas.GeoDataFrame").area.iloc[0]
    assert area == pytest.approx(2 * 300.0 * length + np.pi * 300.0 ** 2, rel=0.01)

    if PROCESSES.has("pytopoviz.swath_profile_lsdtt3"):
        out = Workflow.from_json(doc("pytopoviz.swath_profile_lsdtt3",
                                     {"half_width": 300.0, "along_bin": 600.0})).run(
            {"dem": grid, "line": track}, session=session)
        assert session.get(out["out"]).n_rows == 10
        raster = CONVERTERS.convert(grid, "lsdtt3.Raster")
        run = PROCESSES.get("pytopoviz.swath_profile_lsdtt3")
        res = run(dem=raster, line=track.to_crs(4326))  # empty sizes: 100 / 10 cells
        assert res["outline"].epsg == 32611 and res["along_track"].n_rows == 20
        assert set(res) == {"along_track", "outline", "signed_distance", "distance",
                            "along_distance"}
        assert "outline" not in run(dem=raster, line=track, half_width=0.0)  # every cell


def test_line_sources_and_line_index(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    grid, track, x = _swath_setup()
    cs = grid.cellsize
    # From a file: only the lines are kept.
    frame = gpd.GeoDataFrame({"name": ["a", "p", "b"]}, geometry=[
        LineString(track.xy), Point(track.xy[0]), LineString(track.xy + [3 * cs, 0.0])], crs=32611)
    frame.to_file(tmp_path / "l.gpkg")
    lines = PROCESSES.get("pytopoviz.read_lines")(path=str(tmp_path / "l.gpkg"))["lines"]
    assert lines.n_features == 2 and lines.feature_attrs["name"] == ["a", "b"]
    frame.iloc[[1]].to_file(tmp_path / "p.gpkg")
    with pytest.raises(ValueError):
        PROCESSES.get("pytopoviz.read_lines")(path=str(tmp_path / "p.gpkg"))

    # Which line: the second one is 3 cells east.
    run = PROCESSES.get("pytopoviz.swath_profile_topotoolbox")
    res = run(dem=grid, line=lines, line_index=1, half_width=300.0)
    assert res["track"].xy[0, 0] == pytest.approx(x + 3 * cs)
    with pytest.raises(ValueError):
        run(dem=grid, line=lines, line_index=2)

    # Flow path from a point: starts in its cell, goes down to an outlet.
    start = GeoVector("points", [track.xy[0]], epsg=32611)
    ttb_out = PROCESSES.get("pytopoviz.flow_path_topotoolbox")(dem=grid, point=start)
    path = ttb_out["path"]
    assert path.n_features == 1 and np.allclose(path.xy[0], track.xy[0])
    z = [grid.z[int((grid.transform.f - py) / cs), int((px - grid.transform.c) / cs)]
         for px, py in path.xy]
    assert z[-1] < z[0] and "flow_directions" in ttb_out
    again = PROCESSES.get("pytopoviz.flow_path_topotoolbox")(dem=grid, point=start,
                                                             flow=ttb_out["flow_directions"])
    assert "flow_directions" not in again and np.array_equal(again["path"].xy, path.xy)
    if PROCESSES.has("pytopoviz.flow_path_lsdtt3"):
        raster = CONVERTERS.convert(grid, "lsdtt3.Raster")
        ls = PROCESSES.get("pytopoviz.flow_path_lsdtt3")(dem=raster, point=start.to_crs(4326))
        assert np.allclose(ls["path"].xy[0], track.xy[0], atol=1e-6)
        assert ls["path"].n_vertices > 10 and "flow_directions" in ls


# ---- contract ---------------------------------------------------------------

def test_contract_generates_and_is_json_serialisable():
    contract = build_contract(TYPES, CONVERTERS, PROCESSES)
    text = json.dumps(contract)  # must not raise
    again = json.loads(text)
    assert again["schema_version"]
    ids = {p["id"] for p in again["processes"]}
    assert "topotoolbox.gaussian_smooth" in ids
    # kind injected on interface members
    smooth = next(p for p in again["processes"] if p["id"] == "topotoolbox.gaussian_smooth")
    assert smooth["inputs"][0]["kind"] == "grid"
    assert smooth["params"][0]["kind"] == "float"
    # descriptions: process docstring + per-member doc
    assert "Blur" in smooth["description"]
    assert all("doc" in m for m in smooth["inputs"] + smooth["params"] + smooth["outputs"])


def test_contract_lists_vector_and_table_hubs():
    contract = json.loads(json.dumps(build_contract(TYPES, CONVERTERS, PROCESSES)))
    assert "table" in contract["kinds"]["data"]
    kinds = {t["type_id"]: t["kind"] for t in contract["types"]}
    assert [kinds[t] for t in ("geopoints", "geolines", "geopolygons")] == ["vector"] * 3
    assert kinds["datatable"] == "table"
    pairs = {(c["from"], c["to"]) for c in contract["converters"]}
    if "geopandas.GeoDataFrame" in kinds:
        assert ("geopandas.GeoDataFrame", "geolines") in pairs
        assert ("geolines", "geopandas.GeoDataFrame") in pairs


def test_contract_choice_labels_type_doc_and_describe():
    from pytopoviz.core import (
        ConverterRegistry, Output, Param, ProcessRegistry, RegistrationError, TypeRegistry,
        process,
    )

    types = TypeRegistry()
    types.register_type("float", "float", float)
    types.register_type("string", "string", str)
    types.register_type("coded", "string", str, doc="A coded string.")
    procs = ProcessRegistry()

    @process(id="t.p", registry=procs,
             params=[Param("mode", "string", default="a", choices=["a", "b"],
                           choice_labels=["Mode A", "Mode B"]),
                     Param("code", "coded", default="x"),
                     Param("size", "float", default=2.0)],
             outputs=[Output("out", "float")],
             describe=lambda mode="a", code="x", size=2.0: {"cells": size * size})
    def p(mode="a", code="x", size=2.0):
        return size

    entry = build_contract(types, ConverterRegistry(types), procs)["processes"][0]
    mode, code, _ = entry["params"]
    assert mode["choice_labels"] == ["Mode A", "Mode B"]
    assert code["doc"] == "A coded string."  # falls back to the type's doc
    assert entry["describe"] is True
    assert procs.get("t.p").describe(size=3.0) == {"cells": 9.0}

    with pytest.raises(RegistrationError):
        process(id="t.bad", registry=procs,
                params=[Param("m", "string", choices=["a", "b"], choice_labels=["A"])])(
            lambda m="a": m)



def test_contract_leaves_out_processes_with_unregistered_types():
    """A routine over a library that is not installed (its adapter skipped)
    names types nobody registered: the contract leaves it out."""
    from pytopoviz.core import ConverterRegistry, Output, Port, ProcessRegistry, TypeRegistry, process

    types = TypeRegistry()
    types.register_type("float", "float", float)
    procs = ProcessRegistry()
    process(id="t.ok", registry=procs, outputs=[Output("out", "float")])(lambda: 1.0)
    process(id="t.missing", registry=procs, inputs=[Port("dem", "absent.Raster")],
            outputs=[Output("out", "float")])(lambda dem: 1.0)

    ids = [p["id"] for p in build_contract(types, ConverterRegistry(types), procs)["processes"]]
    assert ids == ["t.ok"]

def test_contract_output_switch_label_unit_and_runner_axis():
    from pytopoviz.core import (
        ConverterRegistry, Output, Param, ProcessRegistry, RegistrationError, Runner,
        TypeRegistry, process, runner,
    )
    from pytopoviz.core.runner import RUNNERS

    types = TypeRegistry()
    types.register_type("bool", "bool", bool)
    types.register_type("float", "float", float)
    procs = ProcessRegistry()

    @process(id="t.s", registry=procs,
             params=[Param("return_extra", "bool", default=False),
                     Param("both", "bool", default=False, label="Both (0 = off)", unit="m")],
             outputs=[Output("out", "float"), Output("extra", "float", optional=True),
                      Output("other", "float", optional=True, switch="both")])
    def s(return_extra=False, both=False):
        return 1.0

    @runner("t.s", axis="step")
    class _Run(Runner):
        pass

    try:
        entry = build_contract(types, ConverterRegistry(types), procs)["processes"][0]
    finally:
        RUNNERS.pop("t.s")
    switches = {o["name"]: o.get("switch") for o in entry["outputs"]}
    assert switches == {"out": None, "extra": "return_extra", "other": "both"}
    assert entry["runner"] == {"config": [], "axis": "step", "actions": []}
    both = entry["params"][1]
    assert (both["label"], both["unit"]) == ("Both (0 = off)", "m")
    assert "label" not in entry["params"][0] and "unit" not in entry["params"][0]
    assert _Run.status(None) == {"text": "", "stopped": False}

    with pytest.raises(RegistrationError):  # a switch must be a bool param
        process(id="t.bad", registry=procs, params=[Param("x", "float", default=1.0)],
                outputs=[Output("o", "float", optional=True, switch="x")])(lambda x=1.0: x)


# ---- core import boundary ---------------------------------------------------

FORBIDDEN_IMPORTS = {
    "topotoolbox", "pyfastflow", "lsdtt3", "numpy", "scipy",
    "matplotlib", "pyvista",
}


def test_core_imports_no_science_or_viz_library():
    """core/ must not import any science/viz lib nor anything from pytopoviz
    outside core (DESIGN.md §1)."""
    offenders = []
    for fname in os.listdir(CORE_DIR):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(CORE_DIR, fname)
        with open(path, "r") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_IMPORTS:
                        offenders.append((fname, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 1:
                    offenders.append((fname, f"relative level {node.level}"))
                mod = (node.module or "").split(".")[0]
                if mod in FORBIDDEN_IMPORTS:
                    offenders.append((fname, node.module))
    assert not offenders, f"core import-boundary violations: {offenders}"


def test_georaster_onto_another_grid():
    g = GeoRaster(np.arange(16.0).reshape(4, 4) + 0.5, cell_size=10.0, epsg=32632)
    assert g.same_grid(g.meta()) and not g.same_grid(dict(g.meta(), cell_size=20.0))
    coarse = g.onto(dict(g.meta(), n_rows=2, n_cols=2, cell_size=20.0))
    assert coarse.z.tolist() == [[3.0, 5.0], [11.0, 13.0]]  # cell averages
    wider = g.onto(dict(g.meta(), n_rows=6, n_cols=6, x_min=-10.0, y_min=-10.0))
    assert np.isnan(wider.z[0]).all() and np.array_equal(wider.z[1:5, 1:5], g.z)
    labels = GeoRaster(np.array([[1.0, 2.0], [3.0, 4.0]]), cell_size=10.0)
    fine = labels.onto(dict(labels.meta(), n_rows=4, n_cols=4, cell_size=5.0))
    assert set(np.unique(fine.z)) == {1.0, 2.0, 3.0, 4.0}  # nearest: no blends


def test_georaster_read_file(tmp_path):
    """A DTM as nevertheLAS writes it: EHdr float grid, nodata, CRS beside it."""
    pytest.importorskip("rasterio")
    z = np.array([[1, 2, 3], [4, -9999, 6]], dtype=np.float32)
    z.tofile(tmp_path / "elevation.flt")
    (tmp_path / "elevation.hdr").write_text(
        "ncols 3\nnrows 2\nxllcorner 1000\nyllcorner 2000\ncellsize 0.5\n"
        "NODATA_value -9999\nbyteorder LSBFIRST\n")
    r = GeoRaster.read_file(str(tmp_path / "elevation.flt"))
    assert r.z.shape == (2, 3) and r.cell_size == 0.5
    assert (r.x_min, r.y_min) == (1000, 2000)
    assert r.z[0, 0] == 1 and r.z[1, 2] == 6 and np.isnan(r.z[1, 1])
    assert r.epsg == 0 and r.crs_wkt == ""
