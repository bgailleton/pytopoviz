"""Workflow document v2: params fed by node outputs ($from), requires /
provide, ui, keep, partial runs (until), per-node status; and the
command-line runner (pytopoviz.workflow).

Author: B.G.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pytopoviz.core import (
    ConverterRegistry,
    Output,
    Param,
    ProcessRegistry,
    TypeRegistry,
    ValidationError,
    Workflow,
    WorkflowError,
    process,
    progress,
)
from pytopoviz.core import workflow as wfmod


@pytest.fixture
def regs():
    types = TypeRegistry()
    types.register_type("bool", "bool", bool)
    types.register_type("int", "int", lambda v: isinstance(v, int) and not isinstance(v, bool))
    types.register_type("float", "float", float)
    types.register_type("string", "string", str)
    convs = ConverterRegistry(types)
    procs = ProcessRegistry()
    calls = []

    @process(id="t.const", registry=procs, params=[Param("value", "float", default=1.0)],
             outputs=[Output("out", "float")])
    def const(value=1.0):
        calls.append("const")
        return float(value)

    @process(id="t.count", registry=procs, params=[Param("n", "int", default=2)],
             outputs=[Output("out", "int")])
    def count(n=2):
        calls.append("count")
        return int(n)

    @process(id="t.scale", registry=procs,
             params=[Param("x", "float", default=0.0), Param("k", "float", default=1.0, min=0.0)],
             outputs=[Output("out", "float")])
    def scale(x=0.0, k=1.0):
        calls.append("scale")
        progress.report(1, 2, "half")
        progress.report(2, 2, "half")
        return x * k

    @process(id="t.name", registry=procs, outputs=[Output("out", "string")])
    def name():
        return "a"

    @process(id="t.fail", registry=procs, params=[Param("x", "float", default=0.0)],
             outputs=[Output("out", "float")])
    def fail(x=0.0):
        raise RuntimeError("boom")

    return types, convs, procs, calls


def _wf(doc, regs):
    types, convs, procs, _ = regs
    return Workflow.from_json(doc, types, convs, procs)


CHAIN = {
    "version": 2,
    "nodes": [
        {"id": "a", "process": "t.const", "params": {"value": 3.0}},
        {"id": "n", "process": "t.count", "params": {"n": 4}},
        {"id": "s", "process": "t.scale", "params": {"x": {"$from": "a.out"}, "k": {"$from": "n.out"}}},
        {"id": "b", "process": "t.const", "params": {"value": 10.0}},
    ],
    "outputs": {"result": "s.out", "other": "b.out"},
    "ui": {"positions": {"a": [0, 0]}},
}


def test_param_from_node_output_and_int_widened(regs):
    from pytopoviz.core import Session

    types, convs, _, _ = regs
    session = Session(types, convs)
    out = _wf(CHAIN, regs).run(session=session)
    assert session.get(out["result"]) == 12.0
    assert isinstance(session.get(out["result"]), float)


def test_param_from_validation(regs):
    bad_node = json.loads(json.dumps(CHAIN))
    bad_node["nodes"][2]["params"]["x"] = {"$from": "zz.out"}
    with pytest.raises(ValidationError, match="unknown node"):
        _wf(bad_node, regs)
    bad_type = {"nodes": [{"id": "t", "process": "t.name"},
                          {"id": "s", "process": "t.scale", "params": {"x": {"$from": "t.out"}}}]}
    with pytest.raises(ValidationError, match="not accepted"):
        _wf(bad_type, regs)
    cycle = {"nodes": [{"id": "p", "process": "t.scale", "params": {"x": {"$from": "q.out"}}},
                       {"id": "q", "process": "t.scale", "params": {"x": {"$from": "p.out"}}}]}
    with pytest.raises(ValidationError, match="cycle"):
        _wf(cycle, regs)


def test_linked_param_bounds_checked_at_run(regs):
    doc = {"nodes": [{"id": "a", "process": "t.const", "params": {"value": -1.0}},
                     {"id": "s", "process": "t.scale", "params": {"k": {"$from": "a.out"}}}],
           "outputs": {"r": "s.out"}}
    with pytest.raises(WorkflowError, match="min"):
        _wf(doc, regs).run()


def test_requires_refused_until_provided(regs):
    doc = dict(CHAIN, requires=["host-x"])
    with pytest.raises(ValidationError, match="host-x"):
        _wf(doc, regs)
    wfmod.provide("host-x")
    try:
        assert _wf(doc, regs).requires == ["host-x"]
    finally:
        wfmod.PROVIDED.discard("host-x")


def test_until_runs_only_what_is_needed_and_keeps_it(regs):
    _, _, _, calls = regs
    res = _wf(CHAIN, regs).execute(until=["a"])
    assert calls == ["const"]
    assert set(res.kept) == {"a.out"}
    assert res.outputs == {}
    with pytest.raises(WorkflowError, match="unknown node"):
        _wf(CHAIN, regs).execute(until=["nope"])


def test_keep_flag(regs):
    doc = json.loads(json.dumps(CHAIN))
    doc["nodes"][1]["keep"] = True
    res = _wf(doc, regs).execute()
    assert set(res.kept) == {"n.out"}
    assert set(res.outputs) == {"result", "other"}


def test_status_events_and_relayed_progress(regs):
    events = []
    _wf(CHAIN, regs).execute(until=["s"], on_status=events.append)
    states = [(e["node"], e["state"]) for e in events]
    assert states[-4:] == [("s", "running"), ("s", "progress"), ("s", "progress"), ("s", "done")]
    assert events[-2]["done"] == 2 and events[-2]["phase"] == "half"
    # A cancel from the listener around the run stops the node.
    with pytest.raises(progress.Cancelled):
        with progress.listening(lambda *a: None, cancelled=lambda: True):
            _wf(CHAIN, regs).execute(until=["s"], on_status=events.append)
    assert events[-1]["state"] == "cancelled"
    failing = {"nodes": [{"id": "f", "process": "t.fail"}], "outputs": {"r": "f.out"}}
    events.clear()
    with pytest.raises(RuntimeError):
        _wf(failing, regs).execute(on_status=events.append)
    assert events[-1] == {"node": "f", "state": "error", "message": "boom"}


def test_to_json_round_trip_ui_and_v1(regs):
    wf = _wf(CHAIN, regs)
    doc = wf.to_json()
    assert doc["version"] == 2 and doc["ui"] == CHAIN["ui"]
    assert "ui" not in wf.to_json(ui=False)
    assert _wf(json.loads(json.dumps(doc)), regs).to_json() == doc
    v1 = {"version": 1, "nodes": [{"id": "a", "process": "t.const"}], "outputs": {"r": "a.out"}}
    assert set(_wf(v1, regs).run()) == {"r"}
    with pytest.raises(ValidationError, match="newer"):
        _wf({"version": 99, "nodes": []}, regs)


# ---- command line -----------------------------------------------------------

def test_cli_runs_a_grid_workflow(tmp_path, capsys):
    pytest.importorskip("topotoolbox")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from pytopoviz import workflow as cli

    z = np.add.outer(np.arange(20.0), np.arange(30.0)).astype(np.float32)
    z[10, 15] -= 50.0  # a pit
    src = tmp_path / "dem.tif"
    with rasterio.open(src, "w", driver="GTiff", width=30, height=20, count=1, dtype="float32",
                       transform=from_origin(0, 20, 1, 1)) as ds:
        ds.write(z, 1)
    doc = {"version": 2,
           "inputs": {"dem": {"type": "topotoolbox.GridObject", "required": True},
                      "hybrid": {"type": "bool", "default": True}},
           "nodes": [{"id": "fill", "process": "topotoolbox.fillsinks",
                      "params": {"hybrid": {"$ref": "hybrid"}}, "inputs": {"dem": {"$ref": "dem"}}}],
           "outputs": {"filled": "fill.filled"}}
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(doc))
    out = tmp_path / "filled.tif"
    assert cli.main([str(graph), "--input", f"dem={src}", "--input", "hybrid=false",
                     "--output", f"filled={out}"]) == 0
    with rasterio.open(out) as ds:
        filled = ds.read(1)
    assert filled[10, 15] > z[10, 15]
    assert "[1/1] fill" in capsys.readouterr().err


# ---- building blocks (adapters/basics.py) -----------------------------------

def _grid(z, **meta):
    from pytopoviz.georaster import GeoRaster

    return GeoRaster(np.asarray(z, dtype=float), **meta)


def test_basic_numbers():
    from pytopoviz.core import PROCESSES

    def run(pid, **kw):
        return PROCESSES.get("pytopoviz.basic." + pid)(**kw)["value"]

    assert PROCESSES.get("pytopoviz.basic.float")(value=2.5)["number"] == 2.5
    assert PROCESSES.get("pytopoviz.basic.int")(value=3)["number"] == 3
    assert run("math", a=2.0, b=3.0, op="pow") == 8.0
    assert np.isinf(run("math", a=1.0, b=0.0, op="/"))
    assert run("unary", x=-4.0, op="abs") == 4.0 and np.isnan(run("unary", x=-1.0, op="sqrt"))
    assert run("compare", a=1.0, b=2.0, op="<") is True
    assert run("switch", condition=False, a=1.0, b=2.0) == 2.0


def test_basic_grids():
    import pytopoviz.adapters  # noqa: F401
    from pytopoviz.core import PROCESSES

    def run(pid, **kw):
        out = PROCESSES.get("pytopoviz.basic." + pid)(**kw)
        return next(iter(out.values()))

    a = _grid([[1.0, 2.0], [np.nan, 4.0]], cell_size=2.0, x_min=10.0)
    b = _grid([[1.0, 1.0], [1.0, 1.0]], cell_size=2.0, x_min=10.0)
    s = run("grid_math", a=a, b=b, op="+")
    np.testing.assert_array_equal(s.z, [[2.0, 3.0], [np.nan, 5.0]])
    assert s.same_grid(a.meta())
    with pytest.raises(ValueError, match="same lattice"):
        run("grid_math", a=a, b=_grid([[1.0, 1.0], [1.0, 1.0]]), op="+")
    np.testing.assert_array_equal(run("grid_scalar_math", a=a, b=10.0, op="-", swap=True).z,
                                  [[9.0, 8.0], [np.nan, 6.0]])
    mask = run("threshold", grid=a, value=1.5, op=">")
    np.testing.assert_array_equal(mask.z, [[0.0, 1.0], [np.nan, 1.0]])
    np.testing.assert_array_equal(run("apply_mask", grid=a, mask=mask).z,
                                  [[np.nan, 2.0], [np.nan, 4.0]])
    np.testing.assert_array_equal(run("apply_mask", grid=a, mask=mask, invert=True).z,
                                  [[1.0, np.nan], [np.nan, np.nan]])
    np.testing.assert_array_equal(run("where", mask=mask, a=a, b=b).z, [[1.0, 2.0], [np.nan, 4.0]])
    np.testing.assert_array_equal(run("clamp", grid=a, lo=1.5, hi=3.0).z, [[1.5, 2.0], [np.nan, 3.0]])
    np.testing.assert_array_equal(run("rescale", grid=a, lo=0.0, hi=3.0).z, [[0.0, 1.0], [np.nan, 3.0]])
    np.testing.assert_array_equal(run("fill_nodata", grid=a, value=-1.0).z, [[1.0, 2.0], [-1.0, 4.0]])
    np.testing.assert_array_equal(run("set_nodata", grid=a, value=2.0).z,
                                  [[1.0, np.nan], [np.nan, 4.0]])
    assert run("statistic", grid=a, stat="mean") == pytest.approx(7.0 / 3.0)
    assert run("percentile", grid=a, p=100.0) == 4.0
    assert (run("count", grid=a), run("count", grid=a, what="nodata")) == (3, 1)
    assert run("mask_area", mask=mask) == 8.0
    assert run("grid_info", grid=a, what="x_max") == 14.0


def test_basic_blocks_in_a_workflow_with_library_grids():
    """A topotoolbox grid through the georaster blocks: a statistic feeds a
    threshold's param."""
    pytest.importorskip("topotoolbox")
    import pytopoviz.adapters  # noqa: F401
    from pytopoviz.core import CONVERTERS, TYPES, Session

    grid = CONVERTERS.convert(_grid(np.arange(12.0).reshape(3, 4)), "topotoolbox.GridObject",
                              "georaster")
    doc = {"version": 2,
           "inputs": {"dem": {"type": "topotoolbox.GridObject", "required": True}},
           "nodes": [
               {"id": "mean", "process": "pytopoviz.basic.statistic", "inputs": {"grid": {"$ref": "dem"}},
                "params": {"stat": "mean"}},
               {"id": "high", "process": "pytopoviz.basic.threshold", "inputs": {"grid": {"$ref": "dem"}},
                "params": {"value": {"$from": "mean.value"}, "op": ">"}},
               {"id": "area", "process": "pytopoviz.basic.mask_area", "inputs": {"mask": "high.mask"}}],
           "outputs": {"area": "area.value"}}
    session = Session(TYPES, CONVERTERS)
    out = Workflow.from_json(doc).run({"dem": grid}, session=session)
    cell = float(grid.cellsize)
    assert session.get(out["area"]) == 6 * cell ** 2  # 6 of 12 cells above the mean 5.5


def test_problems_lists_all_with_node_and_field(regs):
    types, convs, procs, _ = regs
    doc = {"version": 2, "requires": ["host-y"],
           "nodes": [{"id": "a", "process": "nope"},
                     {"id": "s", "process": "t.scale",
                      "params": {"x": {"$from": "zz.out"}, "k": -1.0, "bad": 1}}],
           "outputs": {"o": "q.out"}}
    wf = Workflow.from_json(doc, types, convs, procs, check=False)
    got = {(p["node"], p["where"], p["field"]) for p in wf.problems()}
    assert got == {(None, None, None), ("a", None, None), ("s", "param", "bad"),
                   ("s", "param", "x"), ("s", "param", "k"), (None, "output", "o")}
    with pytest.raises(ValidationError, match="host-y"):
        wf.validate()


def test_cancel_checked_between_nodes(regs):
    _, _, _, calls = regs
    with pytest.raises(progress.Cancelled):
        with progress.listening(lambda *a: None, cancelled=lambda: True):
            _wf(CHAIN, regs).execute(until=["a"])
    assert calls == []
