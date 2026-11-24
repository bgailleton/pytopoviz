"""Workflow runner nodes (advances, stop, record) and zones (repeat, sweep,
for each), with a runner kept across a repeat zone's iterations.

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import pytest

from pytopoviz.core import (
    ConverterRegistry,
    Output,
    Param,
    Port,
    ProcessRegistry,
    Runner,
    Session,
    TypeRegistry,
    ValidationError,
    Workflow,
    WorkflowError,
    process,
    run_once,
)
from pytopoviz.core import series as seriesmod
from pytopoviz.core.runner import RUNNERS
from pytopoviz.geovector import GeoVector

OPENED = []


class _Accumulate(Runner):
    """Adds ``add`` to its array at each advance; stops at ``limit``."""

    process_id = "t.accum"
    config = ("limit",)
    axis = "step"

    def __init__(self, inputs, p):
        OPENED.append(self)
        self.z = np.array(inputs["arr"], dtype=float)
        self.limit, self.steps, self.closed = p["limit"], 0, False

    def update(self, inputs):
        self.z = np.array(inputs["arr"], dtype=float)

    def advance(self, p):
        self.z = self.z + p["add"]
        self.steps += 1
        return {"out": self.z.copy()}

    def coordinate(self):
        return {"step": self.steps}

    def status(self):
        return {"text": "", "stopped": bool(self.z.max() >= self.limit)}

    def close(self):
        self.closed = True


class _Fixed(_Accumulate):
    update = Runner.update  # takes no new inputs


@pytest.fixture
def regs():
    types = TypeRegistry()
    types.register_type("bool", "bool", bool)
    types.register_type("int", "int", lambda v: isinstance(v, int) and not isinstance(v, bool))
    types.register_type("float", "float", float)
    types.register_type("arr", "field", lambda v: isinstance(v, np.ndarray))
    types.register_type("pts", "vector", lambda v: isinstance(v, GeoVector))
    convs = ConverterRegistry(types)
    procs = ProcessRegistry()

    @process(id="t.num", params=[Param("value", "float", default=0.0)],
             outputs=[Output("n", "float")], registry=procs)
    def num(value=0.0):
        return float(value)

    @process(id="t.mul", params=[Param("a", "float", default=0.0), Param("b", "float", default=1.0)],
             outputs=[Output("v", "float")], registry=procs)
    def mul(a=0.0, b=1.0):
        return float(a) * float(b)

    @process(id="t.arr", params=[Param("value", "float", default=0.0)],
             outputs=[Output("arr", "arr")], registry=procs)
    def arr(value=0.0):
        return np.full((2, 2), float(value))

    @process(id="t.shift", inputs=[Port("arr", "arr")], params=[Param("by", "float", default=0.0)],
             outputs=[Output("out", "arr")], registry=procs)
    def shift(arr, by=0.0):
        return arr + float(by)

    @process(id="t.xsum", inputs=[Port("pts", "pts")], outputs=[Output("v", "float")], registry=procs)
    def xsum(pts):
        return float(pts.xy[:, 0].sum())

    for pid, cls in (("t.accum", _Accumulate), ("t.fixed", _Fixed)):
        def body(arr, add=1.0, limit=1e9, _cls=cls):
            return run_once(_cls, {"arr": arr}, {"add": add, "limit": limit})

        process(id=pid, inputs=[Port("arr", "arr")],
                params=[Param("add", "float", default=1.0), Param("limit", "float", default=1e9)],
                outputs=[Output("out", "arr")], registry=procs)(body)
        RUNNERS[pid] = cls
    OPENED.clear()
    yield types, convs, procs
    RUNNERS.pop("t.accum", None)
    RUNNERS.pop("t.fixed", None)


def _run(regs, doc, **kw):
    types, convs, procs = regs
    wf = Workflow.from_json(doc, types=types, converters=convs, processes=procs)
    session = Session(types, convs)
    return wf, session, wf.execute(session=session, **kw)


def _frames(session, series, field):
    return [(c, session.get(h)) for c, h in seriesmod.frames(session, series, field)]


def test_runner_node_advances_and_records(regs):
    doc = {"nodes": [
        {"id": "a", "process": "t.arr", "params": {"value": 1.0}},
        {"id": "r", "process": "t.accum", "inputs": {"arr": "a.arr"}, "params": {"add": 2.0},
         "runner": {"advances": 3, "record_every": 1}}],
        "outputs": {"out": "r.out"}}
    _wf, session, res = _run(regs, doc)
    assert session.get(res.outputs["out"])[0, 0] == 7.0
    frames = _frames(session, res.series["r"], "out")
    assert [c for c, _ in frames] == [1, 2, 3]
    assert [v[0, 0] for _, v in frames] == [3.0, 5.0, 7.0]
    assert len(OPENED) == 1 and OPENED[0].closed


def test_runner_until_stopped(regs):
    doc = {"nodes": [
        {"id": "a", "process": "t.arr"},
        {"id": "r", "process": "t.accum", "inputs": {"arr": "a.arr"},
         "params": {"add": 1.0, "limit": 4.0}, "runner": {"advances": 50, "until_stopped": True}}],
        "outputs": {"out": "r.out"}}
    _wf, session, res = _run(regs, doc)
    assert session.get(res.outputs["out"])[0, 0] == 4.0


def test_repeat_state_index_and_collect(regs):
    doc = {"nodes": [
        {"id": "x0", "process": "t.num", "params": {"value": 1.0}},
        {"id": "dbl", "process": "t.mul", "params": {"a": {"$from": "z.x"}, "b": 2.0}},
        {"id": "idx", "process": "t.mul", "params": {"a": {"$from": "z.index"}, "b": 10.0}},
        {"id": "after", "process": "t.mul", "params": {"a": {"$from": "z_out.x"}, "b": 1.0}}],
        "zones": [{"id": "z", "kind": "repeat", "count": 5,
                   "state": {"x": {"type": "float", "init": "x0.n"}}, "next": {"x": "dbl.v"},
                   "collect": {"x": "dbl.v", "i": "idx.v"}, "label": "doubling"}],
        "outputs": {"out": "after.v", "last_index": "idx.v"}}
    _wf, session, res = _run(regs, doc)
    assert session.get(res.outputs["out"]) == 32.0
    assert session.get(res.outputs["last_index"]) == 40.0
    series = res.series["z"]
    assert session.item(series)["axis"] == "iteration"
    assert [v for _, v in _frames(session, series, "x")] == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert [v for _, v in _frames(session, series, "i")] == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_repeat_count_zero_gives_init(regs):
    doc = {"nodes": [
        {"id": "x0", "process": "t.num", "params": {"value": 3.0}},
        {"id": "dbl", "process": "t.mul", "params": {"a": {"$from": "z.x"}, "b": 2.0}}],
        "zones": [{"id": "z", "kind": "repeat", "count": 0,
                   "state": {"x": {"type": "float", "init": "x0.n"}}, "next": {"x": "dbl.v"}}],
        "outputs": {"out": "z_out.x"}}
    _wf, session, res = _run(regs, doc)
    assert session.get(res.outputs["out"]) == 3.0


def test_repeat_keeps_runner_and_updates_it(regs):
    # each iteration: the runner advances once (+1) from the grid the last
    # iteration left, then the grid is shifted by 10 before the next.
    doc = {"nodes": [
        {"id": "a", "process": "t.arr"},
        {"id": "r", "process": "t.accum", "inputs": {"arr": "z.g"}, "params": {"add": 1.0}},
        {"id": "s", "process": "t.shift", "inputs": {"arr": "r.out"}, "params": {"by": 10.0}}],
        "zones": [{"id": "z", "kind": "repeat", "count": 3,
                   "state": {"g": {"type": "arr", "init": "a.arr"}}, "next": {"g": "s.out"}}],
        "outputs": {"out": "z_out.g"}}
    _wf, session, res = _run(regs, doc)
    assert session.get(res.outputs["out"])[0, 0] == 33.0
    assert len(OPENED) == 1 and OPENED[0].steps == 3 and OPENED[0].closed


def test_repeat_runner_without_update_is_refused(regs):
    doc = {"nodes": [
        {"id": "a", "process": "t.arr"},
        {"id": "r", "process": "t.fixed", "inputs": {"arr": "z.g"}},
        {"id": "s", "process": "t.shift", "inputs": {"arr": "r.out"}, "params": {"by": 1.0}}],
        "zones": [{"id": "z", "kind": "repeat", "count": 2,
                   "state": {"g": {"type": "arr", "init": "a.arr"}}, "next": {"g": "s.out"}}],
        "outputs": {"out": "z_out.g"}}
    with pytest.raises(WorkflowError, match="cannot take new inputs"):
        _run(regs, doc)
    assert OPENED[0].closed


def test_repeat_runner_on_constant_input_advances(regs):
    # the runner reads a grid made before the zone, but the zone's output
    # reads it: it is in the body and advances once per iteration.
    doc = {"nodes": [
        {"id": "a", "process": "t.arr"},
        {"id": "r", "process": "t.fixed", "inputs": {"arr": "a.arr"}, "params": {"add": 2.0}}],
        "zones": [{"id": "z", "kind": "repeat", "count": 4, "collect": {"arr": "r.out"}}],
        "outputs": {"out": "r.out"}}
    _wf, session, res = _run(regs, doc)
    assert session.get(res.outputs["out"])[0, 0] == 8.0
    assert len(OPENED) == 1
    assert [v[0, 0] for _, v in _frames(session, res.series["z"], "arr")] == [2.0, 4.0, 6.0, 8.0]


def test_sweep(regs):
    doc = {"nodes": [{"id": "m", "process": "t.mul", "params": {"a": {"$from": "z.value"}, "b": 2.0}}],
           "zones": [{"id": "z", "kind": "sweep", "name": "a", "values": {"start": 1, "stop": 3, "step": 1},
                      "collect": {"v": "m.v"}}],
           "outputs": {"last": "m.v"}}
    _wf, session, res = _run(regs, doc)
    frames = _frames(session, res.series["z"], "v")
    assert frames == [(1, 2.0), (2, 4.0), (3, 6.0)]
    assert session.item(res.series["z"])["axis"] == "a"
    assert session.get(res.outputs["last"]) == 6.0


def test_sweep_runner_fresh_each_value(regs):
    doc = {"nodes": [
        {"id": "a", "process": "t.arr"},
        {"id": "r", "process": "t.accum", "inputs": {"arr": "a.arr"},
         "params": {"add": {"$from": "z.value"}}, "runner": {"advances": 2}}],
        "zones": [{"id": "z", "kind": "sweep", "values": {"list": [1.0, 5.0]}, "collect": {"arr": "r.out"}}],
        "outputs": {}}
    _wf, session, res = _run(regs, doc)
    assert [v[0, 0] for _, v in _frames(session, res.series["z"], "arr")] == [2.0, 10.0]
    assert len(OPENED) == 2 and all(r.closed for r in OPENED)


def test_foreach_vector_features_and_many_input(regs):
    pts = GeoVector("points", np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]))
    doc = {"inputs": {"pts": {"type": "pts"}},
           "nodes": [{"id": "s", "process": "t.xsum", "inputs": {"pts": "z.item"}}],
           "zones": [{"id": "z", "kind": "foreach", "items": {"$ref": "pts"}, "collect": {"x": "s.v"}}],
           "outputs": {}}
    types, convs, procs = regs
    wf = Workflow.from_json(doc, types=types, converters=convs, processes=procs)
    session = Session(types, convs)
    res = wf.execute({"pts": pts}, session=session)
    assert _frames(session, res.series["z"], "x") == [(0, 1.0), (1, 2.0), (2, 4.0)]

    doc["inputs"]["pts"]["many"] = True
    wf = Workflow.from_json(doc, types=types, converters=convs, processes=procs)
    res = wf.execute({"pts": [pts, pts.feature(2)]}, session=session)
    assert _frames(session, res.series["z"], "x") == [(0, 7.0), (1, 4.0)]


def test_nested_zones(regs):
    # outer repeat 2x over x, inner sweep adds value to x via collect only;
    # the inner sweep's series is one per outer iteration.
    doc = {"nodes": [
        {"id": "x0", "process": "t.num", "params": {"value": 1.0}},
        {"id": "dbl", "process": "t.mul", "params": {"a": {"$from": "z.x"}, "b": 2.0}},
        {"id": "m", "process": "t.mul", "params": {"a": {"$from": "w.value"}, "b": {"$from": "z.x"}}}],
        "zones": [
            {"id": "z", "kind": "repeat", "count": 2,
             "state": {"x": {"type": "float", "init": "x0.n"}}, "next": {"x": "dbl.v"}},
            {"id": "w", "kind": "sweep", "values": {"list": [1.0, 3.0]}, "collect": {"m": "m.v"}}],
        "outputs": {"out": "z_out.x"}}
    wf, session, res = _run(regs, doc)
    _deps, body, owner = wf._structure()
    assert owner["m"] == "w" and owner["w"] == "z" and owner["dbl"] == "z"
    assert session.get(res.outputs["out"]) == 4.0
    # one inner series per outer iteration
    assert _frames(session, res.series["w (iteration = 1)"], "m") == [(1.0, 1.0), (3.0, 3.0)]
    assert _frames(session, res.series["w (iteration = 2)"], "m") == [(1.0, 2.0), (3.0, 6.0)]


def test_zone_problems(regs):
    types, convs, procs = regs
    doc = {"nodes": [
        {"id": "x0", "process": "t.num"},
        {"id": "dbl", "process": "t.mul", "params": {"a": {"$from": "z.x"}}},
        {"id": "bad", "process": "t.mul", "params": {"a": {"$from": "dbl.v"}, "b": {"$from": "z_out.x"}}}],
        "zones": [{"id": "z", "kind": "repeat", "count": 2,
                   "state": {"x": {"type": "float", "init": "x0.n"}}, "next": {"x": "dbl.v"}}],
        "outputs": {}}
    with pytest.raises(ValidationError, match="read both inside the zone and its result"):
        Workflow.from_json(doc, types=types, converters=convs, processes=procs)
    doc["nodes"].pop()
    doc["zones"][0]["next"] = {}
    doc["zones"][0]["count"] = -1
    wf = Workflow.from_json(doc, types=types, converters=convs, processes=procs, check=False)
    messages = " ".join(p["message"] for p in wf.problems())
    assert "has no 'next'" in messages and "count must be" in messages
    doc["nodes"][0]["runner"] = {"advances": 2}
    wf = Workflow.from_json(doc, types=types, converters=convs, processes=procs, check=False)
    assert any("has no live runner" in p["message"] for p in wf.problems())


def test_zones_round_trip(regs):
    types, convs, procs = regs
    doc = {"version": 2, "inputs": {"pts": {"type": "pts", "many": True}},
           "nodes": [{"id": "s", "process": "t.xsum", "params": {}, "inputs": {"pts": "z.item"}}],
           "zones": [{"id": "z", "kind": "foreach", "out": "z_out", "items": {"$ref": "pts"},
                      "collect": {"x": "s.v"}}],
           "outputs": {}}
    wf = Workflow.from_json(doc, types=types, converters=convs, processes=procs)
    assert wf.to_json() == doc
