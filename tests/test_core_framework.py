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
from pytopoviz.core import (
    CONVERTERS,
    PROCESSES,
    TYPES,
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
    """topotoolbox DEM -> (auto GridObject->Raster convert) -> lsdtt3 flow -> area."""
    if not PROCESSES.has("lsdtt3.flow_info"):
        pytest.skip("lsdtt3 adapter not loaded")
    doc = {
        "version": 1,
        "nodes": [
            {"id": "load", "process": "topotoolbox.load_dem",
             "params": {"name": "bigtujunga"}},
            {"id": "fill", "process": "topotoolbox.fillsinks",
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
