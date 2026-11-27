"""pyfastflow's quadtree floods (QuadtreeInertial, QuadtreeFusw, QuadtreeFlood)
as a pytopoviz process with a live runner, ported from nevertheLAS's
quadtree_flood_service (its Run: solver, refinement schedule, time step,
sources), and the upstream entries of a flood domain (Q = P x A over IGN's
RGE ALTI, routed with pyfastflow).

The flood runs on the DEM's cells as the finest level: leaves start at
``levels`` (the cell x 2^levels; ``edge_level`` along the footprint's edge, 0
under held sources) and pyfastflow splits and merges them by the conditions.
Each advance runs ``chunk_seconds`` of wall time (0: to the end) and returns
the leaves' values on the DEM's cells: the depth as each leaf's own value (the
raster holds the flood's volume), the unit discharge and slope interpolated
from coarser leaves, as nevertheLAS's export. Elevations go to the solver
relative to the lowest cell (absolute heights in f32 would lose the depth of
shallow water).

``output_cell`` puts them on a coarser lattice instead (nevertheLAS's export
at a cell). A ``level_field`` gives each cell its starting level (nevertheLAS's
working-grid levels). Between advances the runner answers refine_now, dt_limit
and inspect (Runner.actions).

Left out of the port: the view's files (leaves, atlases) and explain_bed
(nevertheLAS's viewer).

Author: B.G.
"""

from __future__ import annotations

import math
import time

import numpy as np
from pyfastflow.flood import QuadtreeFusw, QuadtreeInertial, SteadyState
from pyfastflow.graphflood import QuadtreeFlood
from pyfastflow.quadtree import condition_names, footprint_outlets

from ..core import Output, Param, Port, Runner, process, progress, run_once, runner
from ..georaster import GeoRaster
from ..geovector import GeoVector
from .pyfastflow import (_MM_PER_H, _backend, _field, _like, _metric_cell_size, _num,
                         _fusw_series, _series_factor, _source_discharge, _source_stamps,
                         _sources_port)

WET_DEPTH = 1.0e-3
INACTIVE = 1.0e30  # The wall base of a leaf off the footprint.
GRAVITY = 9.81  # As the solvers' default.

_SCHEMES = {"inertial": QuadtreeInertial, "fusw": QuadtreeFusw, "graphflood": QuadtreeFlood}
#: Refinement conditions: (quantity, label, on by default, refine at or above).
_CONDITIONS = [("q", "|q| (m²/s)", True, 0.02), ("h", "water depth (m)", True, 0.05),
               ("speed", "speed (m/s)", False, 0.5), ("u", "velocity x (m/s)", False, 0.5),
               ("v", "velocity y (m/s)", False, 0.5)]
_REFINE = ["interval", "plan", "steady", "off"]
_REFINE_LABELS = ["Every N", "By a plan", "At steady state", "Never"]
_REGULATORS = ["s_upwind", "bates", "q_upwind", "q_centered", "s_centered"]
_REGULATOR_LABELS = ["s-upwind", "Bates", "de Almeida upwind", "de Almeida centred", "s-centred"]
_OUTLETS = ["depth", "flux", "discharge", "closed"]
_OUTLET_LABELS = ["Held at a set depth", "Edge flux", "Discharge drawn", "Closed"]
_FUSW_OUTLETS = ["normal_depth", "critical", "neumann", "sink"]
_FUSW_OUTLET_LABELS = ["Normal depth", "Critical", "Free (zero gradient)", "Sink"]


def _pairs(text, name, who, positive=1.0e-9):
    """Sorted (x, y) of "x:y, x:y, ..." (y at least ``positive``)."""
    out = []
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            x, y = (float(v) for v in item.split(":"))
        except ValueError:
            raise ValueError(f"{who}: {name}: {item!r} is not x:value") from None
        out.append((x, max(y, positive)))
    if not out:
        raise ValueError(f"{who}: {name} needs a point")
    return sorted(out)


def curve_value(points, x, ramp):
    """A curve of sorted ``(x, y)`` at ``x``: y holds from a point to the next, or with
    ``ramp`` moves to it on a log scale. Before the first point, the first y."""
    value = points[0][1]
    for index, (at, y) in enumerate(points):
        if x < at:
            break
        value = y
        if ramp and index + 1 < len(points):
            next_at, next_y = points[index + 1]
            if x < next_at and next_at > at:
                value = math.exp(math.log(y) + (math.log(next_y) - math.log(y)) * (x - at) / (next_at - at))
    return value


def _condition_params():
    out = []
    for name, label, on, above in _CONDITIONS:
        out += [
            Param(f"refine_on_{name}", "bool", label=f"Refine on {label}", default=on,
                  doc=f"Split a leaf whose {label} is at or above the value."),
            Param(f"{name}_above", "float", label=f"{label}: at or above", default=above, min=0.0,
                  doc="The value a leaf splits at."),
            Param(f"{name}_merge_below", "float", label=f"{label}: merge below", default=0.0, min=0.0,
                  doc="Siblings all below it merge (0: never). Merging takes a value on every "
                      "condition on; keep it under the refine value, or leaves split and "
                      "merge in turn."),
        ]
    return out


def to_lattice(source, x_offset, y_offset, size, shape, cell, interpolate=True):
    """``source``, cells of ``size`` from the south, on the ``shape`` cells of
    ``cell`` (nevertheLAS's to_lattice). The offsets go from the lattice's
    south-west corner to the source's. Coarser than the source: the mean by
    area of what a cell covers; finer: bilinear from the cell under its
    centre, none if that one has none, or that cell's own value without
    ``interpolate``."""
    rows, columns = shape
    ratio = cell / size
    if ratio >= 1.0 - 1.0e-9:
        valid = ~np.isnan(source)
        num, den = np.where(valid, source, 0.0).astype(np.float64), valid.astype(np.float64)
        for axis, count, offset in ((1, columns, x_offset), (0, rows, y_offset)):
            start = np.arange(source.shape[axis], dtype=np.float64) + offset / size
            first = np.floor(start / ratio + 1.0e-9).astype(np.int64)
            first_share = np.minimum(1.0, (first + 1) * ratio - start)
            pieces = []
            for values in (num, den):
                moved = np.moveaxis(values, axis, 0)
                out = np.zeros((count,) + moved.shape[1:])
                for index, share in ((first, first_share), (first + 1, 1.0 - first_share)):
                    keep = (index >= 0) & (index < count) & (share > 0.0)
                    np.add.at(out, index[keep], moved[keep] * share[keep].reshape((-1,) + (1,) * (moved.ndim - 1)))
                pieces.append(np.moveaxis(out, 0, axis))
            num, den = pieces
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0.0, num / den, np.nan).astype(np.float32)
    height, width = source.shape
    cx = (np.arange(columns) + 0.5) * ratio - 0.5
    cy = (np.arange(rows) + 0.5) * ratio - 0.5
    nx, ny = np.rint(cx).astype(np.int64).clip(0, width - 1), np.rint(cy).astype(np.int64).clip(0, height - 1)
    if not interpolate:
        return source[np.ix_(ny, nx)].astype(np.float32)
    x0, y0 = np.floor(cx).astype(np.int64), np.floor(cy).astype(np.int64)
    fx, fy = np.clip(cx - x0, 0.0, 1.0)[None, :], np.clip(cy - y0, 0.0, 1.0)[:, None]
    x1, y1 = (x0 + 1).clip(0, width - 1), (y0 + 1).clip(0, height - 1)
    x0, y0 = x0.clip(0, width - 1), y0.clip(0, height - 1)
    total, weight = np.zeros(shape), np.zeros(shape)
    for ys, xs, share in ((y0, x0, (1 - fx) * (1 - fy)), (y0, x1, fx * (1 - fy)), (y1, x0, (1 - fx) * fy),
                          (y1, x1, fx * fy)):
        values = source[np.ix_(ys, xs)]
        known = ~np.isnan(values)
        total += np.where(known, values, 0.0) * share
        weight += known * share
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(weight > 0.0, total / weight, np.nan)
    out[np.isnan(source[np.ix_(ny, nx)])] = np.nan
    return out.astype(np.float32)


@process(
    id="pyfastflow.quadtree_flood",
    category="Flood",
    label="Quadtree flood (adaptive leaves)",
    inputs=[
        Port("dem", "georaster", label="Topography",
             doc="Bed elevation (projected, metres): its cells are the finest leaves; "
                 "nodata cells are off the footprint, their edge an outlet."),
        _sources_port(),
        Port("precipitation_field", "georaster", label="Rainfall field", unit="mm/h", optional=True,
             doc="Rain per cell (mm/h) added to Rainfall, times Field scale; same grid as the "
                 "DEM. Not held at the finest level."),
        Port("source_field", "georaster", label="Source field", unit="m³/s per cell", optional=True,
             doc="Water added to each cell (m³/s), times Field scale; same grid as the DEM."),
        Port("level_field", "georaster", label="Starting levels", optional=True,
             doc="Starting level of each cell (rounded; the leaf side is the cell x 2^level), in "
                 "place of Levels; a cell without one takes the coarsest found. Same grid as the DEM."),
    ],
    params=[
        Param("scheme", "string", default="inertial", choices=list(_SCHEMES),
              choice_labels=["Local inertia", "FullSWOF", "GraphFlood (steady)"],
              doc="inertial: local-inertia shallow water on the leaves; fusw: FullSWOF "
                  "finite volumes; graphflood: the steady flood of the rain, refined "
                  "each time it is steady."),
        Param("rainfall", "float", unit="mm/h", default=20.0, min=0.0,
              doc="Rain rate (mm/h), on top of the rainfall field; times the rain series "
                  "factor (Inertial and FullSWOF). GraphFlood and refining at steady state "
                  "need some rain. A change applies from the next advance."),
        Param("rain_series", "string", label="Rain factor over time", default="",
              doc="Rain factor over model time, \"t:factor, t:factor, ...\" (s); empty: 1. "
                  "Before the first time and after the last, the end factors hold. It "
                  "scales the rain, the rainfall field and the linked sources (Inertial and "
                  "FullSWOF; GraphFlood is steady)."),
        Param("series_interp", "string", label="Between series times", default="step",
              choices=["step", "linear"],
              choice_labels=["Hold each factor until the next time", "Linear"],
              doc="step: a factor holds from its time on; linear: interpolated, updated "
                  "every chunk of steps."),
        Param("duration", "float", unit="s", default=0.0, min=0.0,
              doc="Model time to simulate (s; 0: until stopped, or until steady refining "
                  "ends). Inertial and FullSWOF."),
        Param("chunk_seconds", "float", label="Wall time per advance", unit="s", default=2.0, min=0.0,
              doc="Wall-clock seconds each advance runs before returning the leaves "
                  "(0: to the end)."),
        # Leaves.
        Param("levels", "int", label="Start level", default=5, min=0, max=14,
              doc="Leaves start at the DEM cell x 2^level."),
        Param("edge_level", "int", label="Level along the edges", default=-1, min=-1, max=14,
              doc="Leaves along the footprint's edge start at most at this level (-1: as "
                  "the others)."),
        Param("capacity", "int", label="Most leaves", default=5_000_000, min=1000,
              doc="Sets the memory the flood takes: about 0.1 GB per million, more once "
                  "used. Past it, the leaves furthest over their conditions split first "
                  "and the others wait."),
        Param("bed_rule", "string", label="Leaf bed", default="percentile",
              choices=["percentile", "mean"], choice_labels=["Percentile of its cells", "Mean of its cells"],
              doc="The bed of a leaf larger than a cell."),
        Param("bed_percentile", "float", label="Percentile", default=5.0, min=0.0, max=100.0,
              doc="Percentile of its cells' beds a leaf takes (Leaf bed: percentile)."),
        *_condition_params(),
        Param("refine_neighbours", "bool", label="Also refine the neighbours", default=True,
              doc="One leaf out per refinement around a leaf meeting a condition."),
        # Schedule.
        Param("refine", "string", label="Refine", default="interval", choices=_REFINE,
              choice_labels=_REFINE_LABELS,
              doc="When the leaves are refined (inertial, FullSWOF): every N, by a plan of "
                  "N over a measure of the run, each time the flood is steady, or never. "
                  "GraphFlood refines each time it is steady."),
        Param("refine_every", "float", label="Every N", default=60.0, min=0.1,
              doc="N between refinements, in Refine unit."),
        Param("refine_unit", "string", label="Refine unit", default="time", choices=["time", "steps"],
              choice_labels=["Simulated seconds", "Model steps"], doc="What N counts."),
        Param("plan_points", "string", label="Plan", default="0:300, 1200:60, 3000:10",
              doc="By a plan: \"x:N, x:N, ...\", N from each x on (Plan on)."),
        Param("plan_x", "string", label="Plan on", default="time", choices=["time", "refinements", "leaves"],
              choice_labels=["Simulated time (s)", "Refinements done", "Leaves"],
              doc="What the plan's x are placed on."),
        Param("plan_ramp", "bool", label="Ramp the plan", default=False,
              doc="N moves between checkpoints on a log scale rather than holding."),
        Param("hold", "bool", label="Hold while the wet area changes", default=False,
              doc="A refinement due waits while the wet area changes, up to the longest hold."),
        Param("hold_steps", "float", label="Wet area measured over", unit="steps", default=1000.0, min=1.0,
              doc="Model steps the change of the wet area is measured over."),
        Param("hold_change", "float", label="Held while it changes by more than", unit="%", default=1.0,
              min=0.0, max=100.0, doc="Change of the wet area over those steps."),
        Param("hold_longest", "float", label="Longest hold", unit="x N", default=5.0, min=1.0,
              doc="Past it, the refinement comes anyway."),
        Param("balance", "float", label="Storage change within", unit="% of rain", default=1.0, min=0.0,
              max=100.0, doc="At steady state: storage change against the rain."),
        Param("depth_change", "float", label="Depth change under", unit="mm/min", default=0.1, min=0.0,
              doc="At steady state: the depth change."),
        Param("held", "float", label="Held for", unit="s", default=300.0, min=0.0,
              doc="At steady state: simulated time both hold before a level is steady."),
        Param("maximum_per_level", "float", label="Maximum per level", unit="s", default=21600.0, min=1.0,
              doc="At steady state: simulated time a level may take."),
        # Time step.
        Param("time_step", "string", default="cfl", choices=["cfl", "fixed", "curve"],
              choice_labels=["CFL", "Fixed", "Curve"],
              doc="Inertial and FullSWOF: the CFL step, a fixed one, or one from a curve "
                  "(a fixed step changed along it). Whether it follows the CFL is set when "
                  "the flood starts."),
        Param("cfl", "float", label="CFL (inertial)", default=0.7, min=0.01, max=1.0, doc="Inertial CFL."),
        Param("fusw_cfl", "float", label="CFL (FullSWOF)", default=0.4, min=0.01, max=1.0, doc="FullSWOF CFL."),
        Param("dt", "float", label="Time step", unit="s", default=0.1, min=1e-4, doc="Fixed time step (s)."),
        Param("dt_max", "float", label="Largest time step", unit="s", default=0.0, min=0.0,
              doc="Cap of the CFL step (s; 0: CFL x cell)."),
        Param("dt_curve", "string", label="Time step curve", default="0:0.1, 1200:0.05, 3000:0.02",
              doc="Curve: \"x:dt, x:dt, ...\" (dt from each x on)."),
        Param("dt_curve_x", "string", label="Curve on", default="time", choices=["time", "refinements", "leaves"],
              choice_labels=["Simulated time (s)", "Refinements done", "Leaves"],
              doc="What the curve's x are placed on."),
        Param("dt_curve_ramp", "bool", label="Ramp the curve", default=False,
              doc="dt moves between points on a log scale rather than holding."),
        # Sources.
        Param("hold_sources", "bool", label="Sources at the finest level", default=False,
              doc="The leaves under a discharge source stay at the DEM cell."),
        Param("field_scale", "float", label="Field scale", default=1.0,
              doc="Factor of the rainfall and source fields."),
        # Physics.
        Param("manning", "float", label="Manning n", default=0.033, min=0.001, max=1.0,
              doc="Manning roughness n (s/m^(1/3))."),
        Param("precision", "string", default="f32", choices=["f32", "f64"],
              choice_labels=["32 bit", "64 bit"], doc="Inertial and FullSWOF float precision."),
        Param("regulator", "string", default="s_upwind", choices=_REGULATORS,
              choice_labels=_REGULATOR_LABELS, doc="Inertial: the flux regulator."),
        Param("regulator_theta", "float", label="de Almeida θ", default=0.9, min=0.0, max=1.0,
              doc="Inertial: θ of the de Almeida regulators."),
        Param("froude_limit", "float", default=1.0, min=0.01, max=10.0, doc="Inertial: Froude limit."),
        Param("transfer_fraction", "float", default=0.2, min=0.01, max=1.0,
              doc="Inertial: share of a leaf's water one side may give in a step."),
        Param("min_flow_depth", "float", unit="m", default=1e-4, min=0.0, max=1.0,
              doc="Inertial: shallower links carry nothing."),
        Param("outlet", "string", default="depth", choices=_OUTLETS, choice_labels=_OUTLET_LABELS,
              doc="Inertial: the outlets held at a depth, given an edge flux, drawing a "
                  "discharge, or closed."),
        Param("outlet_value", "float", label="Outlet depth, flux or discharge", default=0.0, min=0.0,
              doc="Inertial: the value of Outlet."),
        Param("order", "int", default=2, choices=[2, 1], choice_labels=["2: MUSCL + Heun", "1"],
              doc="FullSWOF: scheme order."),
        Param("limiter", "string", default="minmod", choices=["minmod", "vanalbada", "vanleer"],
              choice_labels=["minmod", "van Albada", "van Leer"], doc="FullSWOF: slope limiter."),
        Param("flux", "string", default="hll", choices=["hll", "hll2", "hllc", "hllc2", "rusanov"],
              choice_labels=["HLL", "HLL2", "HLLC", "HLLC2", "Rusanov"], doc="FullSWOF: Riemann solver."),
        Param("friction", "string", default="manning",
              choices=["manning", "darcy_weisbach", "laminar", "none"],
              choice_labels=["Manning", "Darcy-Weisbach", "Laminar", "None"], doc="FullSWOF: friction law."),
        Param("max_restarts", "int", label="Most restarts of a step", default=2, min=0, max=20,
              doc="FullSWOF, CFL step: restarts of a step that went unstable."),
        Param("fusw_outlet", "string", label="Outlet (FullSWOF)", default="normal_depth",
              choices=_FUSW_OUTLETS, choice_labels=_FUSW_OUTLET_LABELS, doc="FullSWOF: outlet condition."),
        Param("outlet_slope", "float", default=0.01, min=0.0, max=10.0,
              doc="FullSWOF normal depth: outlet slope (0: the bed's)."),
        Param("outlet_manning", "float", label="Outlet Manning n", default=0.033, min=0.001, max=1.0,
              doc="FullSWOF normal depth: outlet Manning n."),
        Param("relaxation", "float", default=0.1, min=0.001, max=1.0, doc="GraphFlood: relaxation."),
        Param("gf_solver", "string", label="GraphFlood solve", default="bottom_up",
              choices=["bottom_up", "local"], choice_labels=["Bottom-up", "Local"], doc="GraphFlood: solver."),
        Param("local_minima", "string", default="carve_cordonnier",
              choices=["carve_cordonnier", "fill_cordonnier", "rank_cordonnier"],
              choice_labels=["Carve", "Fill", "Rank"], doc="GraphFlood: depression routing."),
        Param("tol", "float", label="Steady below", unit="% of water left", default=2.0, min=0.0, max=100.0,
              doc="GraphFlood: refines once this share of the water is left to move."),
        Param("iterations_per_check", "int", label="Iterations between checks", default=20, min=1,
              doc="GraphFlood."),
        Param("window", "int", label="Steady without a new minimum over", unit="checks", default=10, min=1,
              doc="GraphFlood."),
        Param("eps", "float", label="A new minimum beats the last by", default=0.01, min=0.0, max=1.0,
              doc="GraphFlood."),
        Param("fast_cycles", "int", label="First refinements without waiting", default=0, min=0, max=100,
              doc="GraphFlood."),
        Param("fast_relaxation", "float", label="Relaxation of the first refinements", default=0.2,
              min=0.001, max=1.0, doc="GraphFlood."),
        Param("fast_iterations", "int", label="Iterations between the first refinements", default=5, min=1,
              doc="GraphFlood."),
        Param("return_unit_discharge", "bool", label="Also unit discharge", default=True,
              doc="Also return |q| (m²/s)."),
        Param("return_leaf_size", "bool", label="Also leaf size", default=True,
              doc="Also return the leaf side (m) over its cells."),
        Param("return_slope", "bool", label="Also leaf slope", default=False,
              doc="Also return the slope across each leaf's links."),
        Param("output_cell", "float", label="Output cell", unit="m", default=0.0, min=0.0,
              doc="Cell of the outputs' lattice (from the DEM's south-west corner): 0, the DEM's "
                  "cells; coarser, the mean by area of the leaves (the depth from the leaves of "
                  "that size, each its own value); finer, bilinear (nevertheLAS's export)."),
    ],
    outputs=[
        Output("water_depth", "georaster",
               doc="Water depth (m): each leaf's own on its cells (the flood's volume); NaN off the footprint."),
        Output("unit_discharge", "georaster", optional=True,
               doc="|q| (m²/s), coarse leaves interpolated."),
        Output("leaf_size", "georaster", optional=True, doc="Side (m) of the leaf over each cell."),
        Output("slope", "georaster", optional=True, doc="Slope across each leaf's links, interpolated."),
    ],
    impl="library",
)
def quadtree_flood(dem, sources=None, precipitation_field=None, source_field=None, level_field=None, **params):
    """Adaptive quadtree flood (pyfastflow, GPU, CuPy) on ``dem``, run to its
    end: ``duration`` for an inertial or FullSWOF flood refined by interval or
    plan (or never), the last level for one refined at steady state or a
    GraphFlood."""
    from ..core import PROCESSES

    spec = PROCESSES.get("pyfastflow.quadtree_flood").spec
    args = {p.name: p.default_value for p in spec.params if p.has_default}
    unknown = set(params) - set(args)
    if unknown:
        raise TypeError(f"quadtree_flood: unknown params {sorted(unknown)}")
    args.update(params)
    args["chunk_seconds"] = 0.0
    inputs = {"dem": dem, "sources": sources, "precipitation_field": precipitation_field,
              "level_field": level_field,
              "source_field": source_field}
    return run_once(QuadtreeFloodRun, inputs, args)


def _settings(p):
    """nevertheLAS's run settings of the params."""
    criterion, threshold, below = [], [], []
    for name, *_ in _CONDITIONS:
        if p[f"refine_on_{name}"]:
            criterion.append(name)
            threshold.append(float(p[f"{name}_above"]))
            below.append(float(p[f"{name}_merge_below"]))
    if not criterion:
        raise ValueError("Quadtree flood: refine on at least one condition")
    merges = bool(below) and 0.0 not in below
    if p["refine"] == "plan":
        plan = {"x": p["plan_x"], "unit": p["refine_unit"], "ramp": bool(p["plan_ramp"]),
                "points": _pairs(p["plan_points"], "Plan", "Quadtree flood", 1.0e-3)}
    else:
        plan = {"x": "time", "unit": p["refine_unit"], "ramp": False, "points": [[0.0, float(p["refine_every"])]]}
    fusw = p["scheme"] == "fusw"
    time_spec = {"mode": p["time_step"], "cfl": float(p["fusw_cfl"] if fusw else p["cfl"]),
                 "dt": float(p["dt"]), "dt_max": float(p["dt_max"])}
    if p["time_step"] == "curve":
        time_spec["curve"] = {"points": _pairs(p["dt_curve"], "Time step curve", "Quadtree flood"),
                              "x": p["dt_curve_x"], "ramp": bool(p["dt_curve_ramp"])}
    return {
        "criterion": criterion, "threshold": threshold, "coarsen_below": below if merges else None,
        "refine_neighbours": bool(p["refine_neighbours"]),
        "refine": "interval" if p["refine"] == "plan" else p["refine"], "plan": plan,
        "duration": float(p["duration"]), "balance": float(p["balance"]) / 100.0,
        "depth_change": float(p["depth_change"]) / 1000.0, "held": float(p["held"]),
        "maximum": float(p["maximum_per_level"]), "time": time_spec,
        "hold": {"on": bool(p["hold"]), "steps": float(p["hold_steps"]),
                 "change": float(p["hold_change"]) / 100.0, "longest": float(p["hold_longest"])},
        "hold_sources": bool(p["hold_sources"]), "field_scale": float(p["field_scale"]),
        "options": {"tol": float(p["tol"]) / 100.0},
    }


def _options(p):
    """The solver's constructor options of the scheme (nevertheLAS's ``options``)."""
    scheme = p["scheme"]
    if scheme == "inertial":
        outlet = p["outlet"]
        return {"regulator": p["regulator"], "regulator_theta": float(p["regulator_theta"]),
                "manning": float(p["manning"]), "froude_limit": float(p["froude_limit"]),
                "transfer_fraction": float(p["transfer_fraction"]),
                "min_flow_depth": float(p["min_flow_depth"]), "precision": p["precision"],
                "outlet_depth": float(p["outlet_value"]) if outlet == "depth" else None,
                "boundary_flux": float(p["outlet_value"]) if outlet == "flux" else 0.0,
                "outlet_discharge": float(p["outlet_value"]) if outlet == "discharge" else 0.0}
    if scheme == "fusw":
        return {"manning": float(p["manning"]), "order": int(p["order"]), "limiter": p["limiter"],
                "flux": p["flux"], "friction": p["friction"], "precision": p["precision"],
                "max_restarts": int(p["max_restarts"]), "outlet_type": p["fusw_outlet"],
                "outlet_slope": float(p["outlet_slope"]), "outlet_manning": float(p["outlet_manning"])}
    return {"manning": float(p["manning"]), "relaxation": float(p["relaxation"]), "solver": p["gf_solver"],
            "local_minima": p["local_minima"], "iterations_per_check": int(p["iterations_per_check"]),
            "tol": float(p["tol"]) / 100.0, "window": int(p["window"]), "eps": float(p["eps"]),
            "fast_cycles": int(p["fast_cycles"]), "fast_relaxation": float(p["fast_relaxation"]),
            "fast_iterations": int(p["fast_iterations"])}


@runner("pyfastflow.quadtree_flood", axis="step", config=(
    "scheme", "levels", "edge_level", "capacity", "bed_rule", "bed_percentile",
    "time_step", "manning", "precision", "regulator", "regulator_theta", "froude_limit",
    "transfer_fraction", "min_flow_depth", "outlet", "outlet_value", "order", "limiter", "flux",
    "friction", "max_restarts", "fusw_outlet", "outlet_slope", "outlet_manning", "relaxation",
    "gf_solver", "local_minima", "iterations_per_check", "window", "eps", "fast_cycles",
    "fast_relaxation", "fast_iterations"), actions={
    "refine_now": "Refine now", "dt_limit": "What sets the time step", "inspect": "Inspect a point"})
class QuadtreeFloodRun(Runner):
    """Live quadtree flood: one solver across advances (nevertheLAS's Run).
    The refinement rule, schedule, time step values, duration and source
    options apply from the next advance; the rest is fixed at opening."""

    who = "Quadtree flood"

    def __init__(self, inputs, p):
        who = self.who
        scheme = p["scheme"]
        if scheme not in _SCHEMES:
            raise ValueError(f"{who}: unknown scheme {scheme!r}")
        self.dem = dem_grid = inputs["dem"]
        cell = _metric_cell_size(dem_grid, who)
        dem = np.array(dem_grid.z, np.float64 if p["precision"] == "f64" else np.float32)
        valid = dem_grid.valid & np.isfinite(dem)
        if not valid.any():
            raise ValueError(f"{who}: the DEM has no valid cell")
        dem[~valid] = np.nan
        self.last_params = p
        self.cell, self.x_min, self.y_min = cell, float(dem_grid.x_min), float(dem_grid.y_min)
        self.stamps = _source_stamps(inputs.get("sources"), dem_grid, valid, who)
        shape = dem.shape
        # m³/s per cell of the rainfall field (times the rain factor) and of the source field.
        self._rain_cells = self._src_cells = None
        for port, unit, attr in (("precipitation_field", _MM_PER_H * cell * cell, "_rain_cells"),
                                 ("source_field", 1.0, "_src_cells")):
            if inputs.get(port) is not None:
                value = np.where(valid, np.maximum(_field(inputs[port], 0.0, shape, port, who), 0.0) * unit, 0.0)
                setattr(self, attr, value)
        self.series = _fusw_series(p["rain_series"], "rain", who)
        self.interp = p["series_interp"]
        if self.interp not in ("step", "linear"):
            raise ValueError(f"{who}: unknown series_interp {self.interp!r}")
        graph = scheme == "graphflood"
        self.factor = 1.0 if graph else _series_factor(self.series, self.interp, 0.0)
        rain = float(p["rainfall"]) * _MM_PER_H * self.factor
        if graph and rain <= 0.0 and self._rain_cells is None:
            raise ValueError(f"{who}: GraphFlood needs rain: it solves the steady flood of a rain rate.")
        levels = np.full(shape, int(p["levels"]), np.int32)
        if inputs.get("level_field") is not None:
            given = _field(inputs["level_field"], np.nan, shape, "level_field", who)
            known = np.isfinite(given)
            if not known.any():
                raise ValueError(f"{who}: the level field has no value")
            given = np.rint(np.where(known, given, 0.0)).clip(0).astype(np.int32)
            levels = np.where(known, given, int(given[known].max())).astype(np.int32)
        options = _options(p)
        options["bed_percentile"] = float(p["bed_percentile"]) if p["bed_rule"] == "percentile" else None
        self._open(dem, valid, scheme, rain, _settings(p), options, levels, int(p["edge_level"]), int(p["capacity"]))

    def _open(self, dem, valid, scheme, rain, settings, options, levels, edge_level=-1, capacity=5_000_000):
        """Makes the solver and the run's state (nevertheLAS's Run): ``dem``, rows
        from the north, NaN off ``valid``; ``rain`` in m/s; ``settings`` as
        :meth:`configure` takes them; ``options``, the solver's own; ``levels``,
        the starting level of each cell. Wants ``cell``, ``x_min`` and ``y_min``
        set, and what :meth:`_source_arrays` reads."""
        dem = np.array(dem)
        self.rows, self.columns = dem.shape
        self.lowest = float(dem[valid].min())
        dem -= self.lowest
        self.graph = scheme == "graphflood"
        self.scheme = scheme
        self.bed = dem
        self._valid = valid
        self.rain = rain
        self._settings_now = None
        self._source_spec = None
        options = dict(options)
        time_step = settings.get("time", {})
        if not self.graph:
            # Whether the step comes from the CFL is set for the life of the flood.
            options.update(time_step="cfl" if time_step.get("mode", "cfl") == "cfl" else "fixed")
            options.update({key: float(time_step[key]) for key in ("cfl", "dt", "dt_max") if key in time_step})
        self.time_curve, self.time_step = None, "" if self.graph else options["time_step"]
        edges = footprint_outlets(~valid)
        levels = np.array(levels, np.int32)
        top = int(levels.max())
        # The leaves under a source start at the finest cell.
        held = self._source_arrays(settings)[1]
        if held is not None:
            levels[held] = 0
        # A leaf takes the finest level of its cells.
        if edge_level >= 0:
            levels[edges] = np.minimum(levels[edges], edge_level)
        if not valid.all():
            options["outlet_mask"] = edges
        self.flood = _SCHEMES[scheme](
            _backend(), np.where(valid, dem, np.nan), self.cell, rain, levels=levels, max_level=top,
            criterion=settings.get("criterion", "h"), threshold=settings.get("threshold", 0.05),
            coarsen_below=settings.get("coarsen_below"),
            refine_neighbours=bool(settings.get("refine_neighbours", True)),
            capacity=int(capacity), **options)
        try:
            self.top = self.flood.tree.levels
            self.highest = float(np.nanmax(dem))
            self.ticks, self.tick_seconds, self.seconds = 0, 0.0, 0.0
            self.chunk = 20
            self.finished, self.error = False, ""
            self.levels_done = []
            self.level_start = 0.0
            self.level_began = time.perf_counter()
            self.outcome = ""
            self.left = float("inf")
            self.iterations = 0
            self.maximum_discharge = 0.0
            self.maximum_depth, self.wet, self.split = 0.0, 0, 0
            self.duration, self.plan, self.last_refine, self.changes = 0.0, None, 0.0, 0
            self.hold, self.hold_mark, self.hold_change, self.forced = None, None, None, 0
            self.mode, self.steady = "graphflood" if self.graph else None, None
            self.leaves_by_size = []
            self.configure(settings)
            self._tree_changed()
        except Exception:
            self.flood.close()
            raise

    # ---- settings (nevertheLAS's configure) ---------------------------------

    def configure(self, settings):
        """What can change during a run: the refinement rule, the schedule, the duration.

        Clears a stop for lack of capacity: the new rule may ask for less. A
        GraphFlood takes its new steady rule from its next level.
        """
        flood = self.flood
        self._settings_now = settings
        self._apply_sources(settings)
        flood.threshold = settings.get("threshold", flood.threshold)
        flood.criterion = settings.get("criterion", flood.criterion)
        flood.coarsen_below = settings.get("coarsen_below", flood.coarsen_below)
        flood.refine_neighbours = bool(settings.get("refine_neighbours", flood.refine_neighbours))
        flood.stop_reason = None
        if self.graph:
            options = settings.get("options", {})
            flood.tol = float(options.get("tol", flood.tol))
            return
        self._set_time(settings.get("time"))
        now = flood.time
        self.duration = float(settings.get("duration", self.duration))
        if "plan" in settings:
            self._set_plan(settings["plan"])
        elif self.plan is None:
            self._set_plan({"points": [[0.0, 60.0]]})
        if "hold" in settings:
            hold = settings["hold"]
            self.hold = {key: float(hold[key]) for key in ("steps", "change", "longest")} if hold and hold.get("on") else None
            self.hold_mark, self.hold_change = None, None
        mode = settings.get("refine", self.mode or "interval")
        if mode == "steady" and flood.rain_rate <= 0.0:
            raise ValueError("Refining at steady state needs rain: without it, the balance with the outflow cannot be tested.")
        limits = {key: float(settings[key]) for key in ("balance", "depth_change", "held", "maximum") if key in settings}
        if mode == "steady" and self.mode != "steady":
            self.steady = SteadyState(**limits)
            self.steady.update(flood)  # The level's clock starts now.
            self.level_start, self.level_began = now, time.perf_counter()
        elif mode == "steady":
            for key, value in limits.items():
                setattr(self.steady, key, value)
        else:
            self.steady = None
        self.mode = mode

    def _source_arrays(self, settings):
        """``(m3/s per cell, cells held at the finest level)``, ``None`` for what there is not:
        the sources (a linked one rain x area) and the fields, times their scale."""
        raster, held = None, None
        shape = (self.rows, self.columns)
        scale = float(settings.get("field_scale", 1.0))
        rain_cells = None if self._rain_cells is None else self._rain_cells * self.factor * scale
        if self.stamps:
            # A linked source takes the rain at its cells: the rate and the field.
            rain = self.rain if rain_cells is None else self.rain + rain_cells / (self.cell * self.cell)
            raster = _source_discharge(self.stamps, rain, shape).astype(np.float32)
            if settings.get("hold_sources", False):
                held = np.zeros(self.rows * self.columns, bool)
                for cells, *_ in self.stamps:
                    held[cells] = True
                held = held.reshape(shape)
        fields = [f for f in (rain_cells, None if self._src_cells is None else self._src_cells * scale)
                  if f is not None]
        if fields:
            scaled = sum(fields).astype(np.float32)
            raster = scaled if raster is None else raster + scaled
        return raster, None if held is None or not held.any() else held

    def _apply_sources(self, settings):
        """Takes what changed of the source options; they apply from now on."""
        spec = (bool(settings.get("hold_sources", False)), float(settings.get("field_scale", 1.0)),
                self.factor, self.rain)
        if spec == self._source_spec or (not self.stamps and self._rain_cells is None and self._src_cells is None):
            return
        self._source_spec = spec
        raster, held = self._source_arrays(settings)
        self.flood.set_source(raster, held)

    def _follow_rain(self):
        """Rate (``rainfall`` of the last params) and series factor at the flood's
        time: the solver's rain and the sources follow them from now on."""
        p = self.last_params
        factor = 1.0 if self.graph else _series_factor(self.series, self.interp, self.flood.time)
        rain = float(p["rainfall"]) * _MM_PER_H * factor
        if (rain, factor) == (self.rain, self.factor):
            return
        if self.graph and rain <= 0.0 and self._rain_cells is None:
            raise ValueError(f"{self.who}: GraphFlood needs rain: it solves the steady flood of a rain rate.")
        self.rain, self.factor = rain, factor
        self.flood.set_rain(rain)
        if self._settings_now is not None:
            self._apply_sources(self._settings_now)

    def _set_time(self, spec):
        """The values of the time step. ``mode``: ``cfl``, ``fixed`` or ``curve``, which takes
        dt from ``curve``. A flood made on the CFL stays on it, and one made on a fixed step
        stays on one."""
        if not spec:
            return
        self.time_curve = None
        if self.time_step == "cfl":
            self.flood.set_time_step(cfl=spec.get("cfl"), dt_max=spec.get("dt_max"))
        elif spec.get("mode") == "curve":
            curve = spec["curve"]
            points = sorted((float(x), max(float(dt), 1.0e-9)) for x, dt in curve["points"])
            self.time_curve = {"points": points, "x": curve.get("x", "time"), "ramp": bool(curve.get("ramp", False))}
            self._follow_time_curve()
        elif "dt" in spec:
            self.flood.set_time_step(dt=spec["dt"])

    def _measure(self, measure):
        return self.flood.time if measure == "time" else float(self.changes) if measure == "refinements" else float(self.flood.tree.n)

    def _follow_time_curve(self):
        if self.time_curve is not None:
            curve = self.time_curve
            self.flood.set_time_step(dt=curve_value(curve["points"], self._measure(curve["x"]), curve["ramp"]))

    def _set_plan(self, plan):
        """The refinement schedule: every N, N following checkpoints of a measure of the run."""
        points = sorted((float(x), max(float(n), 1.0e-3)) for x, n in plan["points"])
        if not points:
            raise ValueError("A refinement plan needs a checkpoint")
        unit = plan.get("unit", "time")
        if self.plan is not None and self.plan["unit"] != unit:
            self.last_refine = self.flood.time if unit == "time" else self.flood.steps
        self.plan = {"points": points, "x": plan.get("x", "time"), "unit": unit, "ramp": bool(plan.get("ramp", False))}

    def _plan_x(self):
        return self._measure(self.plan["x"])

    def _every(self):
        """N now, in the plan's unit."""
        every = curve_value(self.plan["points"], self._plan_x(), self.plan["ramp"])
        return max(1.0, round(every)) if self.plan["unit"] == "steps" else every

    def _position(self):
        """Where the run is in the plan's unit."""
        return self.flood.steps if self.plan is not None and self.plan["unit"] == "steps" else self.flood.time

    # ---- advance -----------------------------------------------------------

    def advance(self, p):
        """Runs about ``chunk_seconds`` of wall time (0: to the end), unless the
        end comes first; returns the leaves on the DEM's cells."""
        self.last_params = p
        self.configure(_settings(p))
        self._follow_rain()
        seconds = float(p["chunk_seconds"])
        if seconds <= 0.0:
            if not self.graph and self.mode in ("interval", "off") and self.duration <= 0.0:
                raise ValueError(f"{self.who}: running to the end needs a duration (or refining at steady state)")
            seconds = math.inf
        began = time.perf_counter()
        self._run_for(seconds)
        self._values()
        self.seconds += time.perf_counter() - began
        if self.error:
            raise ValueError(f"{self.who}: {self.error}")
        return self._outputs(p)

    def _run_for(self, seconds):
        """Runs about ``seconds`` of wall time (inf: to the end), unless the end comes first."""
        began = time.perf_counter()
        flood = self.flood
        while self.graph and not self.finished and time.perf_counter() - began < seconds:
            self._advance_graph()
            progress.report(0, 0, f"GraphFlood, level {len(self.levels_done) + 1}")
        while not self.graph and not self.finished and time.perf_counter() - began < seconds:
            chunk_began = time.perf_counter()
            self._follow_time_curve()
            self._follow_rain()
            steps = self.chunk
            target = self._next_stop()
            if target is not None and flood.dt > 0.0:
                # Half of what is left: dt can grow within a chunk.
                steps = max(1, min(steps, math.ceil(0.5 * (target - flood.time) / flood.dt)))
            if self.mode == "interval" and flood.stop_reason is None:
                steps = max(1, min(steps, self._steps_to_stop()))
            flood.run(steps)
            now = flood.time  # Reading it waits for the device.
            self._track_wet()
            taken = time.perf_counter() - chunk_began
            if steps == self.chunk:
                aim = seconds / 3.0 if math.isfinite(seconds) else 1.0
                wanted = self.chunk * max(aim, 1.0e-3) / max(taken, 1.0e-6)
                self.chunk = max(1, min(int(wanted), self.chunk * 2, 20000))
            self._schedule(now)
            progress.report(now, self.duration if self.duration > 0.0 and self.mode != "steady" else 0.0,
                            f"{self.scheme}, t = {now:.4g} s")

    def _advance_graph(self):
        """One check of QuadtreeFlood; its own rule says when to refine."""
        flood = self.flood
        self.left, reason = flood.advance()
        self.iterations += flood.iterations_per_check if reason != "fast" else flood.fast_iterations
        if reason is None:
            return
        self.levels_done.append({"cells": self._active(), "iterations": flood.iterations,
                                 "seconds": time.perf_counter() - self.level_began,
                                 "left": self.left if math.isfinite(self.left) else 1.0, "fast": reason == "fast"})
        self.level_began = time.perf_counter()
        began = time.perf_counter()
        added = flood.refine()
        self.ticks += 1
        if added is None:
            self.finished, self.outcome = True, flood.stop_reason
        else:
            self._tree_changed()
        self.tick_seconds += time.perf_counter() - began

    def _next_stop(self):
        stops = []
        if self.duration > 0.0 and self.mode != "steady":
            stops.append(self.duration)
        if self.mode == "interval" and self.flood.stop_reason is None and self.plan["unit"] == "time":
            due = self.last_refine + self._every()
            # A refinement held waits for the wet area, measured in steps, or the longest hold.
            stops.append(due if self.hold is None or self.flood.time < due else self.last_refine + self.hold["longest"] * self._every())
        if self.mode == "steady" and self.steady.started is not None:
            stops.append(self.steady.started + self.steady.maximum)
        if len(self.series[0]):
            # Not past a time of the rain series: the rain changes there.
            stops.extend(float(at) for at in self.series[0] if at > self.flood.time)
        if self.time_curve is not None and self.time_curve["x"] == "time":
            # Not past a point of the curve: dt changes there.
            stops.extend(at for at, _ in self.time_curve["points"] if at > self.flood.time)
        return min(stops) if stops else None

    def _steps_to_stop(self):
        """Steps to the next refinement due or the next measure of the wet area, at least 1."""
        stops = []
        if self.hold is not None:
            stops.append(self.hold["steps"] if self.hold_mark is None else self.hold_mark[0] + self.hold["steps"] - self.flood.steps)
        if self.plan["unit"] == "steps":
            due = self.last_refine + self._every()
            stops.append(due - self.flood.steps if self.hold is None or self.flood.steps < due else self.last_refine + self.hold["longest"] * self._every() - self.flood.steps)
        return max(1, int(min(stops))) if stops else self.chunk

    def _leaf_side(self):
        """Side of each current leaf in finest cells, as a device array."""
        import cupy as cp

        return cp.left_shift(cp.ones(self.flood.tree.n, cp.int32), self.flood.tree.level.astype(cp.int32))

    def _wet_area(self):
        """m2 of the leaves deeper than WET_DEPTH."""
        import cupy as cp

        tree = self.flood.tree
        n = tree.n
        side = self._leaf_side().astype(cp.float64)
        wet = (self.flood.h[:n] > WET_DEPTH) & tree.active[:n].astype(bool)
        return float(cp.sum(cp.where(wet, side * side, 0.0))) * self.cell * self.cell

    def _track_wet(self):
        """Every ``hold["steps"]``: the change of the wet area over them, a share of it."""
        if self.hold is None or self.mode != "interval":
            return
        steps = self.flood.steps
        if self.hold_mark is None:
            self.hold_mark = (steps, self._wet_area())
            return
        span = steps - self.hold_mark[0]
        if span >= self.hold["steps"]:
            area = self._wet_area()
            self.hold_change = abs(area - self.hold_mark[1]) / max(area, self.cell * self.cell) * self.hold["steps"] / span
            self.hold_mark = (steps, area)

    def _holding(self):
        """A refinement due waits while the wet area changes more than allowed, up to the longest hold."""
        if self.hold is None:
            return False
        if self._position() - self.last_refine >= self.hold["longest"] * self._every():
            return False
        return self.hold_change is None or self.hold_change > self.hold["change"]

    def _schedule(self, now):
        if self.mode == "interval":
            if self.flood.stop_reason is None and self._position() - self.last_refine >= self._every() and not self._holding():
                if self.hold is not None and (self.hold_change is None or self.hold_change > self.hold["change"]):
                    self.forced += 1
                self._refine()
        elif self.mode == "steady":
            outcome = self.steady.update(self.flood)
            if outcome is not None:
                self._end_level(now, outcome)
                self._refine_steady()
        if self.duration > 0.0 and self.mode != "steady" and now >= self.duration:
            self.finished = True
            self.outcome = "Reached the duration"

    def _end_level(self, now, outcome):
        self.levels_done.append({"cells": self._active(), "simulated": now - self.level_start,
                                 "seconds": time.perf_counter() - self.level_began,
                                 "imbalance": self.steady.imbalance, "depth_change": self.steady.rate,
                                 "steady": outcome == "steady"})
        self.level_start = now
        self.level_began = time.perf_counter()

    def _refine_steady(self):
        changed = self._refine()
        self.steady.reset()
        self.steady.update(self.flood)  # The level's clock starts now.
        if self.flood.stop_reason is not None:
            self.finished, self.outcome = True, self.flood.stop_reason
        elif not changed:
            self.finished, self.outcome = True, "No room left to refine" if self.flood.left_unsplit else "Nothing left to refine"

    def _refine(self):
        import cupy as cp

        began = time.perf_counter()
        tree = self.flood.tree
        before = cp.asnumpy(tree._key).copy()
        self.flood.refine()
        self.ticks += 1
        self.last_refine = self._position()
        # The wet area is measured again on the new leaves.
        self.hold_mark, self.hold_change = None, None
        # A split and a merge can leave the count as it was.
        changed = not np.array_equal(before, cp.asnumpy(tree._key))
        if changed:
            self.changes += 1
            self._tree_changed()
        self.tick_seconds += time.perf_counter() - began
        return changed

    def _active(self):
        return int(self.flood.tree.active.sum())

    def _tree_changed(self):
        import cupy as cp

        tree = self.flood.tree
        level = cp.asnumpy(tree.level)
        active = cp.asnumpy(tree.active)
        self.leaves_by_size = [[self.cell * (1 << int(l)), int(c)]
                               for l, c in enumerate(np.bincount(level[active].astype(np.int64))) if c > 0]

    def _velocity(self):
        """Velocity x and y per leaf; none yet from a GraphFlood not run."""
        flood = self.flood
        if self.graph:
            return flood.velocity() if flood._warm else None
        return flood.velocity()

    def _split_flags(self):
        import cupy as cp

        if self.graph and not self.flood._warm and set(condition_names(self.flood.criterion)) - {"h"}:
            return cp.zeros(self.flood.tree.n, cp.int32)
        return self.flood.flags()

    def _values(self):
        """The run's maxima (nevertheLAS's _write_values, without its files)."""
        import cupy as cp

        flood, tree = self.flood, self.flood.tree
        n = tree.n
        active = cp.asnumpy(tree.active)
        h = np.where(active, cp.asnumpy(flood.h[:n]), 0.0)
        velocity = self._velocity()
        q = 0.0 if velocity is None else np.hypot(cp.asnumpy(velocity[0][:n]), cp.asnumpy(velocity[1][:n])) * h
        self.maximum_depth = float(h.max()) if n else 0.0
        if not math.isfinite(self.maximum_depth):
            self.finished, self.error = True, "The flood went unstable: try a smaller CFL or time step"
        self.wet = int((h > WET_DEPTH).sum())
        self.split = int(cp.asnumpy(self._split_flags() > 0).sum())
        self.maximum_discharge = float(np.max(np.where(active, q, 0.0))) if n else 0.0

    # ---- outputs (nevertheLAS's export on the DEM's cells) -------------------

    def _wall_and_slope(self, tree, z, active):
        """Per leaf, (n, 3): the lowest of itself and its neighbours (INACTIVE off the
        footprint); dz/dx and dz/dy across its links."""
        import cupy as cp

        n = tree.n
        slots = cp.asnumpy(tree._slots).astype(np.int64)
        row, slot = np.nonzero((slots[:, None] >> np.arange(8)) & 1)  # Ascending slots: the order of ADJ.
        adj = cp.asnumpy(tree._array("ADJ")[: row.size]).astype(np.int64)
        heights = np.where(active, z, np.inf)
        link = active[adj] & active[row]
        base = heights.copy()
        np.minimum.at(base, row[link], heights[adj[link]])
        side = self.cell * np.left_shift(1, cp.asnumpy(tree.level).astype(np.int64)).astype(np.float64)
        sides = np.zeros((n, 4))
        counts = np.zeros((n, 4))
        face = slot >> 1  # 0 north, 1 west, 2 east, 3 south.
        np.add.at(sides, (row[link], face[link]), heights[adj[link]])
        np.add.at(counts, (row[link], face[link]), 1.0)
        filled = counts > 0.0
        means = np.where(filled, sides / np.maximum(counts, 1.0), np.where(active, z, 0.0)[:, None])
        # Mean centre distance over a side is the leaf's own side up to the 2:1 balance.
        run_x = np.where(filled[:, 2], 1.0, 0.0) + np.where(filled[:, 1], 1.0, 0.0)
        run_y = np.where(filled[:, 0], 1.0, 0.0) + np.where(filled[:, 3], 1.0, 0.0)
        dzdx = (means[:, 2] - means[:, 1]) / np.maximum(run_x * side, 1.0e-9)
        dzdy = (means[:, 0] - means[:, 3]) / np.maximum(run_y * side, 1.0e-9)
        base = np.where(active, base, INACTIVE)
        return np.stack([base, np.where(active, dzdx, 0.0), np.where(active, dzdy, 0.0)], axis=1)

    def _leaf_field(self, layer):
        """A layer per leaf, NaN off the footprint."""
        import cupy as cp

        flood, tree = self.flood, self.flood.tree
        n = tree.n
        h = cp.asnumpy(flood.h[:n]).astype(np.float64)
        if layer == "water_depth":
            values = h
        elif layer == "unit_discharge":
            velocity = self._velocity()
            values = np.zeros(n) if velocity is None else np.hypot(cp.asnumpy(velocity[0][:n]), cp.asnumpy(velocity[1][:n])) * h
        elif layer == "slope":
            z = cp.asnumpy(flood.z[:n]).astype(np.float64)
            slopes = self._wall_and_slope(tree, z, cp.asnumpy(tree.active))
            values = np.hypot(slopes[:, 1], slopes[:, 2])
        elif layer == "leaf_size":
            values = self.cell * np.left_shift(1, cp.asnumpy(tree.level).astype(np.int64)).astype(np.float64)
        else:
            raise ValueError(f"Unknown layer {layer!r}")
        return np.where(cp.asnumpy(tree.active), values, np.nan)

    def _pyramid(self, values, target, interpolate=True):
        """Leaf ``values`` on the cells of level ``target`` (rows from the north), as the
        working grids do: finer leaves averaged, coarser ones interpolated, 3/4 and 1/4
        along each axis, leaving out what has none; or, without ``interpolate``, copied."""
        import cupy as cp

        tree = self.flood.tree
        level = cp.asnumpy(tree.level).astype(np.int64)
        x, y = cp.asnumpy(tree.x).astype(np.int64), cp.asnumpy(tree.y).astype(np.int64)
        top = max(int(level.max()), target)
        shape = lambda l: (((self.rows - 1) >> l) + 1, ((self.columns - 1) >> l) + 1)  # noqa: E731

        fine = level <= target
        at = (y[fine] >> target, x[fine] >> target)
        total, weight = np.zeros(shape(target)), np.zeros(shape(target))
        known = np.zeros(shape(target), bool)
        known[at] = True
        area = 4.0 ** (level[fine] - target)
        has = ~np.isnan(values[fine])
        np.add.at(total, (at[0][has], at[1][has]), (area * values[fine])[has])
        np.add.at(weight, (at[0][has], at[1][has]), area[has])
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(weight > 0.0, total / weight, np.nan)
        pyramid = {target: (value, known)}

        for l in range(target + 1, top + 1):  # Up: the mean of the level finer, and the level's own leaves.
            finer, finer_known = pyramid[l - 1]
            rows, columns = shape(l)
            has = np.zeros((2 * rows, 2 * columns), bool)
            sums = np.zeros((2 * rows, 2 * columns))
            any_known = np.zeros((2 * rows, 2 * columns), bool)
            good = finer_known & ~np.isnan(finer)
            has[: good.shape[0], : good.shape[1]] = good
            sums[: good.shape[0], : good.shape[1]] = np.where(good, finer, 0.0)
            any_known[: good.shape[0], : good.shape[1]] = finer_known
            count = has.reshape(rows, 2, columns, 2).sum((1, 3))
            with np.errstate(invalid="ignore", divide="ignore"):
                value = np.where(count > 0, sums.reshape(rows, 2, columns, 2).sum((1, 3)) / count, np.nan)
            known = any_known.reshape(rows, 2, columns, 2).any((1, 3))
            own = level == l
            value[y[own] >> l, x[own] >> l] = values[own]
            known[y[own] >> l, x[own] >> l] = True
            pyramid[l] = (value, known)

        for l in range(top, target, -1):  # Down: what no leaf of a level covers.
            coarse = pyramid[l][0]
            value, known = pyramid[l - 1]
            rows, columns = np.nonzero(~known)
            px, py = columns >> 1, rows >> 1
            if not interpolate:
                value[rows, columns] = coarse[py, px]
                del pyramid[l]
                continue
            sx, sy = np.where(columns & 1, 1, -1), np.where(rows & 1, 1, -1)
            total, weight = np.zeros(rows.size), np.zeros(rows.size)
            for dx, dy, share in ((0, 0, 9.0), (sx, 0, 3.0), (0, sy, 3.0), (sx, sy, 1.0)):
                ax, ay = px + dx, py + dy
                inside = (ax >= 0) & (ay >= 0) & (ax < coarse.shape[1]) & (ay < coarse.shape[0])
                here = np.full(rows.size, np.nan)
                here[inside] = coarse[ay[inside], ax[inside]]
                has = ~np.isnan(here)
                total[has] += share * here[has]
                weight[has] += share
            with np.errstate(invalid="ignore", divide="ignore"):
                filled = np.where(weight > 0.0, total / weight, np.nan)
            filled[np.isnan(coarse[py, px])] = np.nan  # A child of a cell with none has none.
            value[rows, columns] = filled
            del pyramid[l]
        return pyramid[target][0]

    def _grid(self, layer, interpolate, cell=0.0):
        if cell > 0.0 and abs(cell - self.cell) > 1.0e-9 * self.cell:
            return self._grid_at(layer, cell)
        z = self._pyramid(self._leaf_field(layer), 0, interpolate)[: self.rows, : self.columns]
        # A leaf can straddle the edge of the footprint.
        return _like(self.dem, np.where(self._valid, z, np.nan).astype(np.float32))

    def _grid_at(self, layer, cell):
        dem = self.dem
        return GeoRaster(z=self._lattice(layer, cell).astype(np.float32), cell_size=cell, x_min=dem.x_min,
                         y_min=dem.y_min, epsg=dem.epsg, crs_wkt=dem.crs_wkt)

    def _lattice(self, layer, cell):
        """``layer`` on the lattice of ``cell`` from the DEM's south-west corner, rows from
        the north (nevertheLAS's export): coarser, from the pyramid level of that size;
        the depth keeps each leaf's own value. ``elevation``: the bed itself, not its
        mean over the leaves."""
        shape = (max(1, math.ceil(self.rows * self.cell / cell - 1.0e-9)),
                 max(1, math.ceil(self.columns * self.cell / cell - 1.0e-9)))
        ground = to_lattice(self.bed[::-1], 0.0, 0.0, self.cell, shape, cell)
        if layer == "elevation":
            source, size, interpolate = (self.bed + self.lowest)[::-1], self.cell, True
        else:
            target = 0
            while self.cell * 2 ** (target + 1) <= cell * (1.0 + 1.0e-9):
                target += 1
            interpolate = layer != "water_depth"
            source = self._pyramid(self._leaf_field(layer), target, interpolate)[::-1]
            size = self.cell * 2 ** target
        y_offset = self.rows * self.cell - source.shape[0] * size  # Coarse cells can pass the south edge.
        raster = to_lattice(source, 0.0, y_offset, size, shape, cell, interpolate)
        raster[np.isnan(ground)] = np.nan  # A leaf can straddle the edge of the footprint.
        return raster[::-1]

    def _outputs(self, p):
        cell = float(p["output_cell"])
        # Depth keeps each leaf's own value, so the raster holds the flood's volume.
        out = {"water_depth": self._grid("water_depth", False, cell)}
        if p["return_unit_discharge"]:
            out["unit_discharge"] = self._grid("unit_discharge", True, cell)
        if p["return_leaf_size"]:
            out["leaf_size"] = self._grid("leaf_size", False, cell)
        if p["return_slope"]:
            out["slope"] = self._grid("slope", True, cell)
        return out

    # ---- actions (nevertheLAS's refine_now, dt_limit, inspect) ----------------

    def refine_now(self):
        """One refinement, whatever the schedule says; the outputs on the new
        leaves."""
        self._refine_now()
        return {"text": self.status()["text"], "outputs": self._outputs(self.last_params)}

    def _refine_now(self):
        flood = self.flood
        if self.graph:
            began = time.perf_counter()
            if flood.refine() is not None:
                self._tree_changed()
            flood.stop_reason = None
            self.ticks += 1
            self.tick_seconds += time.perf_counter() - began
        else:
            self._refine()
            if self.steady is not None:
                self.steady.reset()
                self.steady.update(flood)

    def _place(self, leaf, side):
        """Corner and size of a leaf in map coordinates: x west, y south."""
        tree = self.flood.tree
        size = side * self.cell
        north = self.y_min + self.rows * self.cell - int(tree.y[leaf]) * self.cell
        return {"x": self.x_min + int(tree.x[leaf]) * self.cell, "y": north - size, "size": size}

    def _cfl_rates(self):
        """Per leaf or link, as the solver takes its step: (rate, leaf a, leaf b); b is None on a leaf."""
        import cupy as cp

        flood, tree = self.flood, self.flood.tree
        n = tree.n
        h = flood.h[:n].astype(cp.float32)
        dx = (self._leaf_side() * self.cell).astype(cp.float32)
        if self.scheme == "fusw":
            return (cp.maximum(cp.abs(flood.u[:n]), cp.abs(flood.v[:n]))
                    + cp.sqrt(GRAVITY * cp.maximum(h, 0.0))) / dx, None
        ptr = tree._array("PTR")[: n + 1]
        adj = tree._array("ADJ")[: tree.n_links]
        rows = cp.repeat(cp.arange(n, dtype=cp.int32), ptr[1:] - ptr[:-1])
        return (cp.sqrt(GRAVITY * cp.maximum(cp.maximum(h[rows], h[adj]), 0.0))
                / cp.minimum(dx[rows], dx[adj])), (rows, adj)

    def dt_limit(self):
        """The leaf whose water sets the CFL step (none when no CFL sets it,
        or nothing is wet)."""
        if self.graph or self.time_step != "cfl":
            return {"text": "No CFL sets the time step of this flood."}
        facts = self._dt_facts()
        if not facts:
            return {"text": "Nothing is wet: no leaf sets the time step."}
        about = "" if facts["exact"] else "about "
        size, depth, allowed, top = facts["size"], facts["depth"], facts["allowed"], facts["cap"]
        if facts["capped"]:
            text = (f"dt is capped at the largest step, {top:.3f} s. The tightest leaf allows "
                    f"{about}{allowed:.3f} s: {size:.2f} m, {depth:.3f} m deep")
        else:
            text = f"dt is set by a {size:.2f} m leaf {depth:.3f} m deep, which allows {about}{allowed:.3f} s"
        return {"text": text, "at": [facts["x"] + 0.5 * size, facts["y"] + 0.5 * size]}

    def _dt_facts(self):
        """The leaf whose water sets the CFL step: its place, depth, the step it allows,
        the flood's dt and cap ({} when no CFL sets it, or nothing is wet)."""
        import cupy as cp

        flood = self.flood
        if self.graph or self.time_step != "cfl":
            return {}
        rates, links = self._cfl_rates()
        at = int(cp.argmax(rates))
        rate = float(rates[at])
        if rate <= 0.0:
            return {}
        leaf = at
        if links is not None:
            a, b = int(links[0][at]), int(links[1][at])
            leaf = a if float(flood.h[a]) >= float(flood.h[b]) else b
        cfl = float(flood.cfl.read())
        top = float(flood.dt_max.read())
        side = 1 << int(flood.tree.level[leaf])
        return {**self._place(leaf, side), "depth": float(flood.h[leaf]), "allowed": cfl / rate, "dt": flood.dt,
                "capped": top > 0.0 and cfl / rate > top, "cap": top, "exact": self.scheme != "fusw"}

    def inspect(self, x, y):
        """What the flood holds at a point, and why its leaf is the size it is."""
        f = self._inspect_facts(float(x), float(y))
        if not f["inside"]:
            return {"text": "That point is off the flood's cells."}
        bed, h, speed = f["bed"], f["depth"], f["speed"]
        lines = [f"Cell {f['column']}, {f['row']} · x {f['cell_x']:.1f} · y {f['cell_y']:.1f}",
                 f"Leaf: {f['size']:.2f} m",
                 f"Water: {h:.3f} m deep · surface {bed + h:.2f} m · speed {speed:.3f} m/s · |q| {speed * h:.4f} m²/s"]
        if f.get("allowed") is not None:
            lines.append(f"Time step: this leaf allows {f['allowed']:.3f} s; the flood is at {f['dt']:.3f} s")
        lines += ["", "Size: " + f["refining"]]
        if f["conditions"]:
            lines.append(" · ".join(f"{c['name']} {c['value']:.4f} of {c['threshold']:.4f}" for c in f["conditions"]))
        lines += ["", f"Bed of the leaf: {bed:.2f} m"]
        if f["raster_bed"] is not None:
            lines.append(f"Bed of this cell: {f['raster_bed']:.2f} m")
        if f["neighbour_low"] is not None:
            lines.append(f"Leaves next to it: {f['neighbour_low']:.2f} to {f['neighbour_high']:.2f} m")
        return {"text": "\n".join(lines), "at": [f["cell_x"], f["cell_y"]]}

    def _inspect_facts(self, x, y):
        """What the flood holds at a point of map coordinates, and why it is so (nevertheLAS's inspect)."""
        import cupy as cp

        flood, tree = self.flood, self.flood.tree
        column = math.floor((x - self.x_min) / self.cell)
        row = math.floor((self.y_min + self.rows * self.cell - y) / self.cell)
        out = {"inside": False, "column": column, "row": row}
        if not (0 <= column < self.columns and 0 <= row < self.rows):
            return out
        sides = self._leaf_side()
        found = cp.nonzero((tree.x <= column) & (column < tree.x + sides) & (tree.y <= row)
                           & (row < tree.y + sides) & tree.active)[0]
        if found.size == 0:
            return out
        leaf = int(found[0])
        side = int(sides[leaf])
        ptr = tree._array("PTR")
        links = tree._array("ADJ")[int(ptr[leaf]): int(ptr[leaf + 1])]
        links = links[tree.active[links]]
        beds = flood.z[links]
        velocity = self._velocity()
        h = float(flood.h[leaf])
        u, v = (0.0, 0.0) if velocity is None else (float(velocity[0][leaf]), float(velocity[1][leaf]))
        names = condition_names(flood.criterion)
        above = flood.threshold if not isinstance(flood.threshold, (int, float)) else [flood.threshold] * len(names)
        conditions, own, beside = [], False, False
        for name, threshold in zip(names, above):
            if self.graph and not flood._warm and name != "h":
                continue
            values = flood.quantity(name)
            value = float(values[leaf])
            conditions.append({"name": name, "value": value, "threshold": float(threshold)})
            own = own or value >= threshold
            beside = beside or bool(links.size and (values[links] >= threshold).any())
        centre = (self.y_min + self.rows * self.cell - (row + 0.5) * self.cell, self.x_min + (column + 0.5) * self.cell)
        raster = self.bed[row, column]
        out.update(self._place(leaf, side))
        out.update(inside=True, cell_x=centre[1], cell_y=centre[0], cell=self.cell, level=int(tree.level[leaf]),
                   bed=float(flood.z[leaf]) + self.lowest,
                   raster_bed=None if math.isnan(raster) else float(raster) + self.lowest,
                   neighbour_low=float(beds.min()) + self.lowest if beds.size else None,
                   neighbour_high=float(beds.max()) + self.lowest if beds.size else None,
                   depth=h, speed=math.hypot(u, v), discharge=math.hypot(u, v) * h, conditions=conditions,
                   refining=self._why_size(int(tree.level[leaf]), int(self._split_flags()[leaf]), own, beside))
        if not self.graph and self.time_step == "cfl":
            if self.scheme == "fusw":
                rate = (max(abs(u), abs(v)) + math.sqrt(GRAVITY * max(h, 0.0))) / (side * self.cell)
            else:
                rate = math.sqrt(GRAVITY * max(h, 0.0)) / (side * self.cell)
                if links.size:
                    rate = (math.sqrt(GRAVITY * max(h, float(flood.h[links].max()), 0.0))
                            / (min(side, int(sides[links].min())) * self.cell))
            out.update(allowed=float(flood.cfl.read()) / rate if rate > 0.0 else None, dt=flood.dt)
        return out

    def _why_size(self, level, flag, own, beside):
        """Why a leaf is the size it is, and what comes next for it."""
        flood = self.flood
        size = f"{self.cell * (1 << level):g} m"
        if level == 0:
            return f"At the minimum cell ({size}): it cannot split." + (" Its water is over a value, and stays so." if own else "")
        if flag > 0:
            cause = "Over a value" if own else "A neighbour is over a value" if beside else "It holds a source"
            if flood.stop_reason:
                return f"{cause}, but refining stopped: {flood.stop_reason}."
            if flood.left_unsplit:
                return (f"{cause}, but Most leaves is reached: {flood.left_unsplit} leaves are left as they are, "
                        "those furthest over their values split first.")
            if self.mode == "off":
                return f"{cause}, but refining is off."
            return f"{cause}: it splits at the next refinement."
        if flag < 0:
            return "Below every merge value: it merges with its siblings at the next refinement, if they all are."
        if flood.coarsen_below is None:
            gone = f" Merging is off: if it was split for water now gone, it stays at {size}." if level < self.top else ""
            return "Under every value: it does not split." + gone
        return "Under every value: it does not split, and is not low enough to merge."

    # ---- runner ------------------------------------------------------------

    def coordinate(self):
        return {"step": self.iterations if self.graph else self.flood.steps}

    def status(self):
        flood = self.flood
        parts = [f"{self.flood.tree.n} leaves ({self._active()} active)"]
        if self.graph:
            parts.insert(0, f"level {len(self.levels_done) + 1}, {self.iterations} iterations")
            if math.isfinite(self.left):
                parts.append(f"{_num(100.0 * self.left)} % of the water left to move")
        else:
            parts.insert(0, f"t = {_num(flood.time, 's')}, dt = {_num(flood.dt, 's')}")
            if self.mode == "interval" and self.plan is not None:
                left = max(self.last_refine + self._every() - self._position(), 0.0)
                parts.append(f"next refinement in {_num(left, 's' if self.plan['unit'] == 'time' else 'steps')}")
            elif self.mode == "steady" and self.steady is not None:
                parts.append(f"level {len(self.levels_done) + 1}, {_num(flood.time - self.level_start, 's')} on it")
        parts.append(f"max depth {_num(self.maximum_depth, 'm')}, {self.wet} wet")
        if flood.stop_reason:
            parts.append(f"refining stopped: {flood.stop_reason}")
        text = ", ".join(parts) + "."
        if self.finished:
            text += f" Done: {self.outcome}."
        return {"text": text, "stopped": self.finished}

    def close(self):
        self.flood.close()


# ---------------------------------------------------------------------------
# Upstream entries
# ---------------------------------------------------------------------------

@process(
    id="pyfastflow.upstream_entries",
    category="Flow",
    label="Upstream entries (Q = P x A)",
    inputs=[
        Port("dem", "georaster", label="Flood domain",
             doc="The flood's DEM: its valid cells are the domain. France only (IGN RGE ALTI)."),
    ],
    params=[
        Param("cell", "float", label="Routing cell", unit="m", default=5.0, choices=[1.0, 5.0, 10.0, 25.0],
              doc="Cell of the elevation read around the domain and routed."),
        Param("max_km2", "float", label="Largest area read", unit="km²", default=500.0, min=1.0,
              doc="The box read grows while a catchment draining to the domain is cut by its "
                  "edge, up to this."),
        Param("min_km2", "float", label="Smallest entry", unit="km²", default=0.1, min=0.0,
              doc="Smaller ones are added to the nearest kept entry: the total area is kept."),
        Param("radius", "float", label="Spread radius", unit="m", default=0.0, min=0.0,
              doc="Radius of the sources made (0: each entry on one cell, the lowest near "
                  "where the routed flow crosses into the domain)."),
    ],
    outputs=[
        Output("sources", "geopoints",
               doc="One discharge source per entry: area (m²), linked = 1 (q = rain x area), "
                   "q = 0, radius, own_area (m², before merging), merged (entries added)."),
    ],
    impl="library",
)
def upstream_entries(dem, cell=5.0, max_km2=500.0, min_km2=0.1, radius=0.0):
    """Discharge entering ``dem``'s domain from the land upstream of it: IGN's
    RGE ALTI read around the domain at ``cell`` (Lambert-93), routed by
    steepest descent with pyfastflow (Cordonnier carving through pits); each
    entry is the area outside that drains to it, placed on the lowest domain
    cell near the routed crossing (nevertheLAS's upstream_entries)."""
    from . import _upstream

    bed = np.where(dem.valid, dem.z, np.nan).astype(np.float32)
    crs_wkt = dem.crs_wkt
    if not crs_wkt and dem.epsg:
        from rasterio.crs import CRS

        crs_wkt = CRS.from_epsg(int(dem.epsg)).to_wkt()
    found = _upstream.compute(bed, dem.x_min, dem.y_min, dem.cell_size, crs_wkt, float(cell),
                              float(max_km2), float(min_km2), progress.report)
    entries = found["entries"]
    n = len(entries)
    xy = np.array([[e["x"], e["y"]] for e in entries], np.float64).reshape(n, 2)
    attrs = {"q": np.zeros(n), "area": np.array([e["area"] for e in entries], np.float64),
             "linked": np.ones(n), "radius": np.full(n, float(radius)),
             "own_area": np.array([e["own_area"] for e in entries], np.float64),
             "merged": np.array([e["merged"] for e in entries], np.float64)}
    return {"sources": GeoVector(geometry="points", xy=xy, feature_attrs=attrs, epsg=dem.epsg,
                                 crs_wkt=dem.crs_wkt)}
