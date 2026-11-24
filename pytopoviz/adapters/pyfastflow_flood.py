"""One flood process over every pyfastflow flood solver.

``pyfastflow.flood`` takes a ``solver`` and runs that solver's own runner
unchanged: GraphFlood (relaxation, explicit, transient), GraphFlood by
particles, the inertial flood, FullSWOF, the quadtree flood and the quick
flood (a quadtree flood with set settings, nevertheLAS's Quick flood).

Its interface is the union of the solvers' (see ``_build``): a param or port
used by one solver, or the same in all that use it, keeps its name; one that
differs between solvers takes the prefixes of the solvers sharing that
version (``fusw__cfl``, ``inertial_qt__regulator_theta``). Every
member carries ``show_if: {"solver": [...]}``, the solvers using it, and a
``group``, for a host's form.

It adds what every solver takes the same way:

- the rain as a uniform rate (mm/h), a map (mm/h) or a map times the rate
  (``rain_mode``, port ``rain_map``); the rate and the map's scale change
  while running;
- a map of the discharge entering the cells (port ``source_map``), in mm/h
  or m³/s per cell, times a scale (``source_map_unit``, ``source_map_scale``);
- the discharge sources (points; see pyfastflow._source_stamps).

The rain factor over time (``rain_series``) is read by the transient solvers
(inertial, FullSWOF, quadtree); GraphFlood is steady.

Author: B.G.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core import PROCESSES, Output, Param, Port, Runner, process, run_once, runner
from ..core.runner import RUNNERS
from .pyfastflow import _MM_PER_H, _field, _like, _metric_cell_size, _source_cells

#: Solver key -> (label, process id, params set for it).
SOLVERS = {
    "graphflood_relax": ("GraphFlood: relaxation", "pyfastflow.graphflood", {"solver": "analytical"}),
    "graphflood_vanilla": ("GraphFlood: explicit (vanilla)", "pyfastflow.graphflood", {"solver": "explicit"}),
    "graphflood_transient": ("GraphFlood: transient", "pyfastflow.graphflood", {"solver": "transient"}),
    "graphflood_particle": ("GraphFlood: particles", "pyfastflow.graphflood_particle", {}),
    "inertial": ("Inertial (local inertia)", "pyfastflow.inertial_flood", {}),
    "fusw": ("FullSWOF (shallow water)", "pyfastflow.fusw_flood", {}),
    "quadtree": ("Quadtree (adaptive)", "pyfastflow.quadtree_flood", {}),
    "quick": ("Quick flood", "pyfastflow.quadtree_flood", {
        "scheme": "inertial", "levels": 7, "refine": "interval", "refine_unit": "steps",
        "refine_every": 6500.0, "manning": 0.033, "time_step": "cfl", "cfl": 0.4, "dt_max": 30.0,
        "refine_on_h": True, "h_above": 0.05, "refine_on_q": True, "q_above": 0.02,
        "hold_steps": 1000.0, "hold_change": 1.0, "hold_longest": 5.0, "duration": 0.0}),
}
#: Quick flood: the quadtree params it shows; the others are its set values or the defaults.
QUICK_SHOWN = ("rainfall", "hold")
#: Process id -> prefix of its params that differ from the other solvers'.
PREFIX = {"pyfastflow.graphflood": "gf", "pyfastflow.graphflood_particle": "gfp",
          "pyfastflow.inertial_flood": "inertial", "pyfastflow.fusw_flood": "fusw",
          "pyfastflow.quadtree_flood": "qt"}
#: Members the flood sets itself, per process: the rain and source fields,
#: GraphFlood's solver, the continuation states (the runner continues).
SET_HERE = {"precipitation_field", "source_field", "previous_state", "return_state", "state",
            "field_scale"}
#: Shared whatever their defaults (the first solver's default and doc are kept).
#: Others are shared only when alike and fixed at opening by all their solvers
#: or by none (else changing one would reopen a solver that takes it live).
SHARED = {"rainfall"}
#: Solver key -> params of its process it does not read (GraphFlood's three
#: modes are one process).
_RELAX_ONLY = {"analytical_relaxation", "analytical_solver", "warmup_steps", "warmup_relaxation",
               "depth_cap"}
UNUSED = {
    "graphflood_relax": {"dt"},
    "graphflood_vanilla": _RELAX_ONLY,
    "graphflood_transient": _RELAX_ONLY | {"local_minima", "carve_slope_min"},
}
_CONVERGED = {n: {"stop": ["converged"]} for n in (
    "convergence_tol", "convergence_window", "convergence_eps")}
#: Process id -> {param: {its param: [values]}}: read only when that param
#: has one of the values (added to the show_if when every solver of the
#: flood's param agrees).
WHEN = {
    "pyfastflow.graphflood": _CONVERGED,
    "pyfastflow.graphflood_particle": _CONVERGED,
    "pyfastflow.inertial_flood": {"dt": {"time_step": ["fixed"]},
                                  "dt_max": {"time_step": ["auto"]},
                                  "cfl": {"time_step": ["auto"]}},
    "pyfastflow.fusw_flood": {"dt": {"time_step": ["fixed"]},
                              "dt_max": {"time_step": ["cfl"]},
                              "cfl": {"time_step": ["cfl"]},
                              "limiter": {"order": [2]},
                              "max_restarts": {"order": [2]},
                              "friction_coefficient": {
                                  "friction": ["manning", "darcy_weisbach", "laminar"]}},
    "pyfastflow.quadtree_flood": {
        "dt": {"scheme": ["inertial", "fusw"], "time_step": ["fixed"]},
        "dt_max": {"scheme": ["inertial", "fusw"], "time_step": ["cfl"]},
        "cfl": {"scheme": ["inertial"], "time_step": ["cfl"]},
        "fusw_cfl": {"scheme": ["fusw"], "time_step": ["cfl"]},
        "max_restarts": {"scheme": ["fusw"], "time_step": ["cfl"]},
        "limiter": {"scheme": ["fusw"], "order": [2]},
        **{n: {"scheme": ["inertial", "fusw"], "time_step": ["curve"]}
           for n in ("dt_curve", "dt_curve_x", "dt_curve_ramp")}},
}
#: Kept apart per process even when alike (their WHEN rows differ).
SPLIT = {"dt_max", "limiter"}

_RAIN_MODES = ["uniform", "map", "map_rate"]
_SOURCE_UNITS = ["mm_h", "m3s_cell"]

_GROUPS = {
    "start": ("h_init",),
    "sources": ("sources",),
    "rain": ("rainfall", "rain_series", "series_interp", "storm_duration", "inflow_series",
             "source_series"),
    "friction": ("manning", "friction", "friction_coefficient", "friction_exponent",
                 "manning_field", "friction_field"),
    "infiltration": ("infiltration", "ks", "kc", "dtheta", "psi", "zcrust", "imax", "ie_ca",
                     "ks_field", "kc_field", "dtheta_field", "psi_field", "zcrust_field",
                     "imax_field"),
    "run": ("steps", "launches", "duration", "chunk_seconds"),
    "convergence": ("stop", "check_every", "convergence_metric", "convergence_tol",
                    "convergence_window", "convergence_eps", "convergence_percentile",
                    "convergence_memory", "tol", "iterations_per_check", "window", "eps"),
    "time": ("time_step", "dt", "dt_max", "cfl", "fusw_cfl", "dt_update_steps", "dt_curve",
             "dt_curve_x", "dt_curve_ramp", "max_restarts", "check_steps"),
    "start": ("initial_water", "warmup_steps", "warmup_relaxation", "depth_cap",
              "source_percentile"),
    "domain": ("topology", "boundary", "outlet_edges", "nodata_outlets", "local_minima",
               "carve_slope_min", "fix_outlet_depth", "outlet_depth", "boundary_flux",
               "outlet_discharge", "top_inflow", "outlet", "outlet_value", "fusw_outlet",
               "outlet_slope", "outlet_manning", "nodata_boundary", "inflow_discharge",
               "inflow_slope", "inflow_lines"),
    "particles": ("particles", "h_relaxation", "walk_steps", "spawn_pad", "propagate", "seed",
                  "threads"),
    "refinement": ("levels", "edge_level", "capacity", "bed_rule", "bed_percentile",
                   "refine_neighbours", "refine", "refine_every", "refine_unit", "plan_points",
                   "plan_x", "plan_ramp", "hold", "hold_steps", "hold_change", "hold_longest",
                   "balance", "depth_change", "held", "maximum_per_level", "hold_sources",
                   "fast_cycles", "fast_relaxation", "fast_iterations", "level_field"),
    "outputs": ("report_every", "probe_every", "output_cell", "probes"),
}
_GROUP_OF = {name: group for group, names in _GROUPS.items() for name in names}


def _group(name):
    if name.startswith("return_"):
        return "outputs"
    if name.startswith(("refine_on_", "q_", "h_", "speed_", "u_", "v_")):
        return "refinement"
    if name.split("_")[0] in ("north", "south", "east", "west"):
        return "domain"
    return _GROUP_OF.get(name, "scheme")


def _same(a, b):
    return (a.types, a.default, list(a.choices or []), a.min, a.max) == \
        (b.types, b.default, list(b.choices or []), b.min, b.max)


def _members(kind):
    """{name: {process id: (solver keys, member)}} of ``kind`` (inputs,
    params, outputs) over the solvers, the members set here left out."""
    found = {}
    for key, (_label, pid, fixed) in SOLVERS.items():
        for m in getattr(PROCESSES.get(pid).spec, kind):
            if m.name in SET_HERE or (kind == "params" and m.name in fixed):
                continue
            if key == "quick" and kind == "params" and m.name not in QUICK_SHOWN:
                continue
            if kind == "params" and m.name in UNUSED.get(key, ()):
                continue
            found.setdefault(m.name, {}).setdefault(pid, ([], m))[0].append(key)
    return found


def _label(m):
    return m.label or m.name.replace("_", " ").capitalize()


def _build():
    """(ports, params, outputs, names): the union interface, and per solver
    key {its process's member name: the flood's name}."""
    names = {key: {} for key in SOLVERS}
    ports, params, outputs, when = [], [], [], []
    for kind, out in (("inputs", ports), ("params", params), ("outputs", outputs)):
        for name, by_pid in _members(kind).items():
            # Classes of processes taking it alike: the same definition, and
            # fixed at opening by all or by none.
            versions = []
            for pid, (keys, m) in by_pid.items():
                for cls in versions:
                    other_pid = SOLVERS[cls[0][0]][1]
                    if kind == "params" and name in SPLIT and other_pid != pid:
                        continue
                    if kind != "params" or name in SHARED or name.startswith("return_") or (
                            _same(cls[1], m)
                            and (name in RUNNERS[pid].config) == (name in RUNNERS[other_pid].config)):
                        cls[0].extend(keys)
                        break
                else:
                    versions.append((list(keys), m))
            alike = len(versions) == 1
            for keys, m in versions:
                prefixes = list(dict.fromkeys(PREFIX[SOLVERS[k][1]] for k in keys))
                flood_name = name if alike else f"{'_'.join(prefixes)}__{name}"
                for k in keys:
                    names[k][name] = flood_name
                if kind == "outputs":
                    out.append(Output(flood_name, m.type, doc=m.doc, optional=name != "water_depth",
                                      label=m.label, unit=m.unit))
                    continue
                hints = dict(show_if={"solver": list(keys)}, group=_group(name),
                             label=_label(m), unit=m.unit, doc=m.doc)
                if kind == "inputs":
                    out.append(Port(flood_name, m.type, optional=True, **hints))
                else:
                    out.append(Param(flood_name, m.type, default=m.default, choices=m.choices,
                                     min=m.min, max=m.max, choice_labels=m.choice_labels,
                                     **hints))
                    when.append((len(out) - 1, name, keys))
    for i, name, keys in when:
        conds = []
        for k in keys:
            raw = WHEN.get(SOLVERS[k][1], {}).get(name)
            conds.append(None if raw is None or k == "quick" else
                         {names[k].get(o, o): list(v) for o, v in raw.items()})
        if conds[0] is not None and all(c == conds[0] for c in conds):
            params[i] = replace(params[i], show_if={**params[i].show_if, **conds[0]})
    return ports, params, outputs, names


_PORTS, _PARAMS, _OUTPUTS, _NAMES = _build()
_ALL = list(SOLVERS)

_FLOOD_PORTS = [
    Port("dem", "georaster", label="Topography", group="inputs",
         doc="Topography (projected, metres); nodata cells stay out."),
    Port("rain_map", "georaster", label="Rain map", optional=True, group="rain",
         show_if={"rain_mode": ["map", "map_rate"]},
         doc="Map rain: the rate per cell (mm/h). Map x rate: the weight of the rate per "
             "cell. Same grid as the DEM; NaN: no rain."),
    Port("source_map", "georaster", label="Map of inputs", optional=True, group="sources",
         doc="Discharge entering each cell, in Its values are units, times the scale; added "
             "to the rain. Same grid as the DEM."),
    *[p for p in _PORTS if p.name != "dem"],
]
_FLOOD_PARAMS = [
    Param("solver", "string", default="inertial", choices=_ALL,
          choice_labels=[SOLVERS[k][0] for k in _ALL], group="solver",
          doc="The flood solver; its params show under it. GraphFlood: the steady water depth "
              "for a constant rain (relaxation, explicit, transient steps, or particles). "
              "Inertial and FullSWOF: transient shallow water. Quadtree: inertial, FullSWOF or "
              "GraphFlood on leaves refined where the water is. Quick flood: a quadtree flood "
              "with set settings."),
    Param("rain_mode", "string", default="uniform", choices=_RAIN_MODES,
          choice_labels=["Uniform rate", "Rain map (mm/h)", "Map x rate"], group="rain",
          doc="Uniform: Rainfall everywhere. Rain map: the map's rate per cell. Map x rate: "
              "the map (a weight) times Rainfall."),
    Param("source_map_unit", "string", default="m3s_cell", choices=_SOURCE_UNITS,
          choice_labels=["mm/h", "m³/s per raster cell"], group="sources",
          label="Its values are", doc="Unit of the map of inputs."),
    Param("source_map_scale", "float", default=1.0, min=0.0, group="sources",
          label="Multiplied by", doc="Factor of the map of inputs; changes while running."),
    *_PARAMS,
]
for _p in _FLOOD_PARAMS:
    if _p.name == "rainfall":
        _FLOOD_PARAMS[_FLOOD_PARAMS.index(_p)] = Param(
            "rainfall", "float", unit="mm/h", default=20.0, min=0.0, label="Rainfall",
            group="rain", show_if={"rain_mode": ["uniform", "map_rate"]},
            doc="Rain rate (mm/h); with Map x rate, times the map. Times the rain factor over "
                "time (transient solvers). Changes while running.")
        break

#: Runner config: GraphFlood's and the solvers' params fixed at opening, under the flood's names.
_CONFIG = ["solver", "rain_mode", "source_map_unit"] + sorted({
    names[n] for key, names in _NAMES.items() for n in RUNNERS[SOLVERS[key][1]].config
    if n in names})


@process(
    id="pyfastflow.flood",
    category="Flood",
    label="Flood (every solver)",
    inputs=_FLOOD_PORTS,
    params=_FLOOD_PARAMS,
    outputs=_OUTPUTS,
    impl="library",
)
def flood(**kwargs):
    """A flood by the chosen ``solver``: its own runner, its params under the
    flood's names (see the module doc), the rain and the map of inputs set the
    same way for all."""
    ports = {p.name for p in _FLOOD_PORTS}
    inputs = {k: kwargs.pop(k) for k in list(kwargs) if k in ports}
    spec = PROCESSES.get("pyfastflow.flood").spec
    params = {p.name: p.default_value for p in spec.params if p.has_default}
    unknown = set(kwargs) - set(params)
    if unknown:
        raise TypeError(f"flood: unknown params {sorted(unknown)}")
    params.update(kwargs)
    return run_once(FloodRun, inputs, params)


@runner("pyfastflow.flood", axis="", config=_CONFIG, actions={
    "refine_now": "Refine now", "dt_limit": "What sets the time step",
    "inspect": "Inspect a point"})
class FloodRun(Runner):
    """The chosen solver's runner, given its params and the rain and map of
    inputs it takes. The rain rate and map scale apply from the next advance."""

    who = "Flood"

    def __init__(self, inputs, p):
        key = p["solver"]
        if key not in SOLVERS:
            raise ValueError(f"{self.who}: unknown solver {key!r}")
        if p["rain_mode"] not in _RAIN_MODES:
            raise ValueError(f"{self.who}: unknown rain_mode {p['rain_mode']!r}")
        if p["source_map_unit"] not in _SOURCE_UNITS:
            raise ValueError(f"{self.who}: unknown source_map_unit {p['source_map_unit']!r}")
        self.key, (_l, self.pid, self.fixed) = key, SOLVERS[key]
        self.mode = p["rain_mode"]
        self.dem = inputs["dem"]
        self.valid = self.dem.valid
        self.cell = _metric_cell_size(self.dem, self.who)
        self.rain_map = inputs.get("rain_map")
        if self.mode != "uniform" and self.rain_map is None:
            raise ValueError(f"{self.who}: the rain mode needs a rain map")
        self.source_map = inputs.get("source_map")
        names = _NAMES[key]
        inner = {"dem": self.dem}
        for name, flood_name in names.items():
            if flood_name in inputs and inputs[flood_name] is not None:
                inner[name] = inputs[flood_name]
        self.rain_now = self._rain_raster(p)
        self.source_now = self._source_raster(p)
        inner["precipitation_field"] = self.rain_now[1]
        inner["source_field"] = self.source_now[1]
        self.inner = RUNNERS[self.pid](inner, self._params(p))

    def _params(self, p):
        """The solver's params from the flood's."""
        spec = PROCESSES.get(self.pid).spec
        out = {m.name: m.default_value for m in spec.params if m.has_default}
        for name, flood_name in _NAMES[self.key].items():
            if flood_name in p and name in out:
                out[name] = p[flood_name]
        out.update(self.fixed)
        if "field_scale" in out:
            out["field_scale"] = 1.0  # the maps are scaled here
        if self.mode != "uniform" and self.pid == "pyfastflow.quadtree_flood":
            out["rainfall"] = 0.0  # the map holds the rain
        return out

    def _rain_raster(self, p):
        """(key, GeoRaster in mm/h or None) of the rain map for this mode and rate."""
        if self.mode == "uniform":
            return (None, None)
        rate = float(p["rainfall"]) if self.mode == "map_rate" else 1.0
        z = np.maximum(np.nan_to_num(_field(self.rain_map, 0.0, self.valid.shape, "rain_map",
                                            self.who), nan=0.0), 0.0) * rate
        return (rate, _like(self.dem, z))

    def _source_raster(self, p):
        """(key, GeoRaster in m³/s per cell or None) of the map of inputs."""
        if self.source_map is None:
            return (None, None)
        scale = float(p["source_map_scale"])
        unit = _MM_PER_H * self.cell * self.cell if p["source_map_unit"] == "mm_h" else 1.0
        z = np.nan_to_num(_field(self.source_map, 0.0, self.valid.shape, "source_map", self.who),
                          nan=0.0) * unit * scale
        return (scale, _like(self.dem, z))

    def _follow_maps(self, p):
        """Gives the solver the rain map and map of inputs again when the rate or
        the scale changed; each solver reads them from the next advance."""
        rain, source = self._rain_raster(p), self._source_raster(p)
        if rain[0] == self.rain_now[0] and source[0] == self.source_now[0]:
            return
        self.rain_now, self.source_now = rain, source
        r, valid = self.inner, self.valid
        if self.pid == "pyfastflow.quadtree_flood":
            cell = r.cell * r.cell
            r._rain_cells = None if rain[1] is None else np.where(valid, rain[1].z * _MM_PER_H * cell, 0.0)
            r._src_cells = None if source[1] is None else np.where(valid, np.maximum(source[1].z, 0.0), 0.0)
            r._source_spec = None
            if r._settings_now is not None:
                r._apply_sources(r._settings_now)
            return
        r.rain_field = rain[1]
        if self.pid == "pyfastflow.fusw_flood":
            r.source_base = None if source[1] is None else np.where(valid, source[1].z, 0.0)
        else:
            r.source_cells = _source_cells(source[1], valid, self.who)

    def advance(self, p):
        self._follow_maps(p)
        out = self.inner.advance(self._params(p))
        back = {name: flood_name for name, flood_name in _NAMES[self.key].items()}
        return {back.get(k, k): v for k, v in out.items()}

    def coordinate(self):
        return self.inner.coordinate()

    def status(self):
        return self.inner.status()

    def close(self):
        self.inner.close()

    def _quadtree(self, action):
        if not hasattr(self.inner, action):
            raise ValueError(f"{self.who}: {SOLVERS[self.key][0]} has no {action.replace('_', ' ')}")
        return getattr(self.inner, action)

    def refine_now(self):
        return self._quadtree("refine_now")()

    def dt_limit(self):
        return self._quadtree("dt_limit")()

    def inspect(self, x, y):
        return self._quadtree("inspect")(x, y)
