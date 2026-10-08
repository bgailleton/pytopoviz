"""pyfastflow adapter.

Registration unit for pyfastflow (DESIGN.md §8). pyfastflow's programs run on a
CuPy (GPU) backend, created once per python session on first use. Each process
builds its programs, runs them and closes them within the call, so no device
state outlives a call; results cross back as numpy-backed types.

- ``pyfastflow.saleve_steady``: Salève steady-state landscape generator, after
  ``compute_landscape`` in pyfastflow's ``examples/saleve_app.py`` (uplift /
  erodibility fields from noise stacks: Perlin, ridged, Worley, gradient;
  white-noise initial surface, one steady multigrid solve).
  Optional post-process: the multi-scale erosion of Schott et al. (2024)
  (``pyfastflow.experimental.mserosion``, a port of its release code).
- ``pyfastflow.perlin_surface``: Perlin noise surface with an optional
  southward regional slope (the starting surface of pyfastflow's GOLEM
  examples).
- ``pyfastflow.golem_nosed``: N time steps of the experimental sediment-free
  GOLEM landscape evolution model (``GolemNoSedProgram``: uplift, SFD
  routing, implicit stream-power incision, implicit linear hillslope
  diffusion) from a given surface. Stateless: a long run is a chain of calls,
  each starting from the previous call's grid.
- ``pyfastflow.graphflood``: N steps of GraphFlood (water depth and
  discharge from rainfall on a DEM; analytical by ``GraphFloodRelax``,
  explicit or transient by ``GraphFloodVanilla``). Stateless like GOLEM: a
  long run is a chain of calls, each starting from the previous call's depth
  (``h_init``).
- ``pyfastflow.graphflood_particle``: the steady state by particle
  processors (``GraphFloodParticles``): warm-up, launches, finish, in one
  call.

Author: B.G.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyfastflow  # noqa: F401  (hard import: the adapter is skipped without it)
from pyfastflow.core import Backend
from pyfastflow.experimental.golem import GolemNoSedProgram
from pyfastflow.experimental.mserosion import PRESET as MSE_PRESET
from pyfastflow.experimental.mserosion import amplify as mse_amplify
from pyfastflow.experimental.mserosion import upsample_x2 as mse_upsample_x2
try:  # GraphFlood API is being reworked in pyfastflow; skip it, keep the rest
    from pyfastflow.graphflood import GraphFloodParticles, GraphFloodRelax, GraphFloodVanilla
except ImportError:
    GraphFloodParticles = GraphFloodRelax = GraphFloodVanilla = None
try:  # added later than the programs; without it GraphFlood still runs, unchecked
    from pyfastflow.graphflood import ConvergenceRule
except ImportError:
    ConvergenceRule = None
try:
    from pyfastflow.flood import InertialFloodProgram
except ImportError:
    InertialFloodProgram = None
from pyfastflow.flow import BOUNDARIES, LOCAL_MINIMA, RECEIVER_MODES, TOPOLOGIES
from pyfastflow.noise import PerlinNoiseProgram
from pyfastflow.saleve import SLOPE_CORRECTIONS, SaleveProgram

from ..core import (Output, Param, Port, Runner, process, progress, register_type,
                    run_once, runner)
from ..datatable import DataTable
from ..georaster import GeoRaster

_BACKEND = None


def _backend():
    """The session's CuPy backend, created on first use (fails there, with
    CuPy's own message, when no GPU is usable)."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = Backend.from_name("cupy")
    return _BACKEND


def _gpu_sync():
    """Waits for the queued GPU work (progress reports follow finished steps)."""
    import cupy as cp

    cp.cuda.Device().synchronize()


def _perlin_noise(nx, ny, dx, amplitude, freq, octaves, persistence, seed, slope=0.0):
    """Perlin noise of ``amplitude`` on an (ny, nx) grid, minus ``slope * row *
    dx`` (row 0 north) when ``slope`` is set, as a CuPy array."""
    # The noise program works on half-integer frequencies.
    freq = max(round(2.0 * freq) / 2.0, 0.5)
    with PerlinNoiseProgram(_backend(), nx=int(nx), ny=int(ny), dx=float(dx),
                            amplitude=float(amplitude), frequency=freq,
                            octaves=int(octaves), persistence=float(persistence),
                            seed=int(seed)) as noise:
        noise.generate()
        if slope:
            noise.slope.set(float(slope))
            noise.add_southward_slope()
        return noise.noise.array.copy()


# Noise stack of a Salève field: components, each a noise in [-1, 1] times
# its contrast (log2), summed. Type -> {param: default}.
NOISE_TYPES = {
    "perlin": {"contrast": 1.0, "frequency": 3.0, "octaves": 4, "persistence": 0.5},
    "ridged": {"contrast": 1.0, "frequency": 3.0, "octaves": 4, "persistence": 0.5,
               "gain": 2.0},
    "worley": {"contrast": 1.0, "cells": 20},
    "gradient": {"contrast": 1.0, "azimuth": 0.0},
}


def _parse_noise(stack, name):
    """The components of a noise stack given as a JSON list (or a list), each
    completed with its type's defaults; unknown types / keys are refused."""
    import json

    if isinstance(stack, str):
        stack = json.loads(stack) if stack.strip() else []
    if not isinstance(stack, list):
        raise ValueError(f"{name}: expected a list of noise components")
    out = []
    for i, c in enumerate(stack):
        kind = c.get("type") if isinstance(c, dict) else None
        if kind not in NOISE_TYPES:
            raise ValueError(f"{name}[{i}]: unknown noise type {kind!r} "
                             f"(one of {sorted(NOISE_TYPES)})")
        extra = set(c) - set(NOISE_TYPES[kind]) - {"type"}
        if extra:
            raise ValueError(f"{name}[{i}] ({kind}): unknown keys {sorted(extra)}")
        out.append({**NOISE_TYPES[kind], **c})
    return out


def _rescale(v):
    """``v`` stretched to [-1, 1] (0 when flat)."""
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return v * 0.0
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def _noise_component(n, c, seed):
    """One component of a noise stack on an (n, n) grid, in [-1, 1], as a
    CuPy array. perlin: Perlin fBm (pyfastflow); ridged: Musgrave's ridged
    multifractal (offset 1) from single-octave Perlin; worley: distance to the
    nearest of `cells` random points (periodic); gradient: a ramp rising
    towards `azimuth` (degrees clockwise from north)."""
    import cupy as cp

    kind = c["type"]
    if kind == "perlin":
        v = _perlin_noise(n, n, 1.0, 1.0, c["frequency"], c["octaves"],
                          c["persistence"], seed).reshape(n, n)
        return v / cp.maximum(cp.abs(v).max(), 1e-12)
    if kind == "ridged":
        result = weight = None
        for o in range(max(int(c["octaves"]), 1)):
            v = _perlin_noise(n, n, 1.0, 1.0, c["frequency"] * 2.0 ** o, 1, 0.5,
                              seed + 31 * o).reshape(n, n)
            v = v / cp.maximum(cp.abs(v).max(), 1e-12)
            signal = (1.0 - cp.abs(v)) ** 2
            if result is None:
                result = signal
            else:
                signal = signal * weight
                result = result + signal * float(c["persistence"]) ** o
            weight = cp.clip(signal * float(c["gain"]), 0.0, 1.0)
        return _rescale(result)
    if kind == "worley":
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(seed)
        pts = rng.random((max(int(c["cells"]), 1), 2))
        g = (np.arange(n) + 0.5) / n
        xx, yy = np.meshgrid(g, g)
        d, _ = cKDTree(pts, boxsize=1.0).query(
            np.column_stack([xx.ravel(), yy.ravel()]), workers=-1)
        return _rescale(cp.asarray(d.reshape(n, n), dtype=cp.float32))
    # gradient; row 0 is north.
    a = np.deg2rad(float(c["azimuth"]))
    g = cp.arange(n, dtype=cp.float32) / max(n - 1, 1)
    return _rescale(np.sin(a) * g[None, :] + np.cos(a) * (1.0 - g[:, None]))


def _noise_field(n, base, stack, seed, name):
    """``base * 2 ** sum(contrast_i * noise_i)`` on an (n, n) grid, float32;
    the scalar ``base`` for an empty stack."""
    import cupy as cp

    comps = _parse_noise(stack, name)
    if not comps:
        return float(base)
    total = cp.zeros((n, n), dtype=cp.float32)
    for i, c in enumerate(comps):
        total += float(c["contrast"]) * _noise_component(n, c, seed + 97 * i)
    return cp.asnumpy(base * cp.exp2(total)).astype(np.float32)


_NOISE_DOC = ("Noise stack, a JSON list of components {type, contrast, ...}; the "
              "field is the mean times 2^(sum of contrast * noise), each noise in "
              "[-1, 1]; empty = uniform. Types and their keys: perlin (frequency, "
              "octaves, persistence), ridged (frequency, octaves, persistence, "
              "gain), worley (cells), gradient (azimuth, degrees from north).")
_UPLIFT_NOISE = '[{"type": "perlin", "contrast": 1.0, "frequency": 3.0, "octaves": 4, "persistence": 0.5}]'
_ERODIBILITY_NOISE = '[{"type": "perlin", "contrast": 1.0, "frequency": 5.0, "octaves": 4, "persistence": 0.5}]'


# The last Salève solve: {"key": parameters repr, "value": (z, uplift, erodibility)}.
_SALEVE_CACHE = {}


@process(
    id="pyfastflow.saleve_steady",
    label="Salève landscape",
    params=[
        # Grid
        Param("n", "int", default=1024, choices=[256, 512, 1024, 2048],
              doc="Cells per side (square grid)."),
        Param("dx", "float", default=50.0, min=0.0,
              doc="Cell size in metres."),
        Param("seed", "int", default=1, min=0,
              doc="Seed of the initial white noise and of the level jitter; "
                  "the uplift / erodibility fields use seed + 1000 / + 2000."),
        # Stream power
        Param("m", "float", default=0.45, min=0.0,
              doc="Drainage-area exponent of the stream-power law."),
        Param("uplift", "float", default=1e-3, min=0.0,
              doc="Uplift rate U (m/yr), mean of the uplift field."),
        Param("erodibility", "float", default=1e-5, min=0.0,
              doc="Erodibility K, mean of the erodibility field."),
        # Fields
        Param("uplift_noise", "string", default=_UPLIFT_NOISE,
              doc="Uplift field. " + _NOISE_DOC),
        Param("erodibility_noise", "string", default=_ERODIBILITY_NOISE,
              doc="Erodibility field. " + _NOISE_DOC),
        Param("return_fields", "bool", default=False,
              doc="Also return the uplift and erodibility fields."),
        # Hillslope
        Param("hillslope_erosion", "float", default=0.4, min=0.0,
              doc="Hillslope diffusivity D (m²/yr): adds D / (c A^h) to the "
                  "erosion term (Tzathas et al. 2024, Eqn. 26); 0 = off."),
        Param("hack_constant", "float", default=1.5, min=0.0,
              doc="Hack's law constant c (hillslope length c A^h)."),
        Param("hack_exponent", "float", default=0.6, min=0.0,
              doc="Hack's law exponent h."),
        # Thermal (talus)
        Param("thermal_erosion", "float", default=0.0, min=0.0,
              doc="Thermal (talus) coefficient kt (m/yr): where the slope exceeds Sc, "
                  "adds kt to the erosion term and kt Sc to the uplift (Eqns. 28-29); 0 = off."),
        Param("critical_slope", "float", default=0.5, min=0.0,
              doc="Critical slope Sc of thermal erosion (tan of the talus angle)."),
        # Network
        Param("topology", "string", default="D8", choices=TOPOLOGIES,
              doc="Flow neighbourhood."),
        Param("boundary", "string", default="periodic_EW", choices=BOUNDARIES,
              doc="Grid edges: normal (all edges are outlets) or periodic "
                  "east-west / north-south."),
        Param("local_minima", "string", default="cordonnier_carve", choices=LOCAL_MINIMA,
              doc="How local minima are resolved when routing flow."),
        Param("epsilon", "float", default=1e-3, min=0.0,
              doc="Epsilon of Salève's finite-time solver. SaleveProgram hands it "
                  "only to that solver, so it has no effect on this steady solve; "
                  "kept to match saleve_app."),
        Param("jitter", "float", default=10.0, min=0.0,
              doc="Random jitter added between grid levels."),
        Param("receiver_mode", "string", default="steepest", choices=RECEIVER_MODES,
              doc="Receiver choice: steepest, stochastic or slope-weighted fixed."),
        Param("receiver_seed", "int", default=0, min=0,
              doc="Seed of the stochastic receiver choice."),
        Param("slope_correction", "string", default="gradient", choices=SLOPE_CORRECTIONS,
              doc="gradient: adjust link travel time with the local downhill gradient."),
        Param("min_link_slope", "float", default=1e-6, min=0.0,
              doc="Minimum link slope."),
        Param("max_slope_correction", "float", default=100.0, min=1.0,
              doc="Maximum slope correction factor."),
        # Cliff optimisation
        Param("cliff_optimization", "bool", default=False,
              doc="Fixed-network cliff post-correction after each high-level solve."),
        Param("cliff_iterations", "int", default=50, min=0,
              doc="Cliff optimisation iterations."),
        Param("cliff_learning_rate", "float", default=0.01, min=0.0,
              doc="Cliff optimisation learning rate."),
        Param("cliff_river_weight", "float", default=1.0 / 3.0, min=0.0, max=1.0,
              doc="Weight of the river term in the cliff optimisation."),
        # Multigrid
        Param("levels", "int", default=5, min=1,
              doc="Multigrid levels."),
        Param("iterations", "int", default=24, min=1,
              doc="Multigrid iterations."),
        Param("relaxation", "float", default=0.25, min=0.0, max=1.0,
              doc="Multigrid relaxation."),
        # Multi-scale erosion post-process (Schott et al. 2024)
        Param("mse_stages", "int", default=0, min=0, max=len(MSE_PRESET),
              doc="Multi-scale erosion post-process (Schott et al. 2024, release "
                  "code port): number of stages of its preset, each erosion, "
                  "thermal then deposition steps, the resolution doubling before "
                  "every stage after the first; 0 = off."),
        Param("mse_k", "float", default=0.0005, min=0.0,
              doc="Multi-scale erosion: stream power coefficient k."),
        Param("mse_p_sa", "float", default=0.8, min=0.0,
              doc="Multi-scale erosion: drainage exponent of the stream power."),
        Param("mse_p_sl", "float", default=2.0, min=0.0,
              doc="Multi-scale erosion: slope exponent of the stream power."),
        Param("mse_dt", "float", default=1.0, min=0.0,
              doc="Multi-scale erosion: time step of the erosion steps."),
        Param("mse_max_spe", "float", default=10000.0, min=0.0,
              doc="Multi-scale erosion: cap of the stream power before k."),
        Param("mse_flow_p", "float", default=1.3, min=0.0,
              doc="Multi-scale erosion: exponent of the slope weights of the "
                  "multiple-flow split (erosion and deposition)."),
        Param("mse_eps", "float", default=0.00005, min=0.0,
              doc="Multi-scale erosion: thermal step, height moved per unstable "
                  "neighbour is eps * dx²."),
        Param("mse_noisified", "bool", default=True,
              doc="Multi-scale erosion: thermal talus angle drawn from simplex "
                  "noise between the noise min and max (else the fixed tangent)."),
        Param("mse_tan_angle", "float", default=0.57, min=0.0,
              doc="Multi-scale erosion: fixed tangent of the talus angle."),
        Param("mse_noise_min", "float", default=0.9, min=0.0,
              doc="Multi-scale erosion: lowest noisy talus tangent."),
        Param("mse_noise_max", "float", default=1.4, min=0.0,
              doc="Multi-scale erosion: highest noisy talus tangent."),
        Param("mse_noise_wavelength", "float", default=0.0023, min=0.0,
              doc="Multi-scale erosion: frequency (1/m) of the talus noise."),
        Param("mse_deposition_strength", "float", default=1.0, min=0.0,
              doc="Multi-scale erosion: deposition strength."),
    ],
    outputs=[
        Output("grid", "georaster", doc="The steady-state surface."),
        Output("uplift_field", "georaster", optional=True,
               doc="Uplift rate (m/yr), with return_fields."),
        Output("erodibility_field", "georaster", optional=True,
               doc="Erodibility K, with return_fields."),
    ],
    impl="library",
)
def saleve_steady(n=1024, dx=50.0, seed=1, m=0.45, uplift=1e-3, erodibility=1e-5,
                  uplift_noise=_UPLIFT_NOISE, erodibility_noise=_ERODIBILITY_NOISE,
                  return_fields=False,
                  hillslope_erosion=0.4, hack_constant=1.5, hack_exponent=0.6,
                  thermal_erosion=0.0, critical_slope=0.5,
                  topology="D8", boundary="periodic_EW", local_minima="cordonnier_carve",
                  epsilon=1e-3, jitter=10.0, receiver_mode="steepest", receiver_seed=0,
                  slope_correction="gradient", min_link_slope=1e-6,
                  max_slope_correction=100.0,
                  cliff_optimization=False, cliff_iterations=50,
                  cliff_learning_rate=0.01, cliff_river_weight=1.0 / 3.0,
                  levels=5, iterations=24, relaxation=0.25,
                  mse_stages=0, mse_k=0.0005, mse_p_sa=0.8, mse_p_sl=2.0, mse_dt=1.0,
                  mse_max_spe=10000.0, mse_flow_p=1.3, mse_eps=0.00005,
                  mse_noisified=True, mse_tan_angle=0.57, mse_noise_min=0.9,
                  mse_noise_max=1.4, mse_noise_wavelength=0.0023,
                  mse_deposition_strength=1.0):
    """Generate a steady-state landscape with pyfastflow's Salève stream-power solver.

    Uplift and erodibility fields from noise stacks over a white-noise initial
    surface, one steady multigrid solve on the GPU (CuPy). Square grid of n x n
    cells of dx metres, origin (0, 0), no CRS. The last solve is kept: a call
    differing only in the post-process (``mse_*``) or ``return_fields`` reuses
    it instead of solving again.
    """
    key = repr(sorted((k, v) for k, v in locals().items()
                      if not k.startswith("mse_") and k != "return_fields"))
    n = int(n)
    seed = int(seed)
    backend = _backend()
    phases = 1 + (1 if int(mse_stages) else 0)
    progress.report(0, phases, "steady-state landscape")
    if _SALEVE_CACHE.get("key") == key:
        z, uplift_field, erodibility_field = _SALEVE_CACHE["value"]
        z = z.copy()
    else:
        uplift_field = _noise_field(n, uplift, uplift_noise, seed + 1000, "uplift_noise")
        erodibility_field = _noise_field(n, erodibility, erodibility_noise, seed + 2000,
                                         "erodibility_noise")
        z0 = np.random.default_rng(seed).random((n, n), dtype=np.float32) * 0.01
        if boundary != "periodic_NS":
            z0[[0, -1], :] = 0.0
        if boundary != "periodic_EW":
            z0[:, [0, -1]] = 0.0
        options = dict(
            nx=n, ny=n, dx=dx, m=m,
            uplift=uplift_field, erodibility=erodibility_field,
            local_minima=local_minima, epsilon=epsilon,
            topology=topology, boundary=boundary, outlet="edge",
            nodata=False, jitter=jitter, seed=seed,
            receiver_mode=receiver_mode, receiver_seed=int(receiver_seed),
            slope_correction=slope_correction,
            min_link_slope=min_link_slope,
            max_slope_correction=max_slope_correction,
            thermal_erosion=thermal_erosion, critical_slope=critical_slope,
            hillslope_erosion=hillslope_erosion,
            hack_constant=hack_constant, hack_exponent=hack_exponent,
            cliff_optimization=bool(cliff_optimization),
            cliff_iterations=int(cliff_iterations),
            cliff_learning_rate=cliff_learning_rate,
            cliff_river_weight=cliff_river_weight,
        )
        with SaleveProgram(backend, **options) as saleve:
            saleve.z.from_numpy(z0)
            saleve.initialize()
            saleve.run_multigrid_steady(levels=int(levels), iterations=int(iterations),
                                        relaxation=relaxation)
            z = saleve.z.to_numpy()
        _SALEVE_CACHE.update(key=key, value=(z.copy(), uplift_field, erodibility_field))

    cell = float(dx)
    stages = int(mse_stages)
    if stages:
        import cupy as cp

        progress.report(1, phases, f"multiscale erosion ({stages} stage{'s' if stages > 1 else ''})")
        z, cell = mse_amplify(
            backend, z, cell, stages=stages, k=mse_k, p_sa=mse_p_sa, p_sl=mse_p_sl,
            dt=mse_dt, max_spe=mse_max_spe, flow_p=mse_flow_p, eps=mse_eps,
            noisified=int(bool(mse_noisified)), tan_angle=mse_tan_angle,
            noise_min=mse_noise_min, noise_max=mse_noise_max,
            noise_wavelength=mse_noise_wavelength,
            deposition_strength=mse_deposition_strength)
        z = cp.asnumpy(z)
    # The post-process resamples the same vertices' extent: the first cell
    # centre stays at dx / 2.
    origin = 0.5 * (float(dx) - cell)
    m = z.shape[0]

    def raster(v):
        return GeoRaster(z=v, cell_size=cell, x_min=origin, y_min=origin)

    def field(f):
        if np.isscalar(f):
            return np.full((m, m), np.float32(f))
        f = np.array(f, dtype=np.float32)  # a copy: the cache keeps its own
        for _ in range(stages - 1):
            import cupy as cp

            f = cp.asnumpy(mse_upsample_x2(f))
        return f

    progress.report(phases, phases, "done")
    out = {"grid": raster(z)}
    if return_fields:
        for key, f in (("uplift_field", uplift_field), ("erodibility_field", erodibility_field)):
            out[key] = raster(field(f))
    return out


@process(
    id="pyfastflow.perlin_surface",
    label="Perlin surface",
    params=[
        Param("nx", "int", default=1024, min=8, doc="Columns."),
        Param("ny", "int", default=1024, min=8, doc="Rows."),
        Param("dx", "float", default=30.0, min=0.0, doc="Cell size in metres."),
        Param("amplitude", "float", default=300.0, min=0.0,
              doc="Noise amplitude (m)."),
        Param("frequency", "float", default=5.0, min=0.5,
              doc="Features per domain (rounded to a half integer)."),
        Param("octaves", "int", default=6, min=1, doc="Octaves."),
        Param("persistence", "float", default=0.5, min=0.0, max=1.0,
              doc="Amplitude kept per octave."),
        Param("seed", "int", default=42, min=0, doc="Seed."),
        Param("slope", "float", default=2e-3, min=0.0,
              doc="Regional slope towards the south (m/m); 0 = none."),
    ],
    outputs=[Output("grid", "georaster")],
    impl="library",
)
def perlin_surface(nx=1024, ny=1024, dx=30.0, amplitude=300.0, frequency=5.0,
                   octaves=6, persistence=0.5, seed=42, slope=2e-3):
    """Perlin noise surface (GPU, CuPy) with an optional regional slope down to
    the south. nx x ny cells of dx metres, origin (0, 0), no CRS."""
    import cupy as cp

    z = cp.asnumpy(_perlin_noise(nx, ny, dx, amplitude, frequency, octaves,
                                 persistence, seed, slope))
    return GeoRaster(z=z.reshape(int(ny), int(nx)), cell_size=float(dx),
                     x_min=0.0, y_min=0.0)


# Grid edges whose cells drain out of the domain, by outlet_edges choice.
_OUTLET_SIDES = {
    "all": "nsew", "north": "n", "south": "s", "east": "e", "west": "w",
    "north_south": "ns", "east_west": "ew", "none": "",
}


def _metric_cell_size(grid, who="GOLEM"):
    """The grid's cell size, refused for a geographic CRS (the physics needs
    metres)."""
    if grid.crs_wkt or grid.epsg:
        from pyproj import CRS

        crs = CRS.from_wkt(grid.crs_wkt) if grid.crs_wkt else CRS.from_epsg(grid.epsg)
        if crs.is_geographic:
            raise ValueError(f"{who} needs a projected grid (cell size in metres); "
                             "this one is in geographic coordinates")
    return grid.cell_size


def _outlet_mask(valid, edges, boundary, nodata_outlets, who="GOLEM"):
    """uint8 mask of the cells that drain out: the chosen grid edges (but not
    the periodic ones) and, with ``nodata_outlets``, valid cells touching
    nodata."""
    from scipy.ndimage import binary_dilation

    sides = set(_OUTLET_SIDES[edges])
    if boundary == "periodic_NS":
        sides -= {"n", "s"}
    if boundary == "periodic_EW":
        sides -= {"e", "w"}
    mask = np.zeros(valid.shape, dtype=bool)
    if "n" in sides:
        mask[0, :] = True
    if "s" in sides:
        mask[-1, :] = True
    if "w" in sides:
        mask[:, 0] = True
    if "e" in sides:
        mask[:, -1] = True
    if nodata_outlets and not valid.all():
        mask |= binary_dilation(~valid, structure=np.ones((3, 3), dtype=bool))
    mask &= valid
    if not mask.any():
        raise ValueError(f"{who}: no outlet cell (choose outlet edges, or let "
                         "cells next to nodata drain out)")
    return mask.astype(np.uint8)


def _field(field, scalar, shape, name, who="GOLEM"):
    """A spatial parameter: the grid's values (NaN -> 0) when given, else the
    scalar."""
    if field is None:
        return float(scalar)
    if field.z.shape != shape:
        raise ValueError(f"{who}: {name} is {field.z.shape[0]} x {field.z.shape[1]} "
                         f"cells, the DEM {shape[0]} x {shape[1]}")
    return np.nan_to_num(field.z, nan=0.0).astype(np.float32)


def _like(grid, z):
    """``z`` on the georeference of ``grid``."""
    return GeoRaster(z=z, cell_size=grid.cell_size, x_min=grid.x_min,
                     y_min=grid.y_min, epsg=grid.epsg, crs_wkt=grid.crs_wkt)


@process(
    id="pyfastflow.golem_nosed",
    label="GOLEM landscape evolution (no sediment)",
    inputs=[
        Port("dem", "georaster",
             doc="Surface to evolve (projected, metres); nodata cells stay out."),
        Port("uplift_field", "georaster", optional=True,
             doc="Uplift rate per cell (m/yr), replaces Uplift; same grid as the DEM."),
        Port("erodibility_field", "georaster", optional=True,
             doc="Erodibility per cell, replaces Erodibility; same grid as the DEM."),
    ],
    params=[
        Param("steps", "int", default=100, min=0, doc="Time steps to run."),
        Param("dt", "float", default=5e4, min=0.0, doc="Time step (yr)."),
        Param("uplift", "float", default=1e-3, min=0.0,
              doc="Uplift rate U (m/yr), when no uplift field is given."),
        Param("erodibility", "float", default=1e-5, min=0.0,
              doc="Erodibility K, when no erodibility field is given."),
        Param("m", "float", default=0.4, min=0.0,
              doc="Drainage-area exponent of the stream-power law."),
        Param("n", "float", default=1.6, min=0.0,
              doc="Slope exponent of the stream-power law."),
        Param("hillslope_diffusivity", "float", default=3e-2, min=0.0,
              doc="Linear hillslope diffusivity D (m²/yr); 0 = off."),
        Param("slope_floor", "float", default=1e-6, min=0.0,
              doc="Smallest slope used by the incision solve."),
        Param("newton_iterations", "int", default=8, min=1,
              doc="Newton iterations of the implicit incision solve."),
        Param("diffusion_iterations", "int", default=20, min=1,
              doc="Iterations of the implicit diffusion solve."),
        Param("topology", "string", default="D8", choices=["D4", "D8"],
              doc="Flow neighbourhood."),
        Param("boundary", "string", default="normal",
              choices=["normal", "periodic_EW", "periodic_NS"],
              doc="Grid edges: normal, or periodic east-west / north-south (a periodic "
                  "edge never drains out)."),
        Param("outlet_edges", "string", default="all", choices=list(_OUTLET_SIDES),
              doc="Grid edges whose cells drain out of the domain."),
        Param("nodata_outlets", "bool", default=True,
              doc="Valid cells touching nodata also drain out."),
        Param("local_minima", "string", default="cordonnier_carve",
              choices=["none", "reconstruct_epsilon", "cordonnier_carve",
                       "cordonnier_jump"],
              doc="How local minima are resolved when routing flow."),
        Param("return_drainage_area", "bool", default=False,
              doc="Also return the drainage area (m²) of the last step's routing."),
        Param("return_erosion_rate", "bool", default=False,
              doc="Also return the fluvial erosion rate (m/yr) of the last step."),
    ],
    outputs=[
        Output("grid", "georaster", doc="The evolved surface."),
        Output("drainage_area", "georaster", optional=True,
               doc="Drainage area (m²), routed on the surface before the last step's "
                   "incision."),
        Output("erosion_rate", "georaster", optional=True,
               doc="Fluvial erosion rate of the last step (m/yr); 0 without steps."),
    ],
    impl="library",
)
def golem_nosed(dem, uplift_field=None, erodibility_field=None, steps=100, dt=5e4,
                uplift=1e-3, erodibility=2e-5, m=0.4, n=1.6,
                hillslope_diffusivity=3e-2, slope_floor=1e-6, newton_iterations=8,
                diffusion_iterations=20, topology="D8", boundary="normal",
                outlet_edges="all", nodata_outlets=True,
                local_minima="cordonnier_carve", return_drainage_area=False,
                return_erosion_rate=False):
    """Run ``steps`` time steps of pyfastflow's experimental sediment-free GOLEM
    (GPU, CuPy) from ``dem``.

    Each step: uplift, SFD routing and accumulation, implicit stream-power
    incision, implicit linear hillslope diffusion. Outlet cells keep their
    elevation. Nodata cells are left out and stay nodata.
    """
    args = dict(locals())
    inputs = {k: args.pop(k) for k in ("dem", "uplift_field", "erodibility_field")}
    return run_once(GolemRun, inputs, args)


@runner("pyfastflow.golem_nosed", config=(
    "dt", "uplift", "erodibility", "m", "n", "hillslope_diffusivity", "slope_floor",
    "newton_iterations", "diffusion_iterations", "topology", "boundary", "outlet_edges",
    "nodata_outlets", "local_minima"))
class GolemRun(Runner):
    """Live GOLEM: one program across advances, each running ``steps`` more
    steps from where the last one stopped. Its parameters are program
    constants: a host changing one opens a new runner (from the current
    surface, which is the whole state)."""

    def __init__(self, inputs, p):
        dem = self.dem = inputs["dem"]
        dx = _metric_cell_size(dem)
        shape = self.shape = dem.z.shape
        valid = self.valid = dem.valid
        if not valid.any():
            raise ValueError("GOLEM: the DEM has no valid cell")
        has_nodata = not valid.all()
        options = dict(
            nx=shape[1], ny=shape[0], dx=dx,
            topology=p["topology"], boundary=p["boundary"], outlet="mask",
            nodata=has_nodata, local_minima=p["local_minima"],
            uplift=_field(inputs.get("uplift_field"), p["uplift"], shape, "uplift_field"),
            erodibility=_field(inputs.get("erodibility_field"), p["erodibility"], shape,
                               "erodibility_field"),
            hillslope_diffusivity=p["hillslope_diffusivity"], dt=p["dt"], m=p["m"],
            n=p["n"], slope_floor=p["slope_floor"],
            newton_iterations=int(p["newton_iterations"]),
            diffusion_iterations=int(p["diffusion_iterations"]),
        )
        golem = self.golem = GolemNoSedProgram(_backend(), **options)
        try:
            golem.z.from_numpy(np.where(valid, dem.z, 0.0).astype(np.float32))
            golem.outlet_mask.from_numpy(_outlet_mask(valid, p["outlet_edges"], p["boundary"],
                                                      p["nodata_outlets"]))
            if has_nodata:
                golem.nodata_mask.from_numpy((~valid).astype(np.uint8))
            golem.initialize()
        except Exception:
            golem.close()
            raise
        self.ran = 0

    def advance(self, p):
        golem, dem, valid = self.golem, self.dem, self.valid
        steps = int(p["steps"])
        if steps > 0:
            progress.steps(golem.run_n_step, steps, "GOLEM steps", sync=_gpu_sync)
            self.ran += steps
        elif p["return_drainage_area"]:
            golem.route_and_accumulate()
        out = {"grid": _like(dem, np.where(valid, golem.z.to_numpy(), np.nan))}
        if p["return_drainage_area"]:
            out["drainage_area"] = _like(
                dem, np.where(valid, golem.drainage_area.to_numpy(), np.nan))
        if p["return_erosion_rate"]:
            rate = golem.erosion_rate.to_numpy() if self.ran > 0 else np.zeros(self.shape)
            out["erosion_rate"] = _like(dem, np.where(valid, rate, np.nan))
        return out

    def coordinate(self):
        return {"step": self.ran}

    def close(self):
        self.golem.close()


# ---------------------------------------------------------------------------
# GraphFlood
# ---------------------------------------------------------------------------

#: Rainfall in mm/h -> m/s (the programs' unit).
_MM_PER_H = 1.0 / 3.6e6

#: Program and pipeline run by each solver choice.
_GF_SOLVERS = {
    "analytical": ("relax", "run"),
    "explicit": ("vanilla", "run"),
    "transient": ("vanilla", "run_transient"),
}
_GF_SOLVER_LABELS = ["Analytical (relaxation)", "Explicit (dt)", "Transient (local, dt)"]

#: How the water starts when no h_init is given (see _flood_start).
_INITIAL_WATER = ["dry", "fill_lakes", "flatten_lakes"]
_INITIAL_WATER_LABELS = ["Dry", "Lakes filled with water", "Lakes flattened (static)"]

_GF_LOCAL_MINIMA = ["carve_cordonnier", "fill_cordonnier", "rank_cordonnier",
                    "reconstruct_epsilon"]

_FLOOD_RETURNS = ("discharge", "outflow", "imbalance", "report")


def _flood_grid(dem, who):
    """Cell size, valid mask and the solver's z (nodata cells at 0) of ``dem``."""
    dx = _metric_cell_size(dem, who)
    valid = dem.valid
    if not valid.any():
        raise ValueError(f"{who}: the DEM has no valid cell")
    return dx, valid, np.where(valid, dem.z, 0.0).astype(np.float32)


def _rain(field, rate_mm_h, shape, who):
    """Rainfall in m/s: the field (mm/h, NaN -> 0) when given, else the rate."""
    rain = _field(field, rate_mm_h, shape, "precipitation_field", who)
    if isinstance(rain, float):
        return rain * _MM_PER_H
    return (rain * _MM_PER_H).astype(np.float32)


def _flood_start(prog, z, valid, initial_water, h_init, who):
    """Writes z and the initial depth into ``prog``; returns the lake depth
    (flattened depressions, 0 elsewhere) that h_init and the outputs are
    shifted by.

    - ``dry``: h = 0.
    - ``fill_lakes``: depressions filled with water (``initialize_h_from_fill``).
    - ``flatten_lakes``: depressions become topography (``fill_topography``:
      z replaced by its reconstructed epsilon fill, h = 0); the solver keeps
      that lake water static.
    Depths crossing the interface are always above the input DEM, so with
    ``flatten_lakes`` the lake depth is added to the output and taken off
    ``h_init``.
    """
    prog.z.from_numpy(z)
    lake = np.zeros(z.shape, dtype=np.float32)
    if initial_water == "flatten_lakes":
        prog.fill_topography()
        lake = np.where(valid, prog.z.to_numpy() - z, 0.0).astype(np.float32)
    elif initial_water == "fill_lakes":
        prog.initialize_h_from_fill()
    elif initial_water == "dry":
        prog.reset_h()
    else:
        raise ValueError(f"{who}: unknown initial_water {initial_water!r}")
    if h_init is not None:
        if h_init.z.shape != z.shape:
            raise ValueError(f"{who}: h_init is {h_init.z.shape[0]} x {h_init.z.shape[1]} "
                             f"cells, the DEM {z.shape[0]} x {z.shape[1]}")
        h = np.nan_to_num(h_init.z, nan=0.0) - lake
        prog.h.from_numpy(np.where(valid, np.maximum(h, 0.0), 0.0).astype(np.float32))
    return lake


def _flood_masks(prog, valid, outlet_edges, boundary, nodata_outlets, who):
    """Sets the outlet (and nodata) masks; returns the cupy mask of the cells
    the residual is taken over (valid, not outlets)."""
    import cupy as cp

    outlets = _outlet_mask(valid, outlet_edges, boundary, nodata_outlets, who)
    prog.outlet_mask.from_numpy(outlets)
    if not valid.all():
        prog.nodata_mask.from_numpy((~valid).astype(np.uint8))
    return cp.asarray(valid & (outlets == 0))


def _flood_outputs(dem, valid, lake, h, qi, qo, returns, report):
    """The process outputs: water depth above the input DEM, and the extras
    asked for in ``returns``."""
    out = {"water_depth": _like(dem, np.where(valid, h + lake, np.nan))}
    if "discharge" in returns:
        out["discharge"] = _like(dem, np.where(valid, qi, np.nan))
    if "outflow" in returns:
        out["outflow"] = _like(dem, np.where(valid, qo, np.nan))
    if "imbalance" in returns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(qi > 0.0, qo / qi - 1.0, np.nan)
        out["imbalance"] = _like(dem, np.where(valid, ratio, np.nan))
    if "report" in returns:
        out["report"] = report
    return out


def _flood_returns(**flags):
    return {name for name in _FLOOD_RETURNS if flags[f"return_{name}"]}


def _flood_options(dem_shape, dx, valid, topology, boundary, local_minima):
    """Constructor config shared by the GraphFlood programs."""
    return dict(nx=dem_shape[1], ny=dem_shape[0], dx=dx, topology=topology,
                boundary=boundary, outlet="mask", nodata=not valid.all(),
                mfd_local_minima=local_minima)


def _flood_set(prog, **values):
    """Sets scalar runtime params after construction (a constructor scalar
    would be a constant)."""
    for name, value in values.items():
        getattr(prog, name).set(value)


def _split_fields(**values):
    """(arrays, scalars): array values must go to the constructor (they make
    the param a field), scalar ones are set after it (see _flood_set)."""
    arrays = {k: v for k, v in values.items() if isinstance(v, np.ndarray)}
    return arrays, {k: v for k, v in values.items() if k not in arrays}


def _flood_common_params():
    """Params shared by both GraphFlood processes (rain, friction, start,
    warm-up, grid, outlets)."""
    return [
        Param("rainfall", "float", default=50.0, min=0.0,
              doc="Rainfall rate (mm/h), when no precipitation field is given."),
        Param("manning", "float", default=0.033, min=0.0,
              doc="Manning roughness n (s/m^(1/3)), when no Manning field is given."),
        Param("friction_exponent", "float", default=2.0 / 3.0, min=0.0,
              doc="Exponent of the hydraulic radius in the friction law (2/3: Manning)."),
        Param("initial_water", "string", default="dry", choices=_INITIAL_WATER,
              choice_labels=_INITIAL_WATER_LABELS,
              doc="Start without h_init: dry; depressions filled with water; or depressions "
                  "flattened into the topography (their water is static, still counted in "
                  "the output depth)."),
        Param("warmup_steps", "int", default=5, min=0,
              doc="Capped analytical warm-up passes run first, filling the hillslopes "
                  "(analytical solver and particles; 0: none)."),
        Param("warmup_relaxation", "float", default=0.6, min=0.0, max=1.0,
              doc="Relaxation of the warm-up passes."),
        Param("depth_cap", "float", default=0.5, min=0.0,
              doc="Warm-up: cap (m) of the target and of the depth."),
        Param("carve_slope_min", "float", default=1e-4, min=0.0,
              doc="Smallest slope through carved depressions."),
        Param("topology", "string", default="D8", choices=["D4", "D8"],
              doc="Flow neighbourhood."),
        Param("boundary", "string", default="normal",
              choices=["normal", "periodic_EW", "periodic_NS"],
              doc="Grid edges: normal, or periodic east-west / north-south (a periodic "
                  "edge never lets water out)."),
        Param("local_minima", "string", default="carve_cordonnier", choices=_GF_LOCAL_MINIMA,
              doc="How depressions of the water surface are routed through."),
        Param("outlet_edges", "string", default="all", choices=list(_OUTLET_SIDES),
              doc="Grid edges whose cells let water out of the domain."),
        Param("nodata_outlets", "bool", default=True,
              doc="Valid cells touching nodata also let water out."),
    ]


def _flood_common_outputs():
    return [
        Output("water_depth", "georaster",
               doc="Water depth (m) above the input DEM; NaN on nodata."),
        Output("discharge", "georaster", optional=True,
               doc="Discharge entering each cell, Qin (m³/s)."),
        Output("outflow", "georaster", optional=True,
               doc="Discharge leaving each cell by the friction law at the final depth, "
                   "Qout (m³/s)."),
        Output("imbalance", "georaster", optional=True,
               doc="Qout / Qin - 1 per cell: 0 at steady state; NaN where Qin = 0."),
    ]


#: Metrics the stop rule can watch (keys of the programs' ``convergence()``).
_CONV_METRICS = ["dh_p99", "dh_p90", "dh_p50", "dh_max", "dh_mean", "residual"]

#: Report ``stop`` column: why the run stopped at that row.
_STOP_CODES = {None: 0, "tolerance": 1, "plateau": 2}


def _convergence_params(unit, check_default):
    """Params of the convergence checks and the stop rule; ``unit``: what
    is counted (steps, launches)."""
    return [
        Param("stop", "string", default="count", choices=["count", "converged"],
              choice_labels=["Fixed count", "Until converged (count is the maximum)"],
              doc=f"count: run all the {unit}; converged: stop earlier once the "
                  "convergence metric is below the tolerance or stops improving."),
        Param("check_every", "int", default=check_default, min=0,
              doc=f"Convergence check (one report row) every this many {unit}; 0: one "
                  "check at the end (fixed count only). Each check is one pass over the "
                  "grid."),
        Param("convergence_metric", "string", default="dh_p99", choices=_CONV_METRICS,
              doc="Watched metric: a quantile (or max / mean) of |dh|, the depth change "
                  "(m) that would pass each checked cell's discharge against its "
                  "receiver's head; or the discharge residual sum|Q - Qout| / sum Q."),
        Param("convergence_tol", "float", default=1e-3, min=0.0,
              doc="Stop once the metric is below this (m for dh, a ratio for residual)."),
        Param("convergence_window", "int", default=10, min=1,
              doc="Plateau: stop once the best of the last this many checks is not "
                  "lower than (1 - eps) x the best before them."),
        Param("convergence_eps", "float", default=0.01, min=0.0, max=1.0,
              doc="Plateau: smallest relative improvement over the window."),
        Param("convergence_percentile", "float", default=90.0, min=0.0, max=100.0,
              doc="Checked cells: valid non-outlet cells whose discharge is at or above "
                  "this percentile of the positive discharge (set at the first check)."),
        Param("convergence_memory", "float", default=0.8, min=0.0, max=1.0,
              doc="Drift metrics: weight of the past in the averages of the depth change "
                  "between checks."),
    ]


class _Checks:
    """The convergence checks of a run (across a runner's advances): report
    rows (count, every metric of ``convergence()``, extras, ``stop``) and the
    optional stop rule, whose history survives a change of its settings."""

    def __init__(self, unit, who):
        self.unit, self.who = unit, who
        self.enabled = ConvergenceRule is not None
        self.rows, self.last, self.rule, self.metric = {}, None, None, "dh_p99"
        self._settings = None

    def configure(self, prog, p):
        """Applies the convergence params of ``p`` (see _convergence_params)."""
        who, stop, metric = self.who, p["stop"], p["convergence_metric"]
        if stop not in ("count", "converged"):
            raise ValueError(f"{who}: unknown stop {stop!r}")
        if metric not in _CONV_METRICS:
            raise ValueError(f"{who}: unknown convergence_metric {metric!r}")
        if stop == "converged" and int(p["check_every"]) < 1:
            raise ValueError(f"{who}: stopping on convergence needs check_every >= 1")
        if stop == "converged" and not self.enabled:
            raise ValueError(f"{who}: this pyfastflow has no convergence checks")
        if self.enabled:
            _flood_set(prog, convergence_percentile=float(p["convergence_percentile"]),
                       convergence_memory=float(p["convergence_memory"]))
        settings = (stop, metric, float(p["convergence_tol"]), int(p["convergence_window"]),
                    float(p["convergence_eps"]))
        if settings != self._settings:
            history = self.rule.values if self.rule is not None and metric == self.metric else []
            self.rule = (ConvergenceRule(tol=settings[2], metric=metric, window=settings[3],
                                         eps=settings[4]) if stop == "converged" else None)
            if self.rule is not None:
                self.rule.values = list(history)
            self.metric, self._settings = metric, settings

    def check(self, prog, count, **extra):
        """One check after ``count`` units; True when the rule stops the run."""
        metrics = ({k: float(v) for k, v in prog.convergence().items()}
                   if self.enabled else {})
        reason = None
        if self.rule is not None and not math.isnan(metrics[self.metric]):
            reason = self.rule.update(metrics)
        row = {self.unit: float(count), **metrics, **extra, "stop": _STOP_CODES[reason]}
        for k, v in row.items():
            self.rows.setdefault(k, []).append(float(v))
        self.last = metrics
        return reason is not None

    def phase(self, text):
        """``text``, plus the watched metric of the last check."""
        if not self.last or math.isnan(self.last.get(self.metric, math.nan)):
            return text
        v = self.last[self.metric]
        if self.metric.startswith("dh"):
            shown = f"{v * 1000:.3g} mm" if v < 1.0 else f"{v:.3g} m"
        else:
            shown = f"{v:.3g}"
        return f"{text}  ·  {self.metric.replace('_', ' ')} {shown}"

    def table(self, title, units):
        return DataTable({k: np.asarray(v, dtype=np.float64) for k, v in self.rows.items()},
                         units={k: u for k, u in units.items() if k in self.rows},
                         roles={self.unit: "x"}, title=title)


#: Units of the report columns.
_CONV_UNITS = {"dh_mean": "m", "dh_p50": "m", "dh_p90": "m", "dh_p99": "m", "dh_max": "m",
               "h_max": "m", "time": "s"}


@process(
    id="pyfastflow.graphflood",
    label="GraphFlood (water depth and discharge)",
    inputs=[
        Port("dem", "georaster",
             doc="Topography (projected, metres); nodata cells stay out."),
        Port("h_init", "georaster", optional=True,
             doc="Water depth (m) above the DEM to start from (e.g. a previous run's "
                 "water_depth); replaces initial_water's depth. Same grid as the DEM."),
        Port("precipitation_field", "georaster", optional=True,
             doc="Rainfall per cell (mm/h), replaces Rainfall; same grid as the DEM."),
        Port("manning_field", "georaster", optional=True,
             doc="Manning n per cell (e.g. from land cover), replaces Manning; same grid "
                 "as the DEM."),
    ],
    params=[
        Param("solver", "string", default="analytical", choices=list(_GF_SOLVERS),
              choice_labels=_GF_SOLVER_LABELS,
              doc="analytical: relaxes each cell towards the friction depth of its routed "
                  "discharge (GraphFloodRelax); explicit: h += (Qin - Qout) dt / dx² with "
                  "the routed discharge (GraphFloodVanilla); transient: local conservative "
                  "transport on the raw water surface, no depression routing."),
        Param("steps", "int", default=300, min=1, doc="Steps to run."),
        *_flood_common_params(),
        Param("analytical_relaxation", "float", default=0.1, min=0.0, max=1.0,
              doc="Analytical solver: fraction of the way to the target depth per step."),
        Param("analytical_solver", "string", default="bottom_up",
              choices=["bottom_up", "local"],
              doc="Analytical solver: bottom_up (receivers first, each depth solved "
                  "against its receiver's head) or local (friction depth at the frozen "
                  "steepest slope)."),
        Param("dt", "float", default=1e-3, min=0.0,
              doc="Explicit and transient solvers: time step (s)."),
        *_convergence_params("steps", 25),
        Param("return_discharge", "bool", default=False, doc="Also return Qin."),
        Param("return_outflow", "bool", default=False, doc="Also return Qout."),
        Param("return_imbalance", "bool", default=False, doc="Also return Qout / Qin - 1."),
        Param("return_report", "bool", default=False,
              doc="Also return the report table (one row per convergence check)."),
    ],
    outputs=[
        *_flood_common_outputs(),
        Output("report", "datatable", optional=True,
               doc="Per convergence check: step, the checker's metrics (cells, dh mean / "
                   "p50 / p90 / p99 / max, fill_bias, residual, residual_signed, drift, "
                   "drift_mean, flips), max depth, model time (explicit, transient) and "
                   "stop (1: tolerance reached, 2: plateau, else 0)."),
    ],
    impl="library",
)
def graphflood(dem, h_init=None, precipitation_field=None, manning_field=None,
               solver="analytical", steps=300, rainfall=50.0, manning=0.033,
               friction_exponent=2.0 / 3.0, initial_water="dry", warmup_steps=5,
               warmup_relaxation=0.6, depth_cap=0.5, carve_slope_min=1e-4,
               topology="D8", boundary="normal", local_minima="carve_cordonnier",
               outlet_edges="all", nodata_outlets=True, analytical_relaxation=0.1,
               analytical_solver="bottom_up", dt=1e-3, stop="count", check_every=25,
               convergence_metric="dh_p99", convergence_tol=1e-3, convergence_window=10,
               convergence_eps=0.01, convergence_percentile=90.0, convergence_memory=0.8,
               return_discharge=False, return_outflow=False, return_imbalance=False,
               return_report=False):
    """Run ``steps`` steps of pyfastflow's GraphFlood (GPU, CuPy) on ``dem``.

    Analytical and explicit steps route the rain over the water surface z + h
    (multiple flow, depressions resolved by ``local_minima``) and update the
    depth; transient steps move water locally. Outlet cells pass their
    discharge out. Nodata cells are left out and stay NaN.
    """
    args = dict(locals())
    inputs = {k: args.pop(k) for k in ("dem", "h_init", "precipitation_field", "manning_field")}
    return run_once(GraphFloodRun, inputs, args)


def _make_counter(run, owner, attr, dt=None):
    """``run`` that also adds the steps it ran to ``owner.attr`` (and the
    model time to ``owner.time`` when ``dt()`` is given), so a chunk cancelled
    half way still counts what it did."""
    def counted(k):
        run(k)
        setattr(owner, attr, getattr(owner, attr) + k)
        if dt is not None:
            owner.time += k * dt()
    return counted


@runner("pyfastflow.graphflood", config=(
    "solver", "analytical_solver", "topology", "boundary", "local_minima", "outlet_edges",
    "nodata_outlets", "initial_water", "warmup_steps", "warmup_relaxation", "depth_cap"))
class GraphFloodRun(Runner):
    """Live GraphFlood: one program across advances. Each advance runs
    ``steps`` more steps (the warm-up before the first); counts, report rows
    and the stop rule continue from the previous advance."""

    who = "GraphFlood"

    def __init__(self, inputs, p):
        who = self.who
        solver = p["solver"]
        if solver not in _GF_SOLVERS:
            raise ValueError(f"{who}: unknown solver {solver!r}")
        self.dem = inputs["dem"]
        self.dx, self.valid, z = _flood_grid(self.dem, who)
        shape = self.shape = z.shape
        self.kind, pipeline = _GF_SOLVERS[solver]
        self.solver = solver
        options = _flood_options(shape, self.dx, self.valid, p["topology"], p["boundary"],
                                 p["local_minima"])
        fields, _ = _split_fields(
            precipitation=_rain(inputs.get("precipitation_field"), p["rainfall"], shape, who),
            friction_coefficient=_field(inputs.get("manning_field"), p["manning"], shape,
                                        "manning_field", who))
        self.fields = set(fields)
        if self.kind == "relax":
            prog = GraphFloodRelax(_backend(), analytical_solver=p["analytical_solver"],
                                   **options, **fields)
        else:
            prog = GraphFloodVanilla(_backend(), **options, **fields)
        self.flood = prog
        try:
            if self.kind == "relax":
                _flood_set(prog, warmup_relaxation=float(p["warmup_relaxation"]),
                           depth_cap=float(p["depth_cap"]))
            _flood_masks(prog, self.valid, p["outlet_edges"], p["boundary"],
                         p["nodata_outlets"], who)
            self.lake = _flood_start(prog, z, self.valid, p["initial_water"],
                                     inputs.get("h_init"), who)
        except Exception:
            prog.close()
            raise
        self.checks = _Checks("step", who)
        self.warm_pending = int(p["warmup_steps"]) if self.kind == "relax" else 0
        self.done, self.warmed, self.time = 0, 0, 0.0
        self.dt = float(p["dt"])
        self.run = _make_counter(getattr(prog, pipeline), self, "done",
                                 dt=(lambda: self.dt) if self.kind == "vanilla" else None)

    def advance(self, p):
        who, flood = self.who, self.flood
        steps = int(p["steps"])
        if steps < 1:
            raise ValueError(f"{who}: steps must be >= 1")
        scalars = {"friction_exponent": float(p["friction_exponent"]),
                   "carve_slope_min": float(p["carve_slope_min"])}
        if "precipitation" not in self.fields:
            scalars["precipitation"] = float(p["rainfall"]) * _MM_PER_H
        if "friction_coefficient" not in self.fields:
            scalars["friction_coefficient"] = float(p["manning"])
        if self.kind == "relax":
            scalars["relaxation"] = float(p["analytical_relaxation"])
        else:
            self.dt = float(p["dt"])
            scalars["dt"] = self.dt
        _flood_set(flood, **scalars)
        self.checks.configure(flood, p)
        check = int(p["check_every"])
        warm = self.warm_pending
        total = warm + steps
        if warm > 0:
            progress.steps(_make_counter(flood.warmup, self, "warmed"), warm, "warm-up",
                           total=total, sync=_gpu_sync)
            self.warm_pending = 0
        start, target = self.done, self.done + steps
        while self.done < target:
            n = target - self.done if check <= 0 else min(check, target - self.done)
            progress.steps(self.run, n, self.checks.phase(f"{self.solver} steps"),
                           parts=max(1, round(40 * n / steps)),
                           offset=warm + self.done - start, total=total, sync=_gpu_sync)
            extra = {"h_max": float(flood.h.array.max())}
            if self.kind == "vanilla":
                extra["time"] = self.time
            if self.checks.check(flood, self.done, **extra):
                progress.report(total, total, self.checks.phase("converged"))
                break
        returns = _flood_returns(**{k: p[k] for k in (
            "return_discharge", "return_outflow", "return_imbalance", "return_report")})
        report = self.checks.table(f"GraphFlood ({self.solver})", _CONV_UNITS)
        return _flood_outputs(self.dem, self.valid, self.lake, flood.h.to_numpy(),
                              flood.Qi.to_numpy(), flood.Qo.to_numpy(), returns, report)

    def coordinate(self):
        return {"step": self.done}

    def close(self):
        self.flood.close()


@process(
    id="pyfastflow.graphflood_particle",
    label="GraphFlood by particles (steady water depth and discharge)",
    inputs=[
        Port("dem", "georaster",
             doc="Topography (projected, metres); nodata cells stay out."),
        Port("h_init", "georaster", optional=True,
             doc="Water depth (m) above the DEM to start from, before the warm-up (e.g. a "
                 "GraphFlood run's water_depth). Same grid as the DEM."),
        Port("precipitation_field", "georaster", optional=True,
             doc="Rainfall per cell (mm/h), replaces Rainfall; same grid as the DEM."),
        Port("manning_field", "georaster", optional=True,
             doc="Manning n per cell (e.g. from land cover), replaces Manning; same grid "
                 "as the DEM."),
    ],
    params=[
        Param("launches", "int", default=1, min=1,
              doc="Particle launches; a report row after each."),
        Param("particles", "int", default=1_000_000, min=1,
              doc="Particles released per launch."),
        *_flood_common_params(),
        Param("source_percentile", "float", default=99.0, min=0.0, max=100.0,
              doc="Sources: valid cells whose warm-up discharge is at or above this "
                  "percentile of the non-zero discharge; particles spawn on them and "
                  "their steepest paths downstream (the active area)."),
        Param("h_relaxation", "float", default=0.01, min=0.0, max=1.0,
              doc="Fraction of the way to the target depth taken per visit."),
        Param("walk_steps", "int", default=100, min=1,
              doc="Most cells walked by one particle."),
        Param("spawn_pad", "int", default=2, min=0,
              doc="Particles start up to this many rows/columns from the active area."),
        Param("propagate", "float", default=0.5, min=0.0,
              doc="Cells whose inflow a visit changed by more than this fraction are "
                  "processed by the same thread before its next particle (0: off)."),
        Param("seed", "int", default=1, min=0, doc="Random seed of the particle starts."),
        Param("threads", "int", default=32768, min=32,
              doc="GPU threads walking particles."),
        Param("return_discharge", "bool", default=False, doc="Also return Qin."),
        Param("return_outflow", "bool", default=False, doc="Also return Qout."),
        Param("return_imbalance", "bool", default=False, doc="Also return Qout / Qin - 1."),
        *_convergence_params("launches", 1),
        Param("return_active_area", "bool", default=False,
              doc="Also return the active area of the warm-up (1, else 0)."),
        Param("return_report", "bool", default=False,
              doc="Also return the report table (residual and particle statistics per "
                  "launch)."),
    ],
    outputs=[
        *_flood_common_outputs(),
        Output("active_area", "georaster", optional=True,
               doc="Warm-up sources and their steepest paths downstream: 1, else 0."),
        Output("report", "datatable", optional=True,
               doc="Per convergence check: launch, the checker's metrics (as GraphFlood, "
                   "plus staleness, outlet_balance and coverage), the run statistics of "
                   "the launches since the last check (fractions of particles that "
                   "exited / hit walk_steps / got stuck, visits skipped on a locked cell, "
                   "propagated processings per visit, rejected starts per particle) and "
                   "stop (1: tolerance reached, 2: plateau, else 0)."),
    ],
    impl="library",
)
def graphflood_particle(dem, h_init=None, precipitation_field=None, manning_field=None,
                        launches=1, particles=1_000_000, rainfall=50.0, manning=0.033,
                        friction_exponent=2.0 / 3.0, initial_water="dry", warmup_steps=5,
                        warmup_relaxation=0.6, depth_cap=0.5, carve_slope_min=1e-4,
                        topology="D8", boundary="normal", local_minima="carve_cordonnier",
                        outlet_edges="all", nodata_outlets=True, source_percentile=99.0,
                        h_relaxation=0.01, walk_steps=100, spawn_pad=2, propagate=0.5,
                        seed=1, threads=32768, stop="count", check_every=1,
                        convergence_metric="dh_p99", convergence_tol=1e-3,
                        convergence_window=10, convergence_eps=0.01,
                        convergence_percentile=90.0, convergence_memory=0.8,
                        return_discharge=False,
                        return_outflow=False, return_imbalance=False,
                        return_active_area=False, return_report=False):
    """Steady water depth and discharge by pyfastflow's particle GraphFlood
    (GPU, CuPy) on ``dem``: warm-up (capped analytical passes, sources,
    active area), ``launches`` launches of ``particles`` particles, finish.
    Nodata cells are left out and stay NaN.
    """
    args = dict(locals())
    inputs = {k: args.pop(k) for k in ("dem", "h_init", "precipitation_field", "manning_field")}
    return run_once(GraphFloodParticleRun, inputs, args)


@runner("pyfastflow.graphflood_particle", config=(
    "topology", "boundary", "local_minima", "outlet_edges", "nodata_outlets",
    "initial_water", "warmup_steps", "warmup_relaxation", "depth_cap", "source_percentile",
    "threads"))
class GraphFloodParticleRun(Runner):
    """Live particle GraphFlood: one program across advances. The first
    advance runs the warm-up (sources, active area); each advance launches
    ``launches`` more times; counts, report rows and the stop rule continue."""

    who = "GraphFlood particles"
    stats = ("exit", "limited", "stuck", "skipped", "propagated", "rejected")

    def __init__(self, inputs, p):
        who = self.who
        self.dem = inputs["dem"]
        self.dx, self.valid, z = _flood_grid(self.dem, who)
        shape = self.shape = z.shape
        options = _flood_options(shape, self.dx, self.valid, p["topology"], p["boundary"],
                                 p["local_minima"])
        fields, _ = _split_fields(
            precipitation=_rain(inputs.get("precipitation_field"), p["rainfall"], shape, who),
            friction_coefficient=_field(inputs.get("manning_field"), p["manning"], shape,
                                        "manning_field", who))
        self.fields = set(fields)
        gfp = self.gfp = GraphFloodParticles(_backend(), threads=int(p["threads"]), **options,
                                             **fields)
        try:
            _flood_set(gfp, warmup_relaxation=float(p["warmup_relaxation"]),
                       depth_cap=float(p["depth_cap"]),
                       source_percentile=float(p["source_percentile"]))
            _flood_masks(gfp, self.valid, p["outlet_edges"], p["boundary"],
                         p["nodata_outlets"], who)
            self.lake = _flood_start(gfp, z, self.valid, p["initial_water"],
                                     inputs.get("h_init"), who)
        except Exception:
            gfp.close()
            raise
        self.checks = _Checks("launch", who)
        self.warmup_steps = int(p["warmup_steps"])
        self.warmed = False
        self.launched = 0

    def advance(self, p):
        who, gfp = self.who, self.gfp
        scalars = {"friction_exponent": float(p["friction_exponent"]),
                   "carve_slope_min": float(p["carve_slope_min"]),
                   "relaxation": float(p["h_relaxation"]), "propagate": float(p["propagate"]),
                   "n_particles": int(p["particles"]), "walk_steps": int(p["walk_steps"]),
                   "spawn_pad": int(p["spawn_pad"]), "seed": int(p["seed"])}
        if "precipitation" not in self.fields:
            scalars["precipitation"] = float(p["rainfall"]) * _MM_PER_H
        if "friction_coefficient" not in self.fields:
            scalars["friction_coefficient"] = float(p["manning"])
        _flood_set(gfp, **scalars)
        self.checks.configure(gfp, p)
        check = int(p["check_every"])
        total = int(p["launches"])
        units = total + (0 if self.warmed else 1)
        done = 0
        if not self.warmed:
            progress.report(0, units, "warm-up and active area")
            if gfp.warmup(self.warmup_steps) == 0:
                raise ValueError(f"{who}: empty active area (lower the source percentile)")
            self.warmed = True
            done = 1
        ran = 0
        while ran < total:
            n = total - ran if check <= 0 else min(check, total - ran)
            progress.report(done + ran, units, self.checks.phase(
                f"launch {self.launched + 1} ({ran + 1} of {total})"))
            r = gfp.run(n)
            ran += n
            self.launched += n
            if self.checks.check(gfp, self.launched, h_max=float(gfp.h.array.max()),
                                 **{k: r[k] for k in self.stats}):
                break
        gfp.finish()
        progress.report(units, units, self.checks.phase("done"))
        returns = _flood_returns(**{k: p[k] for k in (
            "return_discharge", "return_outflow", "return_imbalance", "return_report")})
        report = self.checks.table("GraphFlood particles", _CONV_UNITS)
        out = _flood_outputs(self.dem, self.valid, self.lake, gfp.h.to_numpy(),
                             gfp.Qi.to_numpy(), gfp.Qo.to_numpy(), returns, report)
        if p["return_active_area"]:
            active = gfp._handle("reach").to_numpy().reshape(self.shape)
            out["active_area"] = _like(self.dem, np.where(
                self.valid, (active != 0).astype(np.float32), np.nan))
        return out

    def coordinate(self):
        return {"launch": self.launched}

    def close(self):
        self.gfp.close()


# ---------------------------------------------------------------------------
# Inertial flood
# ---------------------------------------------------------------------------

@dataclass
class InertialFloodState:
    """What an inertial flood run needs to continue exactly: depth (m), the
    staggered link unit discharges (m²/s; ``qxy``/``qyx`` only in D8), model
    time (s), the running maximum depth (m), the topology and cell size."""

    h: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    qxy: Optional[np.ndarray]
    qyx: Optional[np.ndarray]
    time: float
    max_depth: np.ndarray
    topology: str
    dx: float


register_type("pyfastflow.InertialFloodState", "state",
              lambda v: isinstance(v, InertialFloodState),
              doc="Inertial flood state (depth, link discharges, time) to continue a run.")

_INERTIAL_REGULATORS = ["bates", "q_upwind", "q_centered", "s_upwind", "s_centered"]
_INERTIAL_REGULATOR_LABELS = ["None (Bates)", "Upwind, fixed theta", "Centred, fixed theta",
                              "Upwind, adaptive theta", "Centred, adaptive theta"]


def _shifted(a, dy, dx, periodic_y, periodic_x):
    """``b[y, x] = a[y - dy, x - dx]``; zero where that falls off a
    non-periodic edge."""
    b = np.roll(a, (dy, dx), axis=(0, 1))
    if not periodic_y and dy:
        b[(slice(None, dy) if dy > 0 else slice(dy, None)), :] = 0.0
    if not periodic_x and dx:
        b[:, (slice(None, dx) if dx > 0 else slice(dx, None))] = 0.0
    return b


def _cell_discharge(qx, qy, qxy, qyx, dx, periodic_x, periodic_y):
    """Magnitude (m³/s) of the cell-centred discharge: each link's discharge
    (unit discharge x link width: dx in D4, dx / 2 in D8) along its
    direction, averaged over the link's two cells. Positive qx flows east,
    qy south, qxy south-east and qyx south-west (rows grow southwards)."""
    d8 = qxy is not None
    width = 0.5 * dx if d8 else dx
    east = qx[:, 1:].copy()
    if periodic_x:
        east[:, -1] = qx[:, 0]
    south = qy[1:, :].copy()
    if periodic_y:
        south[-1, :] = qy[0, :]
    fx = 0.5 * width * (qx[:, :-1] + east)
    fy = 0.5 * width * (qy[:-1, :] + south)
    if d8:
        # The cell's own SE / SW links and the ones ending in it (from its
        # north-west / north-east neighbour).
        k = 0.5 * width / np.sqrt(2.0)
        se = k * (qxy + _shifted(qxy, 1, 1, periodic_y, periodic_x))
        sw = k * (qyx + _shifted(qyx, 1, -1, periodic_y, periodic_x))
        fx = fx + se - sw
        fy = fy + se + sw
    return np.hypot(fx, fy)


def _inertial_start(flood, valid, h_init, state, topology, dx, who):
    """Writes the start depth (and fluxes from ``state``) into ``flood``;
    returns (start time, running max depth as a host array)."""
    shape = valid.shape
    if state is not None:
        if state.h.shape != shape:
            raise ValueError(f"{who}: the state is {state.h.shape[0]} x {state.h.shape[1]} "
                             f"cells, the DEM {shape[0]} x {shape[1]}")
        if state.topology != topology:
            raise ValueError(f"{who}: the state was run in {state.topology}, not {topology}")
        if abs(state.dx - dx) > 1e-6 * dx:
            raise ValueError(f"{who}: the state's cell size is {state.dx} m, the DEM's {dx} m")
        flood.h.from_numpy(np.where(valid, state.h, 0.0).astype(np.float32))
        flood.qx.from_numpy(state.qx.astype(np.float32))
        flood.qy.from_numpy(state.qy.astype(np.float32))
        if topology == "D8":
            flood.qxy.from_numpy(state.qxy.astype(np.float32))
            flood.qyx.from_numpy(state.qyx.astype(np.float32))
        return float(state.time), state.max_depth.astype(np.float32)
    if h_init is not None:
        if h_init.z.shape != shape:
            raise ValueError(f"{who}: h_init is {h_init.z.shape[0]} x {h_init.z.shape[1]} "
                             f"cells, the DEM {shape[0]} x {shape[1]}")
        h = np.where(valid, np.maximum(np.nan_to_num(h_init.z, nan=0.0), 0.0), 0.0)
        flood.h.from_numpy(h.astype(np.float32))
    return 0.0, flood.h.to_numpy()


@process(
    id="pyfastflow.inertial_flood",
    label="Inertial flood (transient water depth and discharge)",
    inputs=[
        Port("dem", "georaster",
             doc="Topography (projected, metres); nodata cells stay out."),
        Port("h_init", "georaster", optional=True,
             doc="Water depth (m) to start from, flow at rest (e.g. a GraphFlood steady "
                 "state). Same grid as the DEM. Ignored when previous_state is given."),
        Port("previous_state", "pyfastflow.InertialFloodState", optional=True,
             doc="A previous run's state: continues it exactly (depth, discharges, model "
                 "time, max depth)."),
        Port("precipitation_field", "georaster", optional=True,
             doc="Rainfall per cell (mm/h), replaces Rainfall; same grid as the DEM."),
    ],
    params=[
        Param("duration", "float", default=3600.0, min=0.0,
              doc="Model time to simulate (s), from the start (or previous_state's time)."),
        Param("rainfall", "float", default=50.0, min=0.0,
              doc="Rainfall rate (mm/h), when no precipitation field is given."),
        Param("storm_duration", "float", default=0.0, min=0.0,
              doc="Rain stops at this model time (s; counted from the first run of a "
                  "chain); 0: rain all along."),
        Param("manning", "float", default=0.033, min=0.0,
              doc="Manning roughness n (s/m^(1/3)); one value for the grid."),
        Param("time_step", "string", default="auto", choices=["auto", "fixed"],
              choice_labels=["Automatic (Bates stability limit)", "Fixed"],
              doc="auto: dt = min(dt_max, cfl dx / sqrt(g h_max)), recomputed every "
                  "dt_update_steps steps; fixed: dt."),
        Param("dt", "float", default=1.0, min=0.0, doc="Fixed time step (s)."),
        Param("dt_max", "float", default=10.0, min=0.0,
              doc="Automatic time step: upper bound (s), also used while the grid is dry."),
        Param("cfl", "float", default=0.7, min=0.0, max=1.0,
              doc="Automatic time step: Courant number of the Bates limit."),
        Param("dt_update_steps", "int", default=25, min=1,
              doc="Steps between two time-step updates (also between max-depth samples)."),
        Param("topology", "string", default="D4", choices=["D4", "D8"],
              doc="Links: D4, or D8 (each of the 8 links dx / 2 wide; about 1.5x the "
                  "memory)."),
        Param("boundary", "string", default="normal",
              choices=["normal", "periodic_EW", "periodic_NS"],
              doc="Grid edges: normal, or periodic east-west / north-south (a periodic "
                  "edge never lets water out)."),
        Param("outlet_edges", "string", default="all", choices=list(_OUTLET_SIDES),
              doc="Grid edges whose cells are outlets."),
        Param("nodata_outlets", "bool", default=True,
              doc="Valid cells touching nodata are outlets too."),
        Param("fix_outlet_depth", "bool", default=True,
              doc="Outlet cells are held at Outlet depth (water reaching them leaves)."),
        Param("outlet_depth", "float", default=0.0, min=0.0,
              doc="Depth (m) held on outlet cells."),
        Param("boundary_flux", "float", default=0.0,
              doc="Outward unit discharge (m²/s) on the grid-edge links of outlet cells."),
        Param("outlet_discharge", "float", default=0.0,
              doc="Discharge (m³/s) taken out of every outlet cell (when the outlet depth "
                  "is not fixed)."),
        Param("top_inflow", "float", default=0.0, min=0.0,
              doc="Discharge (m³/s) added to every valid cell of the top (first) row."),
        Param("regulator", "string", default="bates", choices=_INERTIAL_REGULATORS,
              choice_labels=_INERTIAL_REGULATOR_LABELS,
              doc="Smoothing of the provisional link discharge before the depth update: "
                  "none (Bates), with the upwind or both parallel links, at a fixed theta "
                  "(de Almeida et al., 2012) or an adaptive one (Sridharan et al., 2020)."),
        Param("regulator_theta", "float", default=0.9, min=0.0, max=1.0,
              doc="Fixed-theta regulators: weight of the link's own discharge (1: Bates)."),
        Param("froude_limit", "float", default=1.0, min=0.0,
              doc="Cap of the link discharge, as a Froude number."),
        Param("transfer_fraction", "float", default=0.2, min=0.0, max=1.0,
              doc="Most of the donor cell's water a link moves in one step."),
        Param("min_flow_depth", "float", default=1e-4, min=0.0,
              doc="Links with a mean depth at or below this (m) carry nothing."),
        Param("min_depth", "float", default=0.0, min=0.0,
              doc="Floor of the depth (m)."),
        Param("gravity", "float", default=9.81, min=0.0, doc="Gravity (m/s²)."),
        Param("report_every", "float", default=0.0, min=0.0,
              doc="Report row every this much model time (s); 0: at the end only."),
        Param("return_discharge", "bool", default=False,
              doc="Also return the cell discharge (m³/s)."),
        Param("return_velocity", "bool", default=False,
              doc="Also return the flow speed (m/s)."),
        Param("return_max_depth", "bool", default=False,
              doc="Also return the maximum depth reached (m)."),
        Param("return_state", "bool", default=False,
              doc="Also return the state, to continue the run."),
        Param("return_report", "bool", default=False,
              doc="Also return the report table (time, dt, max depth, water balance)."),
    ],
    outputs=[
        Output("water_depth", "georaster",
               doc="Water depth (m) at the end; NaN on nodata."),
        Output("discharge", "georaster", optional=True,
               doc="Cell discharge (m³/s): link discharges along their directions, "
                   "averaged on the cell."),
        Output("velocity", "georaster", optional=True,
               doc="Flow speed (m/s): cell discharge / (dx h); 0 where h <= min flow depth."),
        Output("max_depth", "georaster", optional=True,
               doc="Maximum depth (m) over the run (and the chain it continues), sampled "
                   "every dt_update_steps steps."),
        Output("state", "pyfastflow.InertialFloodState", optional=True,
               doc="State to continue the run (input `previous_state`)."),
        Output("report", "datatable", optional=True,
               doc="Per row: model time, last dt, max depth, stored volume, rain and top "
                   "inflow volumes since the start of this call, and the volume lost "
                   "(start + inputs - stored: outlets, edge fluxes, sinks)."),
    ],
    impl="library",
)
def inertial_flood(dem, h_init=None, previous_state=None, precipitation_field=None,
                   duration=3600.0,
                   rainfall=50.0, storm_duration=0.0, manning=0.033, time_step="auto",
                   dt=1.0, dt_max=10.0, cfl=0.7, dt_update_steps=25, topology="D4",
                   boundary="normal", outlet_edges="all", nodata_outlets=True,
                   fix_outlet_depth=True, outlet_depth=0.0, boundary_flux=0.0,
                   outlet_discharge=0.0, top_inflow=0.0, regulator="bates",
                   regulator_theta=0.9, froude_limit=1.0, transfer_fraction=0.2,
                   min_flow_depth=1e-4, min_depth=0.0, gravity=9.81, report_every=0.0,
                   return_discharge=False, return_velocity=False, return_max_depth=False,
                   return_state=False, return_report=False):
    """Transient local-inertia flood (pyfastflow ``InertialFloodProgram``, GPU,
    CuPy) on ``dem`` for ``duration`` seconds of model time.

    Each step updates the link discharges from the water-surface slope with
    semi-implicit Manning friction, then the depths by continuity with the
    rain. The time step is fixed or follows the Bates stability limit.
    Nodata cells are left out and stay NaN.
    """
    args = dict(locals())
    inputs = {k: args.pop(k) for k in ("dem", "h_init", "previous_state",
                                       "precipitation_field")}
    return run_once(InertialFloodRun, inputs, args)


@runner("pyfastflow.inertial_flood", config=(
    "topology", "boundary", "outlet_edges", "nodata_outlets", "fix_outlet_depth",
    "outlet_depth", "regulator"))
class InertialFloodRun(Runner):
    """Live inertial flood: one program across advances. Each advance
    simulates ``duration`` more seconds; model time, max depth, the water
    balance and the report rows continue. Rain (rate, storm end) and every
    non-config param can change between advances."""

    who = "Inertial flood"
    columns = ("time", "dt", "h_max", "volume", "rain_in", "inflow", "lost")

    def __init__(self, inputs, p):
        import cupy as cp

        who = self.who
        if p["regulator"] not in _INERTIAL_REGULATORS:
            raise ValueError(f"{who}: unknown regulator {p['regulator']!r}")
        self.dem = inputs["dem"]
        self.dx, self.valid, z = _flood_grid(self.dem, who)
        shape = self.shape = z.shape
        self.topology, self.boundary = p["topology"], p["boundary"]
        self.rain_field = inputs.get("precipitation_field")
        rain = self._rain(p["rainfall"])
        flood = self.flood = InertialFloodProgram(
            _backend(), nx=shape[1], ny=shape[0], dx=self.dx, topology=self.topology,
            boundary=self.boundary, outlet="mask", nodata=not self.valid.all(),
            outlet_depth=float(p["outlet_depth"]) if p["fix_outlet_depth"] else None,
            regulator=p["regulator"], rainfall=rain)
        try:
            _flood_masks(flood, self.valid, p["outlet_edges"], self.boundary,
                         p["nodata_outlets"], who)
            flood.z.from_numpy(z)
            flood.reset()
            self.time, max_depth = _inertial_start(flood, self.valid, inputs.get("h_init"),
                                                   inputs.get("previous_state"),
                                                   self.topology, self.dx, who)
        except Exception:
            flood.close()
            raise
        self.max_depth = cp.asarray(max_depth)
        self.h = flood.h.array
        self.cell = self.dx * self.dx
        self.volume0 = float(self.h.sum(dtype=cp.float64)) * self.cell
        self.rain_in = self.inflow = 0.0
        self.rows = {k: [] for k in self.columns}
        self.rain_now = rain  # the field the program holds
        self.next_row = None
        self.step_dt = 0.0

    def _rain(self, rate):
        """The rain field (m/s, 0 on nodata) for ``rate`` (mm/h) or the field input."""
        shape = self.shape
        return np.where(self.valid, np.broadcast_to(
            _rain(self.rain_field, rate, shape, self.who), shape), 0.0).astype(np.float32)

    def _set_rain(self, field):
        if field is not self.rain_now:
            self.flood.rainfall.from_numpy(field)
            self.rain_now = field

    def advance(self, p):
        import cupy as cp

        who, flood, h = self.who, self.flood, self.h
        time_step = p["time_step"]
        if time_step not in ("auto", "fixed"):
            raise ValueError(f"{who}: unknown time_step {time_step!r}")
        duration = float(p["duration"])
        if duration <= 0.0:
            raise ValueError(f"{who}: duration must be > 0")
        step_dt = float(p["dt"]) if time_step == "fixed" else float(p["dt_max"])
        if step_dt <= 0.0:
            raise ValueError(f"{who}: the time step must be > 0")
        _flood_set(flood, manning=float(p["manning"]), gravity=float(p["gravity"]),
                   froude_limit=float(p["froude_limit"]),
                   transfer_fraction=float(p["transfer_fraction"]),
                   regulator_theta=float(p["regulator_theta"]),
                   min_flow_depth=float(p["min_flow_depth"]), min_depth=float(p["min_depth"]),
                   boundary_flux=float(p["boundary_flux"]),
                   outlet_discharge=float(p["outlet_discharge"]),
                   top_inflow=float(p["top_inflow"]))
        dx, g, cell = self.dx, float(p["gravity"]), self.cell
        min_flow = float(p["min_flow_depth"])
        rain_on = self._rain(p["rainfall"])
        if self.rain_field is None and np.array_equal(rain_on, self.rain_now):
            rain_on = self.rain_now
        rain_off = np.zeros(self.shape, np.float32)
        rain_rate = float(rain_on.sum(dtype=np.float64)) * cell
        top_rate = float(p["top_inflow"]) * float(self.valid[0].sum())
        storm = float(p["storm_duration"])
        every = int(p["dt_update_steps"])
        t0 = self.time
        end = t0 + duration
        report_dt = float(p["report_every"])
        if self.next_row is None or self.next_row <= t0:
            self.next_row = t0 + report_dt if report_dt > 0.0 else end
        self.next_row = min(self.next_row, end)
        eps = 1e-9 * max(duration, 1.0)
        progress.report(0.0, duration, "inertial flow")
        while self.time < end - eps:
            t = self.time
            raining = storm <= 0.0 or t < storm - eps
            self._set_rain(rain_on if raining else rain_off)
            if time_step == "auto":
                h_top = float(h.max())
                step_dt = float(p["dt_max"]) if h_top <= min_flow else min(
                    float(p["dt_max"]), float(p["cfl"]) * dx / np.sqrt(g * h_top))
            stop = min(end, self.next_row, storm if raining and storm > 0.0 else end)
            n = min(max(1, int(np.ceil((stop - t) / step_dt - 1e-9))), every)
            landing = n * step_dt >= stop - t - eps
            if landing:
                step_dt = (stop - t) / n  # land exactly on the stop (smaller: stable)
            flood.dt.set(step_dt)
            flood.step(n)
            elapsed = n * step_dt
            self.time = stop if landing else t + elapsed
            self.step_dt = step_dt
            if raining:
                self.rain_in += rain_rate * elapsed
            self.inflow += top_rate * elapsed
            cp.maximum(self.max_depth, h.reshape(self.shape), out=self.max_depth)
            if self.time >= self.next_row - eps:
                self._row()
                self.next_row = (min(self.next_row + report_dt, end) if report_dt > 0.0
                                 else end)
            progress.report(self.time - t0, duration, "inertial flow")
        return self._outputs(p)

    def _row(self):
        import cupy as cp

        volume = float(self.h.sum(dtype=cp.float64)) * self.cell
        for k, v in zip(self.columns, (
                self.time, self.step_dt, float(self.h.max()), volume, self.rain_in,
                self.inflow, self.volume0 + self.rain_in + self.inflow - volume)):
            self.rows[k].append(v)

    def _outputs(self, p):
        import cupy as cp

        flood, valid, dem, dx = self.flood, self.valid, self.dem, self.dx
        h_out = flood.h.to_numpy()
        out = {"water_depth": _like(dem, np.where(valid, h_out, np.nan))}
        d8 = self.topology == "D8"
        qx, qy = flood.qx.to_numpy(), flood.qy.to_numpy()
        qxy = flood.qxy.to_numpy() if d8 else None
        qyx = flood.qyx.to_numpy() if d8 else None
        if p["return_discharge"] or p["return_velocity"]:
            q = _cell_discharge(qx, qy, qxy, qyx, dx, self.boundary == "periodic_EW",
                                self.boundary == "periodic_NS")
            if p["return_discharge"]:
                out["discharge"] = _like(dem, np.where(valid, q, np.nan))
            if p["return_velocity"]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = np.where(h_out > float(p["min_flow_depth"]), q / (dx * h_out), 0.0)
                out["velocity"] = _like(dem, np.where(valid, v, np.nan))
        max_out = cp.asnumpy(self.max_depth)
        if p["return_max_depth"]:
            out["max_depth"] = _like(dem, np.where(valid, max_out, np.nan))
        if p["return_state"]:
            out["state"] = InertialFloodState(h=h_out, qx=qx, qy=qy, qxy=qxy, qyx=qyx,
                                              time=self.time, max_depth=max_out,
                                              topology=self.topology, dx=dx)
        if p["return_report"]:
            out["report"] = DataTable(
                {k: np.asarray(v, dtype=np.float64) for k, v in self.rows.items()},
                units={"time": "s", "dt": "s", "h_max": "m", "volume": "m³", "rain_in": "m³",
                       "inflow": "m³", "lost": "m³"},
                roles={"time": "x", "dt": "aux"}, title="Inertial flood")
        return out

    def coordinate(self):
        return {"time": float(self.time)}

    def close(self):
        self.flood.close()


if GraphFloodRelax is None:
    from ..core.process import DEFAULT_PROCESSES
    for _pid in ("pyfastflow.graphflood", "pyfastflow.graphflood_particle"):
        DEFAULT_PROCESSES._procs.pop(_pid, None)
if InertialFloodProgram is None:
    from ..core.process import DEFAULT_PROCESSES
    DEFAULT_PROCESSES._procs.pop("pyfastflow.inertial_flood", None)
