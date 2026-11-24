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
- ``pyfastflow.inertial_flood``: transient local-inertia flood
  (``InertialFloodProgram``), live runner, continuation state.
- ``pyfastflow.fusw_flood``: transient finite-volume shallow-water flood
  after FullSWOF 2D (``FuswProgram``), live runner, continuation state.

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
try:
    from pyfastflow.flood.fusw import BOUNDARY_CODES as FUSW_CODES
    from pyfastflow.flood.fusw import FuswProgram
    from pyfastflow.flood.fusw._common import B_FACE as _FUSW_B_FACE
    from pyfastflow.flood.fusw._common import C_T_END as _FUSW_T_END
    from pyfastflow.flood.fusw._common import C_TIME as _FUSW_TIME
except ImportError:
    FuswProgram = FUSW_CODES = None
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
    category="Input",
    label="Salève landscape",
    params=[
        # Grid
        Param("n", "int",
              label="Grid size", unit="cells per side", default=1024,
              choices=[256, 512, 1024, 2048],
              doc="Cells per side (square grid)."),
        Param("dx", "float", label="Cell size dx", unit="m", default=50.0, min=0.0,
              doc="Cell size in metres."),
        Param("seed", "int", default=1, min=0,
              doc="Seed of the initial white noise and of the level jitter; "
                  "the uplift / erodibility fields use seed + 1000 / + 2000."),
        # Stream power
        Param("m", "float", label="Area exponent m", default=0.45, min=0.0,
              doc="Drainage-area exponent of the stream-power law."),
        Param("uplift", "float", label="Uplift U", unit="m/yr", default=1e-3, min=0.0,
              doc="Uplift rate U (m/yr), mean of the uplift field."),
        Param("erodibility", "float", label="Erodibility K", default=1e-5, min=0.0,
              doc="Erodibility K, mean of the erodibility field."),
        # Fields
        Param("uplift_noise", "string", label="Uplift field noise", default=_UPLIFT_NOISE,
              doc="Uplift field. " + _NOISE_DOC),
        Param("erodibility_noise", "string",
              label="Erodibility field noise", default=_ERODIBILITY_NOISE,
              doc="Erodibility field. " + _NOISE_DOC),
        Param("return_fields", "bool", default=False,
              doc="Also return the uplift and erodibility fields."),
        # Hillslope
        Param("hillslope_erosion", "float", label="Diffusivity D (0 = off)", default=0.4, min=0.0,
              doc="Hillslope diffusivity D (m²/yr): adds D / (c A^h) to the "
                  "erosion term (Tzathas et al. 2024, Eqn. 26); 0 = off."),
        Param("hack_constant", "float", label="Hack constant c", default=1.5, min=0.0,
              doc="Hack's law constant c (hillslope length c A^h)."),
        Param("hack_exponent", "float", label="Hack exponent h", default=0.6, min=0.0,
              doc="Hack's law exponent h."),
        # Thermal (talus)
        Param("thermal_erosion", "float", label="Thermal kt (0 = off)", default=0.0, min=0.0,
              doc="Thermal (talus) coefficient kt (m/yr): where the slope exceeds Sc, "
                  "adds kt to the erosion term and kt Sc to the uplift (Eqns. 28-29); 0 = off."),
        Param("critical_slope", "float", label="Critical slope Sc", default=0.5, min=0.0,
              doc="Critical slope Sc of thermal erosion (tan of the talus angle)."),
        # Network
        Param("topology", "string", label="Flow topology", default="D8", choices=TOPOLOGIES,
              doc="Flow neighbourhood."),
        Param("boundary", "string", label="Boundaries", default="periodic_EW", choices=BOUNDARIES,
              doc="Grid edges: normal (all edges are outlets) or periodic "
                  "east-west / north-south."),
        Param("local_minima", "string", default="cordonnier_carve", choices=LOCAL_MINIMA,
              doc="How local minima are resolved when routing flow."),
        Param("epsilon", "float", label="Local minima epsilon", default=1e-3, min=0.0,
              doc="Epsilon of Salève's finite-time solver. SaleveProgram hands it "
                  "only to that solver, so it has no effect on this steady solve; "
                  "kept to match saleve_app."),
        Param("jitter", "float", label="Level jitter", default=10.0, min=0.0,
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
        Param("cliff_optimization", "bool", label="Cliff optimisation", default=False,
              doc="Fixed-network cliff post-correction after each high-level solve."),
        Param("cliff_iterations", "int", default=50, min=0,
              doc="Cliff optimisation iterations."),
        Param("cliff_learning_rate", "float", default=0.01, min=0.0,
              doc="Cliff optimisation learning rate."),
        Param("cliff_river_weight", "float", default=1.0 / 3.0, min=0.0, max=1.0,
              doc="Weight of the river term in the cliff optimisation."),
        # Multigrid
        Param("levels", "int", label="Multigrid levels", default=5, min=1,
              doc="Multigrid levels."),
        Param("iterations", "int", label="Multigrid iterations", default=24, min=1,
              doc="Multigrid iterations."),
        Param("relaxation", "float", label="Multigrid relaxation", default=0.25, min=0.0, max=1.0,
              doc="Multigrid relaxation."),
        # Multi-scale erosion post-process (Schott et al. 2024)
        Param("mse_stages", "int",
              label="Stages (0 = off, each after the 1st doubles resolution)", default=0, min=0,
              max=len(MSE_PRESET),
              doc="Multi-scale erosion post-process (Schott et al. 2024, release "
                  "code port): number of stages of its preset, each erosion, "
                  "thermal then deposition steps, the resolution doubling before "
                  "every stage after the first; 0 = off."),
        Param("mse_k", "float", label="Stream power k", default=0.0005, min=0.0,
              doc="Multi-scale erosion: stream power coefficient k."),
        Param("mse_p_sa", "float", label="Drainage exponent", default=0.8, min=0.0,
              doc="Multi-scale erosion: drainage exponent of the stream power."),
        Param("mse_p_sl", "float", label="Slope exponent", default=2.0, min=0.0,
              doc="Multi-scale erosion: slope exponent of the stream power."),
        Param("mse_dt", "float", label="Erosion time step", default=1.0, min=0.0,
              doc="Multi-scale erosion: time step of the erosion steps."),
        Param("mse_max_spe", "float", label="Stream power cap", default=10000.0, min=0.0,
              doc="Multi-scale erosion: cap of the stream power before k."),
        Param("mse_flow_p", "float", label="Flow split exponent", default=1.3, min=0.0,
              doc="Multi-scale erosion: exponent of the slope weights of the "
                  "multiple-flow split (erosion and deposition)."),
        Param("mse_eps", "float", label="Thermal step eps", default=0.00005, min=0.0,
              doc="Multi-scale erosion: thermal step, height moved per unstable "
                  "neighbour is eps * dx²."),
        Param("mse_noisified", "bool", label="Noisy talus angle", default=True,
              doc="Multi-scale erosion: thermal talus angle drawn from simplex "
                  "noise between the noise min and max (else the fixed tangent)."),
        Param("mse_tan_angle", "float", label="Talus tangent (fixed)", default=0.57, min=0.0,
              doc="Multi-scale erosion: fixed tangent of the talus angle."),
        Param("mse_noise_min", "float", label="Talus tangent min (noisy)", default=0.9, min=0.0,
              doc="Multi-scale erosion: lowest noisy talus tangent."),
        Param("mse_noise_max", "float", label="Talus tangent max (noisy)", default=1.4, min=0.0,
              doc="Multi-scale erosion: highest noisy talus tangent."),
        Param("mse_noise_wavelength", "float",
              label="Talus noise frequency", unit="1/m", default=0.0023, min=0.0,
              doc="Multi-scale erosion: frequency (1/m) of the talus noise."),
        Param("mse_deposition_strength", "float",
              label="Deposition strength", default=1.0, min=0.0,
              doc="Multi-scale erosion: deposition strength."),
    ],
    outputs=[
        Output("grid", "georaster", doc="The steady-state surface."),
        Output("uplift_field", "georaster", optional=True, switch="return_fields",
               doc="Uplift rate (m/yr), with return_fields."),
        Output("erodibility_field", "georaster", optional=True, switch="return_fields",
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
    category="Input",
    label="Perlin surface",
    params=[
        Param("nx", "int", label="Columns", default=1024, min=8, doc="Columns."),
        Param("ny", "int", label="Rows", default=1024, min=8, doc="Rows."),
        Param("dx", "float",
              label="Cell size dx", unit="m", default=30.0, min=0.0, doc="Cell size in metres."),
        Param("amplitude", "float", unit="m", default=300.0, min=0.0,
              doc="Noise amplitude (m)."),
        Param("frequency", "float", unit="features per domain", default=5.0, min=0.5,
              doc="Features per domain (rounded to a half integer)."),
        Param("octaves", "int", default=6, min=1, doc="Octaves."),
        Param("persistence", "float", default=0.5, min=0.0, max=1.0,
              doc="Amplitude kept per octave."),
        Param("seed", "int", default=42, min=0, doc="Seed."),
        Param("slope", "float",
              label="Regional slope to the south", unit="m/m", default=2e-3, min=0.0,
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
    category="Landscape evolution",
    label="GOLEM landscape evolution (no sediment)",
    inputs=[
        Port("dem", "georaster", label="Surface",
             doc="Surface to evolve (projected, metres); nodata cells stay out."),
        Port("uplift_field", "georaster", unit="m/yr", optional=True,
             doc="Uplift rate per cell (m/yr), replaces Uplift; same grid as the DEM."),
        Port("erodibility_field", "georaster", optional=True,
             doc="Erodibility per cell, replaces Erodibility; same grid as the DEM."),
    ],
    params=[
        Param("steps", "int", default=100, min=0, doc="Time steps to run."),
        Param("dt", "float",
              label="Time step dt", unit="yr", default=5e4, min=0.0, doc="Time step (yr)."),
        Param("uplift", "float", label="Uplift U", unit="m/yr", default=1e-3, min=0.0,
              doc="Uplift rate U (m/yr), when no uplift field is given."),
        Param("erodibility", "float", label="Erodibility K", default=1e-5, min=0.0,
              doc="Erodibility K, when no erodibility field is given."),
        Param("m", "float", label="Area exponent m", default=0.4, min=0.0,
              doc="Drainage-area exponent of the stream-power law."),
        Param("n", "float", label="Slope exponent n", default=1.6, min=0.0,
              doc="Slope exponent of the stream-power law."),
        Param("hillslope_diffusivity", "float",
              label="Hillslope diffusivity D (0 = off)", unit="m²/yr", default=3e-2, min=0.0,
              doc="Linear hillslope diffusivity D (m²/yr); 0 = off."),
        Param("slope_floor", "float", default=1e-6, min=0.0,
              doc="Smallest slope used by the incision solve."),
        Param("newton_iterations", "int", label="Newton iterations (incision)", default=8, min=1,
              doc="Newton iterations of the implicit incision solve."),
        Param("diffusion_iterations", "int", default=20, min=1,
              doc="Iterations of the implicit diffusion solve."),
        Param("topology", "string", label="Flow topology", default="D8", choices=["D4", "D8"],
              doc="Flow neighbourhood."),
        Param("boundary", "string", label="Boundaries", default="normal",
              choices=["normal", "periodic_EW", "periodic_NS"],
              doc="Grid edges: normal, or periodic east-west / north-south (a periodic "
                  "edge never drains out)."),
        Param("outlet_edges", "string", default="all", choices=list(_OUTLET_SIDES),
              doc="Grid edges whose cells drain out of the domain."),
        Param("nodata_outlets", "bool", label="Cells next to nodata drain out", default=True,
              doc="Valid cells touching nodata also drain out."),
        Param("local_minima", "string", default="cordonnier_carve",
              choices=["none", "reconstruct_epsilon", "cordonnier_carve",
                       "cordonnier_jump"],
              doc="How local minima are resolved when routing flow."),
        Param("return_drainage_area", "bool", label="Also drainage area", default=False,
              doc="Also return the drainage area (m²) of the last step's routing."),
        Param("return_erosion_rate", "bool", label="Also erosion rate", default=False,
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


@runner("pyfastflow.golem_nosed", axis="step", config=(
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

    def update(self, inputs):
        """A new surface (``dem``: same grid and nodata cells) carries on from
        the current step; the uplift and erodibility fields are fixed at
        opening."""
        fixed = [k for k in ("uplift_field", "erodibility_field") if k in inputs]
        if fixed:
            raise ValueError(f"GOLEM: {', '.join(fixed)} cannot change once the run is open")
        dem = inputs.get("dem")
        if dem is None:
            return
        if dem.z.shape != self.shape:
            raise ValueError(f"GOLEM: the new surface is {dem.z.shape[0]} x {dem.z.shape[1]} "
                             f"cells, the run {self.shape[0]} x {self.shape[1]}")
        if not np.array_equal(dem.valid, self.valid):
            raise ValueError("GOLEM: the new surface has other nodata cells than the run")
        self.golem.z.from_numpy(np.where(self.valid, dem.z, 0.0).astype(np.float32))
        self.dem = dem

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

    def status(self):
        return {"text": f"Step {self.ran}.", "stopped": False}

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


#: The sources port, shared by every flood process.
_SOURCES_PORT_DOC = (
    "Discharge sources (points): per point q (m³/s), and/or area (m²) with linked = 1 "
    "for q = rain at the point x area (the rain of the run, as it changes); spread "
    "over radius (m; a smooth bump, the point's cell under half a cell) and added to the "
    "rain. Points sharing a source id (> 0) are one source, its q spread over all their "
    "stamps (circles, brush strokes). Points off the valid cells are skipped.")


def _sources_port():
    return Port("sources", "geopoints", label="Discharge sources", optional=True,
                doc=_SOURCES_PORT_DOC)


def _source_stamps(points, dem, valid, who):
    """Per discharge source on the valid cells: (flat cell indices, weights
    summing to 1, q m³/s, area m² taken when linked); [] for none.

    Each point is a stamp: a radius under half a cell is the cell itself; a
    larger one a smooth bump, (1 - d²/r²)², so the source fades out at its
    edge (as nevertheLAS's painted stamps). Points sharing a ``source`` id
    (> 0) are one source (nevertheLAS's circles and brush strokes): their
    stamps add up and its q and area (the first point's) are spread over all
    of them. Points without one (or 0) are each their own source."""
    if points is None:
        return []
    xy = points.in_crs(dem.crs_wkt or dem.epsg).xy
    n = len(xy)
    attrs = points.feature_attrs

    def column(name, default):
        if name not in attrs:
            return np.full(n, default, np.float64)
        return np.nan_to_num(np.asarray(attrs[name], np.float64), nan=default)

    q, radius, area, linked = column("q", 0.0), column("radius", 0.0), column("area", 0.0), column("linked", 0.0)
    ids = column("source", 0.0)
    rows, cols = valid.shape
    d = dem.cell_size

    def stamp(k):
        """(cells, bump values) of point k on the valid cells, or None."""
        x, y = xy[k]
        c, r = (x - dem.x_min) / d, (dem.y_max - y) / d
        if radius[k] <= 0.5 * d:
            i, j = int(np.floor(r)), int(np.floor(c))
            if not (0 <= i < rows and 0 <= j < cols and valid[i, j]):
                return None
            return np.array([i * cols + j], np.int64), np.ones(1)
        reach = radius[k] / d
        r0, r1 = max(int(np.floor(r - reach)), 0), min(int(np.ceil(r + reach)) + 1, rows)
        c0, c1 = max(int(np.floor(c - reach)), 0), min(int(np.ceil(c + reach)) + 1, cols)
        if r0 >= r1 or c0 >= c1:
            return None
        ii, jj = np.mgrid[r0:r1, c0:c1]
        share = ((ii + 0.5 - r) ** 2 + (jj + 0.5 - c) ** 2) / (reach * reach)
        bump = np.where((share < 1.0) & valid[r0:r1, c0:c1], (1.0 - share) ** 2, 0.0)
        keep = bump > 0.0
        if not keep.any():
            return None
        return (ii[keep] * cols + jj[keep]).astype(np.int64), bump[keep]

    groups = {}  # source id (or -1 - point index when none) -> point indices, in order
    for k in range(n):
        groups.setdefault(ids[k] if ids[k] > 0.0 else -1.0 - k, []).append(k)
    stamps = []
    for members in groups.values():
        first = members[0]
        taken_area = area[first] if linked[first] > 0.0 else 0.0
        if q[first] == 0.0 and taken_area <= 0.0:
            continue
        parts = [part for part in (stamp(k) for k in members) if part is not None]
        if not parts:
            continue
        cells, inverse = np.unique(np.concatenate([c for c, _b in parts]), return_inverse=True)
        weights = np.bincount(inverse, weights=np.concatenate([b for _c, b in parts]))
        stamps.append((cells, weights / weights.sum(), float(q[first]), float(taken_area)))
    if n and not stamps:
        raise ValueError(f"{who}: no discharge source falls on the DEM's valid cells")
    return stamps


def _source_discharge(stamps, rain, shape):
    """m³/s per cell of the sources; ``rain`` (m/s, scalar or field) gives the
    linked ones' q = rain at the point x area."""
    out = np.zeros(shape[0] * shape[1], np.float64)
    flat = None if np.isscalar(rain) else np.broadcast_to(rain, shape).reshape(-1)
    for cells, weights, q, area in stamps:
        total = q
        if area > 0.0:
            total += area * (float(rain) if flat is None else float(np.dot(flat[cells], weights)))
        out[cells] += total * weights
    return out.reshape(shape)


def _rain_with_sources(rain, stamps, shape, dx, field=None):
    """Rain (m/s) with the sources and the source field (m³/s per cell, or
    None) added as depth per second on their cells."""
    if not stamps and field is None:
        return rain
    q = _source_discharge(stamps, rain, shape)
    if field is not None:
        q = q + field
    return (np.broadcast_to(rain, shape) + q / (dx * dx)).astype(np.float32)


def _source_field_port():
    return Port("source_field", "georaster", label="Source field", unit="m³/s per cell",
                optional=True,
                doc="Discharge entering each cell (m³/s), added to the rain; same grid as "
                    "the DEM.")


def _source_cells(field, valid, who):
    """The source field input as m³/s per valid cell (NaN and nodata: 0), or None."""
    if field is None:
        return None
    value = _field(field, 0.0, valid.shape, "source_field", who)
    return np.where(valid, value, 0.0)


def _series_factor(series, interp, t):
    """The factor of ``series`` ((times, factors), see _fusw_series) at model
    time ``t``: 1 without one, the end factors outside its times."""
    times, factors = series
    if not len(times):
        return 1.0
    if interp == "linear":
        return float(np.interp(t, times, factors))
    return float(factors[max(int(np.searchsorted(times, t, side="right")) - 1, 0)])


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
        Param("rainfall", "float", unit="mm/h", default=50.0, min=0.0,
              doc="Rainfall rate (mm/h), when no precipitation field is given."),
        Param("manning", "float", label="Manning n", default=0.033, min=0.0,
              doc="Manning roughness n (s/m^(1/3)), when no Manning field is given."),
        Param("friction_exponent", "float", default=2.0 / 3.0, min=0.0,
              doc="Exponent of the hydraulic radius in the friction law (2/3: Manning)."),
        Param("initial_water", "string", default="dry", choices=_INITIAL_WATER,
              choice_labels=_INITIAL_WATER_LABELS,
              doc="Start without h_init: dry; depressions filled with water; or depressions "
                  "flattened into the topography (their water is static, still counted in "
                  "the output depth)."),
        Param("warmup_steps", "int", label="Warm-up steps", default=5, min=0,
              doc="Capped analytical warm-up passes run first, filling the hillslopes "
                  "(analytical solver and particles; 0: none)."),
        Param("warmup_relaxation", "float",
              label="Warm-up relaxation", default=0.6, min=0.0, max=1.0,
              doc="Relaxation of the warm-up passes."),
        Param("depth_cap", "float", label="Warm-up depth cap", unit="m", default=0.5, min=0.0,
              doc="Warm-up: cap (m) of the target and of the depth."),
        Param("carve_slope_min", "float", default=1e-4, min=0.0,
              doc="Smallest slope through carved depressions."),
        Param("topology", "string", label="Flow topology", default="D8", choices=["D4", "D8"],
              doc="Flow neighbourhood."),
        Param("boundary", "string", label="Boundaries", default="normal",
              choices=["normal", "periodic_EW", "periodic_NS"],
              doc="Grid edges: normal, or periodic east-west / north-south (a periodic "
                  "edge never lets water out)."),
        Param("local_minima", "string",
              label="Depression routing", default="carve_cordonnier", choices=_GF_LOCAL_MINIMA,
              doc="How depressions of the water surface are routed through."),
        Param("outlet_edges", "string", default="all", choices=list(_OUTLET_SIDES),
              doc="Grid edges whose cells let water out of the domain."),
        Param("nodata_outlets", "bool", label="Cells next to nodata drain out", default=True,
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
_STOP_TEXT = {"tolerance": "tolerance reached", "plateau": "plateau"}


def _num(v, unit=""):
    """``v`` with four significant digits, and its unit."""
    return f"{v:.4g}" + (f" {unit}" if unit else "")


def _convergence_params(unit, check_default):
    """Params of the convergence checks and the stop rule; ``unit``: what
    is counted (steps, launches)."""
    return [
        Param("stop", "string", default="count", choices=["count", "converged"],
              choice_labels=["Fixed count", "Until converged (count is the maximum)"],
              doc=f"count: run all the {unit}; converged: stop earlier once the "
                  "convergence metric is below the tolerance or stops improving."),
        Param("check_every", "int", label="Check every (0 = end)", unit=unit,
              default=check_default, min=0,
              doc=f"Convergence check (one report row) every this many {unit}; 0: one "
                  "check at the end (fixed count only). Each check is one pass over the "
                  "grid."),
        Param("convergence_metric", "string",
              label="Metric", default="dh_p99", choices=_CONV_METRICS,
              doc="Watched metric: a quantile (or max / mean) of |dh|, the depth change "
                  "(m) that would pass each checked cell's discharge against its "
                  "receiver's head; or the discharge residual sum|Q - Qout| / sum Q."),
        Param("convergence_tol", "float", label="Tolerance (m for dh)", default=1e-3, min=0.0,
              doc="Stop once the metric is below this (m for dh, a ratio for residual)."),
        Param("convergence_window", "int",
              label="Plateau window", unit="checks", default=10, min=1,
              doc="Plateau: stop once the best of the last this many checks is not "
                  "lower than (1 - eps) x the best before them."),
        Param("convergence_eps", "float",
              label="Plateau improvement", default=0.01, min=0.0, max=1.0,
              doc="Plateau: smallest relative improvement over the window."),
        Param("convergence_percentile", "float",
              label="Checked cells: discharge percentile", default=90.0, min=0.0, max=100.0,
              doc="Checked cells: valid non-outlet cells whose discharge is at or above "
                  "this percentile of the positive discharge (set at the first check)."),
        Param("convergence_memory", "float", label="Drift memory", default=0.8, min=0.0, max=1.0,
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
        self.reason = None
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
        self.reason = reason
        return reason is not None

    def _metric_text(self):
        """The watched metric of the last check as text ("" when none)."""
        if not self.last or math.isnan(self.last.get(self.metric, math.nan)):
            return ""
        v = self.last[self.metric]
        if self.metric.startswith("dh"):
            shown = f"{v * 1000:.3g} mm" if v < 1.0 else f"{v:.3g} m"
        else:
            shown = f"{v:.3g}"
        return f"{self.metric.replace('_', ' ')} {shown}"

    def phase(self, text):
        """``text``, plus the watched metric of the last check."""
        metric = self._metric_text()
        return f"{text}  ·  {metric}" if metric else text

    def status(self, *extra):
        """Runner status (see core.runner) from the last check: the count,
        the watched metric, the ``extra`` texts and why the run stopped."""
        if self.unit not in self.rows:
            return {"text": "", "stopped": False}
        parts = [f"{self.unit.capitalize()} {int(self.rows[self.unit][-1])}"]
        parts += [t for t in (self._metric_text(), *extra) if t]
        text = ", ".join(parts) + "."
        if self.reason is not None:
            text += f" Converged: {_STOP_TEXT[self.reason]}."
        return {"text": text, "stopped": self.reason is not None}

    def table(self, title, units):
        return DataTable({k: np.asarray(v, dtype=np.float64) for k, v in self.rows.items()},
                         units={k: u for k, u in units.items() if k in self.rows},
                         roles={self.unit: "x"}, title=title)


#: Units of the report columns.
_CONV_UNITS = {"dh_mean": "m", "dh_p50": "m", "dh_p90": "m", "dh_p99": "m", "dh_max": "m",
               "h_max": "m", "time": "s"}


@process(
    id="pyfastflow.graphflood",
    category="Flood",
    label="GraphFlood (water depth and discharge)",
    inputs=[
        Port("dem", "georaster", label="Topography",
             doc="Topography (projected, metres); nodata cells stay out."),
        Port("h_init", "georaster", label="Start from water depth", unit="m", optional=True,
             doc="Water depth (m) above the DEM to start from (e.g. a previous run's "
                 "water_depth); replaces initial_water's depth. Same grid as the DEM."),
        Port("precipitation_field", "georaster",
             label="Rainfall field", unit="mm/h", optional=True,
             doc="Rainfall per cell (mm/h), replaces Rainfall; same grid as the DEM."),
        Port("manning_field", "georaster", label="Manning n field", optional=True,
             doc="Manning n per cell (e.g. from land cover), replaces Manning; same grid "
                 "as the DEM."),
        _source_field_port(),
        _sources_port(),
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
        Param("analytical_relaxation", "float", label="Relaxation", default=0.1, min=0.0, max=1.0,
              doc="Analytical solver: fraction of the way to the target depth per step."),
        Param("analytical_solver", "string", label="Analytical solve", default="bottom_up",
              choices=["bottom_up", "local"],
              doc="Analytical solver: bottom_up (receivers first, each depth solved "
                  "against its receiver's head) or local (friction depth at the frozen "
                  "steepest slope)."),
        Param("dt", "float", label="Time step dt", unit="s", default=1e-3, min=0.0,
              doc="Explicit and transient solvers: time step (s)."),
        *_convergence_params("steps", 25),
        Param("return_discharge", "bool",
              label="Also discharge Qin", default=False, doc="Also return Qin."),
        Param("return_outflow", "bool",
              label="Also outflow Qout", default=False, doc="Also return Qout."),
        Param("return_imbalance", "bool",
              label="Also Qout / Qin - 1", default=False, doc="Also return Qout / Qin - 1."),
        Param("return_report", "bool", label="Also the report table", default=False,
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
def graphflood(dem, h_init=None, precipitation_field=None, manning_field=None, source_field=None,
               sources=None,
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
    inputs = {k: args.pop(k) for k in ("dem", "h_init", "precipitation_field", "manning_field",
                                       "source_field", "sources")}
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


@runner("pyfastflow.graphflood", axis="step", config=(
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
        self.rain_field = inputs.get("precipitation_field")
        self.stamps = _source_stamps(inputs.get("sources"), self.dem, self.valid, who)
        self.source_cells = _source_cells(inputs.get("source_field"), self.valid, who)
        fields, _ = _split_fields(
            precipitation=self._rain(p["rainfall"]),
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
        self.h_max = 0.0
        self.dt = float(p["dt"])
        self.run = _make_counter(getattr(prog, pipeline), self, "done",
                                 dt=(lambda: self.dt) if self.kind == "vanilla" else None)

    def _rain(self, rate):
        """Rain (m/s) of ``rate`` (mm/h) or the field, sources added (a field then)."""
        return _rain_with_sources(_rain(self.rain_field, rate, self.shape, self.who),
                                  self.stamps, self.shape, self.dx, self.source_cells)

    def flood_program(self):
        return self.flood

    def advance(self, p):
        who, flood = self.who, self.flood
        steps = int(p["steps"])
        if steps < 1:
            raise ValueError(f"{who}: steps must be >= 1")
        scalars = {"friction_exponent": float(p["friction_exponent"]),
                   "carve_slope_min": float(p["carve_slope_min"])}
        if "precipitation" not in self.fields:
            scalars["precipitation"] = float(p["rainfall"]) * _MM_PER_H
        else:  # the rate of linked sources, or a field changed since (see FloodRun)
            self.flood_program().precipitation.from_numpy(self._rain(p["rainfall"]))
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
            self.h_max = extra["h_max"]
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

    def status(self):
        extra = [f"max depth {_num(self.h_max, 'm')}"] if self.done else []
        if self.kind == "vanilla":
            extra.insert(0, f"t = {_num(self.time, 's')}")
        return self.checks.status(*extra)

    def close(self):
        self.flood.close()


@process(
    id="pyfastflow.graphflood_particle",
    category="Flood",
    label="GraphFlood by particles (steady water depth and discharge)",
    inputs=[
        Port("dem", "georaster", label="Topography",
             doc="Topography (projected, metres); nodata cells stay out."),
        Port("h_init", "georaster", label="Start from water depth", unit="m", optional=True,
             doc="Water depth (m) above the DEM to start from, before the warm-up (e.g. a "
                 "GraphFlood run's water_depth). Same grid as the DEM."),
        Port("precipitation_field", "georaster",
             label="Rainfall field", unit="mm/h", optional=True,
             doc="Rainfall per cell (mm/h), replaces Rainfall; same grid as the DEM."),
        Port("manning_field", "georaster", label="Manning n field", optional=True,
             doc="Manning n per cell (e.g. from land cover), replaces Manning; same grid "
                 "as the DEM."),
        _source_field_port(),
        _sources_port(),
    ],
    params=[
        Param("launches", "int", default=1, min=1,
              doc="Particle launches; a report row after each."),
        Param("particles", "int", label="Particles per launch", default=1_000_000, min=1,
              doc="Particles released per launch."),
        *_flood_common_params(),
        Param("source_percentile", "float",
              label="Source percentile of discharge", default=99.0, min=0.0, max=100.0,
              doc="Sources: valid cells whose warm-up discharge is at or above this "
                  "percentile of the non-zero discharge; particles spawn on them and "
                  "their steepest paths downstream (the active area)."),
        Param("h_relaxation", "float", label="Depth relaxation", default=0.01, min=0.0, max=1.0,
              doc="Fraction of the way to the target depth taken per visit."),
        Param("walk_steps", "int", default=100, min=1,
              doc="Most cells walked by one particle."),
        Param("spawn_pad", "int", label="Spawn padding", unit="cells", default=2, min=0,
              doc="Particles start up to this many rows/columns from the active area."),
        Param("propagate", "float", label="Propagation threshold (0 = off)", default=0.5, min=0.0,
              doc="Cells whose inflow a visit changed by more than this fraction are "
                  "processed by the same thread before its next particle (0: off)."),
        Param("seed", "int", default=1, min=0, doc="Random seed of the particle starts."),
        Param("threads", "int", label="GPU threads", default=32768, min=32,
              doc="GPU threads walking particles."),
        Param("return_discharge", "bool",
              label="Also discharge Qin", default=False, doc="Also return Qin."),
        Param("return_outflow", "bool",
              label="Also outflow Qout", default=False, doc="Also return Qout."),
        Param("return_imbalance", "bool",
              label="Also Qout / Qin - 1", default=False, doc="Also return Qout / Qin - 1."),
        *_convergence_params("launches", 1),
        Param("return_active_area", "bool", label="Also the active area", default=False,
              doc="Also return the active area of the warm-up (1, else 0)."),
        Param("return_report", "bool", label="Also the report table", default=False,
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
                        source_field=None, sources=None,
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
    inputs = {k: args.pop(k) for k in ("dem", "h_init", "precipitation_field", "manning_field",
                                       "source_field", "sources")}
    return run_once(GraphFloodParticleRun, inputs, args)


@runner("pyfastflow.graphflood_particle", axis="launch", config=(
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
        self.rain_field = inputs.get("precipitation_field")
        self.stamps = _source_stamps(inputs.get("sources"), self.dem, self.valid, who)
        self.source_cells = _source_cells(inputs.get("source_field"), self.valid, who)
        fields, _ = _split_fields(
            precipitation=self._rain(p["rainfall"]),
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
        self.h_max = 0.0

    def _rain(self, rate):
        """Rain (m/s) of ``rate`` (mm/h) or the field, sources added (a field then)."""
        return _rain_with_sources(_rain(self.rain_field, rate, self.shape, self.who),
                                  self.stamps, self.shape, self.dx, self.source_cells)

    def flood_program(self):
        return self.gfp

    def advance(self, p):
        who, gfp = self.who, self.gfp
        scalars = {"friction_exponent": float(p["friction_exponent"]),
                   "carve_slope_min": float(p["carve_slope_min"]),
                   "relaxation": float(p["h_relaxation"]), "propagate": float(p["propagate"]),
                   "n_particles": int(p["particles"]), "walk_steps": int(p["walk_steps"]),
                   "spawn_pad": int(p["spawn_pad"]), "seed": int(p["seed"])}
        if "precipitation" not in self.fields:
            scalars["precipitation"] = float(p["rainfall"]) * _MM_PER_H
        else:  # the rate of linked sources, or a field changed since (see FloodRun)
            self.flood_program().precipitation.from_numpy(self._rain(p["rainfall"]))
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
            self.h_max = float(gfp.h.array.max())
            if self.checks.check(gfp, self.launched, h_max=self.h_max,
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

    def status(self):
        return self.checks.status(f"max depth {_num(self.h_max, 'm')}" if self.launched else "")

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
    category="Flood",
    label="Inertial flood (transient water depth and discharge)",
    inputs=[
        Port("dem", "georaster", label="Topography",
             doc="Topography (projected, metres); nodata cells stay out."),
        Port("h_init", "georaster", label="Start from water depth", unit="m", optional=True,
             doc="Water depth (m) to start from, flow at rest (e.g. a GraphFlood steady "
                 "state). Same grid as the DEM. Ignored when previous_state is given."),
        Port("previous_state", "pyfastflow.InertialFloodState", optional=True,
             doc="A previous run's state: continues it exactly (depth, discharges, model "
                 "time, max depth)."),
        Port("precipitation_field", "georaster",
             label="Rainfall field", unit="mm/h", optional=True,
             doc="Rainfall per cell (mm/h), replaces Rainfall; same grid as the DEM."),
        _source_field_port(),
        _sources_port(),
    ],
    params=[
        Param("duration", "float", unit="s", default=3600.0, min=0.0,
              doc="Model time to simulate (s), from the start (or previous_state's time)."),
        Param("rainfall", "float", unit="mm/h", default=50.0, min=0.0,
              doc="Rainfall rate (mm/h), when no precipitation field is given; times the "
                  "rain series factor."),
        Param("rain_series", "string", label="Rain factor over time", default="",
              doc="Rain factor over model time, \"t:factor, t:factor, ...\" (s); empty: 1. "
                  "Before the first time and after the last, the end factors hold. It "
                  "scales the rain and the linked sources, not the fixed ones."),
        Param("series_interp", "string", label="Between series times", default="step",
              choices=["step", "linear"],
              choice_labels=["Hold each factor until the next time", "Linear"],
              doc="step: a factor holds from its time on; linear: interpolated."),
        Param("storm_duration", "float",
              label="Rain stops at (0 = never)", unit="s", default=0.0, min=0.0,
              doc="Rain stops at this model time (s; counted from the first run of a "
                  "chain); 0: rain all along."),
        Param("manning", "float", label="Manning n", default=0.033, min=0.0,
              doc="Manning roughness n (s/m^(1/3)); one value for the grid."),
        Param("time_step", "string", default="auto", choices=["auto", "fixed"],
              choice_labels=["Automatic (Bates stability limit)", "Fixed"],
              doc="auto: dt = min(dt_max, cfl dx / sqrt(g h_max)), recomputed every "
                  "dt_update_steps steps; fixed: dt."),
        Param("dt", "float",
              label="Fixed dt", unit="s", default=1.0, min=0.0, doc="Fixed time step (s)."),
        Param("dt_max", "float", label="Largest dt", unit="s", default=10.0, min=0.0,
              doc="Automatic time step: upper bound (s), also used while the grid is dry."),
        Param("cfl", "float", label="Courant number", default=0.7, min=0.0, max=1.0,
              doc="Automatic time step: Courant number of the Bates limit."),
        Param("dt_update_steps", "int", label="Steps between dt updates", default=25, min=1,
              doc="Steps between two time-step updates (also between max-depth samples)."),
        Param("topology", "string", label="Links", default="D4", choices=["D4", "D8"],
              doc="Links: D4, or D8 (each of the 8 links dx / 2 wide; about 1.5x the "
                  "memory)."),
        Param("boundary", "string", label="Boundaries", default="normal",
              choices=["normal", "periodic_EW", "periodic_NS"],
              doc="Grid edges: normal, or periodic east-west / north-south (a periodic "
                  "edge never lets water out)."),
        Param("outlet_edges", "string", default="all", choices=list(_OUTLET_SIDES),
              doc="Grid edges whose cells are outlets."),
        Param("nodata_outlets", "bool", label="Cells next to nodata are outlets", default=True,
              doc="Valid cells touching nodata are outlets too."),
        Param("fix_outlet_depth", "bool", label="Hold outlet depth", default=True,
              doc="Outlet cells are held at Outlet depth (water reaching them leaves)."),
        Param("outlet_depth", "float", unit="m", default=0.0, min=0.0,
              doc="Depth (m) held on outlet cells."),
        Param("boundary_flux", "float", label="Edge outflow", unit="m²/s", default=0.0,
              doc="Outward unit discharge (m²/s) on the grid-edge links of outlet cells."),
        Param("outlet_discharge", "float", label="Outlet sink", unit="m³/s per cell", default=0.0,
              doc="Discharge (m³/s) taken out of every outlet cell (when the outlet depth "
                  "is not fixed)."),
        Param("top_inflow", "float",
              label="Top-row inflow", unit="m³/s per cell", default=0.0, min=0.0,
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
        Param("min_flow_depth", "float", unit="m", default=1e-4, min=0.0,
              doc="Links with a mean depth at or below this (m) carry nothing."),
        Param("min_depth", "float", label="Depth floor", unit="m", default=0.0, min=0.0,
              doc="Floor of the depth (m)."),
        Param("gravity", "float", unit="m/s²", default=9.81, min=0.0, doc="Gravity (m/s²)."),
        Param("report_every", "float",
              label="Report every (0 = end)", unit="s", default=0.0, min=0.0,
              doc="Report row every this much model time (s); 0: at the end only."),
        Param("return_discharge", "bool", label="Also discharge", default=False,
              doc="Also return the cell discharge (m³/s)."),
        Param("return_velocity", "bool", label="Also velocity", default=False,
              doc="Also return the flow speed (m/s)."),
        Param("return_max_depth", "bool", label="Also max depth", default=False,
              doc="Also return the maximum depth reached (m)."),
        Param("return_state", "bool", default=False,
              doc="Also return the state, to continue the run."),
        Param("return_report", "bool", label="Also the report table", default=False,
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
                   source_field=None, sources=None, duration=3600.0,
                   rainfall=50.0, rain_series="", series_interp="step", storm_duration=0.0, manning=0.033, time_step="auto",
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
                                       "precipitation_field", "source_field", "sources")}
    return run_once(InertialFloodRun, inputs, args)


@runner("pyfastflow.inertial_flood", axis="time", config=(
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
        self.stamps = _source_stamps(inputs.get("sources"), self.dem, self.valid, who)
        self.source_cells = _source_cells(inputs.get("source_field"), self.valid, who)
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

    def _rain(self, rate, raining=True, factor=1.0):
        """The rain field (m/s, 0 on nodata) for ``rate`` (mm/h) or the field
        input times ``factor`` (the rain series'), the sources and the source
        field added; after the storm (``raining`` False) only the sources of a
        fixed q and the source field."""
        shape = self.shape
        rain = _rain(self.rain_field, rate, shape, self.who) * factor if raining else 0.0
        rain = _rain_with_sources(rain, self.stamps, shape, self.dx, self.source_cells)
        return np.where(self.valid, np.broadcast_to(rain, shape), 0.0).astype(np.float32)

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
        series = _fusw_series(p["rain_series"], "rain", who)
        interp = p["series_interp"]
        if interp not in ("step", "linear"):
            raise ValueError(f"{who}: unknown series_interp {interp!r}")
        last = {}  # the last factor asked: (field, m³/s)

        def rain_at(factor):
            if factor not in last:
                field = self._rain(p["rainfall"], factor=factor)
                if np.array_equal(field, self.rain_now):
                    field = self.rain_now
                last.clear()
                last[factor] = (field, float(field.sum(dtype=np.float64)) * cell)
            return last[factor]

        rain_off = self._rain(0.0, raining=False)
        off_rate = float(rain_off.sum(dtype=np.float64)) * cell
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
            rain_on, rain_rate = (rain_at(_series_factor(series, interp, t)) if raining
                                  else (rain_off, off_rate))
            self._set_rain(rain_on)
            if time_step == "auto":
                h_top = float(h.max())
                step_dt = float(p["dt_max"]) if h_top <= min_flow else min(
                    float(p["dt_max"]), float(p["cfl"]) * dx / np.sqrt(g * h_top))
            stop = min(end, self.next_row, storm if raining and storm > 0.0 else end)
            if raining and len(series[0]) and interp == "step":
                ahead = series[0][series[0] > t + eps]
                if len(ahead):  # the factor changes there
                    stop = min(stop, float(ahead[0]))
            n = min(max(1, int(np.ceil((stop - t) / step_dt - 1e-9))), every)
            landing = n * step_dt >= stop - t - eps
            if landing:
                step_dt = (stop - t) / n  # land exactly on the stop (smaller: stable)
            flood.dt.set(step_dt)
            flood.step(n)
            elapsed = n * step_dt
            self.time = stop if landing else t + elapsed
            self.step_dt = step_dt
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

    def status(self):
        rows = self.rows
        if not rows["time"]:
            return {"text": "", "stopped": False}
        return {"text": f"t = {_num(rows['time'][-1], 's')}, max depth "
                        f"{_num(rows['h_max'][-1], 'm')}, lost {_num(rows['lost'][-1], 'm³')}.",
                "stopped": False}

    def close(self):
        self.flood.close()


# ---------------------------------------------------------------------------
# FullSWOF flood (FuswProgram)
# ---------------------------------------------------------------------------

@dataclass
class FuswFloodState:
    """What a FullSWOF flood run needs to continue exactly: depth (m),
    velocities u (east) and v (towards growing rows, south) (m/s), the
    infiltrated depth (m; None without infiltration), model time (s), the
    running maximum depth (m), the solver's cumulative budget vector and the
    cell size."""

    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    infiltrated: Optional[np.ndarray]
    time: float
    max_depth: np.ndarray
    budget: np.ndarray
    dx: float


register_type("pyfastflow.FuswFloodState", "state",
              lambda v: isinstance(v, FuswFloodState),
              doc="FullSWOF flood state (depth, velocities, infiltration, time, budget) to "
                  "continue a run.")

#: Grid edges, in the order they win a cell they share (a corner).
_FUSW_EDGES = ("north", "south", "west", "east")
#: Face bit of each side in ``boundary_faces``.
_FUSW_BIT = {"north": 1, "west": 2, "east": 4, "south": 8}
_FUSW_EDGE_CODES = ["wall", "neumann", "stage", "critical", "normal_depth", "rating",
                    "discharge", "sink"]
_FUSW_EDGE_LABELS = ["Wall (closed)", "Free (zero gradient)", "Imposed water level",
                     "Critical (free outfall)", "Normal depth (uniform flow)", "Rating curve",
                     "Imposed discharge", "Sink (absorbs the water)"]
_FUSW_NODATA_CODES = ["wall", "neumann", "critical", "sink"]
_FUSW_NODATA_LABELS = ["Wall (closed)", "Free (zero gradient)", "Critical (free outfall)",
                       "Sink (absorbs the water)"]
#: Per-edge values: param suffix -> (program param, codes using it).
_FUSW_EDGE_VALUES = {
    "stage": ("stage", ("stage",)),
    "discharge": ("inflow", ("discharge",)),
    "outlet_slope": ("outlet_slope", ("normal_depth",)),
    "outlet_manning": ("outlet_manning", ("normal_depth",)),
    "rating_a": ("rating_a", ("rating",)),
    "rating_b": ("rating_b", ("rating",)),
    "rating_eta0": ("rating_eta0", ("rating",)),
}
#: Green-Ampt params: name -> (default, unit, mm/h rate to convert to m/s).
_FUSW_GREEN_AMPT = {
    "ks": (10.0, "mm/h", True), "kc": (0.0, "mm/h", True), "dtheta": (0.3, "", False),
    "psi": (0.1, "m", False), "zcrust": (0.0, "m", False), "imax": (1000.0, "mm/h", True),
}
_FUSW_FRICTIONS = ["none", "manning", "darcy_weisbach", "laminar"]
_FUSW_FRICTION_LABELS = ["None", "Manning (n)", "Darcy-Weisbach (f)", "Laminar"]
_FUSW_FLUXES = ["rusanov", "hll", "hll2", "hllc", "hllc2"]
_FUSW_LIMITERS = ["minmod", "vanalbada", "vanleer"]
_FUSW_SERIES = ("rain", "inflow", "source")
#: Steps between two reads of the solver (time, max depth, probes).
_FUSW_RECORD_ROWS = 1024


def _fusw_series(text, name, who):
    """(times, factors) of a series written "t0:f0, t1:f1, ..." (s, factor);
    empty text: none (the factor stays 1)."""
    text = str(text or "").strip()
    if not text:
        return np.zeros(0), np.zeros(0)
    times, factors = [], []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            t, f = (float(x) for x in item.split(":"))
        except ValueError:
            raise ValueError(f"{who}: {name} series: {item!r} is not time:factor") from None
        times.append(t)
        factors.append(f)
    if np.any(np.diff(times) < 0.0):
        raise ValueError(f"{who}: {name} series: the times must increase")
    return np.asarray(times), np.asarray(factors)


def _fusw_exposed(valid, boundary):
    """Per side, the active cells whose D4 neighbour on that side is outside
    a non-periodic grid edge (``[side]["edge"]``) or is a nodata cell
    (``[side]["nodata"]``)."""
    ny, nx = valid.shape
    out = {}
    for side, axis, shift in (("north", 0, 1), ("south", 0, -1), ("west", 1, 1),
                              ("east", 1, -1)):
        periodic = boundary == ("periodic_NS" if axis == 0 else "periodic_EW")
        neighbour = np.roll(valid, shift, axis=axis)
        edge = np.zeros_like(valid)
        index = 0 if shift == 1 else -1
        if axis == 0:
            edge[index, :] = True
        else:
            edge[:, index] = True
        if periodic:
            nodata = valid & ~neighbour
            edge[:] = False
        else:
            nodata = valid & ~neighbour & ~edge
            edge &= valid
        out[side] = {"edge": edge, "nodata": nodata}
    return out


def _fusw_codes(exposed, edge_codes, nodata_code):
    """(boundary_type, boundary_faces, edge_of): each cell's code, its
    selected faces and the edge whose values it takes (index in
    _FUSW_EDGES, -1: none). A cell with boundary faces of several conditions
    takes the first non-wall one of its grid-edge faces (in _FUSW_EDGES
    order), else the nodata one; only the faces of that condition are
    selected, the others act as walls. Cells without a boundary face keep
    wall and every face selected (FuswProgram.initialize's default)."""
    shape = exposed["north"]["edge"].shape
    codes = np.full(shape, FUSW_CODES["wall"], dtype=np.uint8)
    faces = np.full(shape, 15, dtype=np.uint8)
    edge_of = np.full(shape, -1, dtype=np.int64)
    done = np.zeros(shape, dtype=bool)
    for k, side in enumerate(_FUSW_EDGES):
        if edge_codes[side] == "wall":
            continue
        take = exposed[side]["edge"] & ~done
        codes[take] = FUSW_CODES[edge_codes[side]]
        edge_of[take] = k
        done |= take
    if nodata_code != "wall":
        take = np.any([exposed[s]["nodata"] for s in _FUSW_EDGES], axis=0) & ~done
        codes[take] = FUSW_CODES[nodata_code]
        done |= take
    bits = np.zeros(shape, dtype=np.uint8)
    for side in _FUSW_EDGES:
        same_edge = exposed[side]["edge"] & (codes == FUSW_CODES[edge_codes[side]])
        same_nodata = exposed[side]["nodata"] & (codes == FUSW_CODES[nodata_code])
        bits[same_edge | same_nodata] |= np.uint8(_FUSW_BIT[side])
    faces[done] = bits[done]
    return codes, faces, edge_of


def _fusw_line_faces(lines, dem, exposed, who):
    """Per feature of ``lines`` (in the DEM's CRS), the boundary faces it
    follows: (rows, cols, sides) of the active cells within one cell of the
    line that have a boundary face, each with its boundary face nearest to
    the line. ValueError for a line with none, or cells in two lines."""
    from scipy.ndimage import binary_dilation

    cs, (ny, nx) = dem.cell_size, dem.z.shape
    xy = lines.in_crs(dem.crs_wkt or dem.epsg).xy
    # Grid coordinates: column, row (row 0 north), in cells.
    gx, gy = (xy[:, 0] - dem.x_min) / cs, (dem.y_max - xy[:, 1]) / cs
    has_face = np.zeros((ny, nx), dtype=bool)
    for side in _FUSW_EDGES:
        has_face |= exposed[side]["edge"] | exposed[side]["nodata"]
    # Face midpoints relative to the cell's corner, in cells.
    mid = {"north": (0.5, 0.0), "south": (0.5, 1.0), "west": (0.0, 0.5), "east": (1.0, 0.5)}
    side_letter = {"north": "N", "south": "S", "west": "W", "east": "E"}
    used = np.zeros((ny, nx), dtype=bool)
    out = []
    parts = lines.part_offsets
    for f in range(lines.n_features):
        segments = []
        for j in range(lines.feature_offsets[f], lines.feature_offsets[f + 1]):
            px, py = gx[parts[j]:parts[j + 1]], gy[parts[j]:parts[j + 1]]
            segments += [(px[i], py[i], px[i + 1], py[i + 1]) for i in range(len(px) - 1)]
        if not segments:
            raise ValueError(f"{who}: inflow line {f + 1} has fewer than two vertices")
        seg = np.asarray(segments)
        near = np.zeros((ny, nx), dtype=bool)
        for x0, y0, x1, y1 in seg:
            n = max(2, int(np.ceil(4.0 * np.hypot(x1 - x0, y1 - y0))) + 1)
            t = np.linspace(0.0, 1.0, n)
            r = np.clip(np.floor(y0 + t * (y1 - y0)).astype(int), 0, ny - 1)
            c = np.clip(np.floor(x0 + t * (x1 - x0)).astype(int), 0, nx - 1)
            near[r, c] = True
        near = binary_dilation(near, structure=np.ones((3, 3), dtype=bool)) & has_face
        rows, cols, sides = [], [], []
        for r, c in zip(*np.nonzero(near)):
            best, best_d = None, 1.0 + 1e-9
            for side in _FUSW_EDGES:
                if not (exposed[side]["edge"][r, c] or exposed[side]["nodata"][r, c]):
                    continue
                d = _segments_distance(c + mid[side][0], r + mid[side][1], seg)
                if d <= best_d:
                    best, best_d = side, d
            if best is not None:
                rows.append(r)
                cols.append(c)
                sides.append(side_letter[best])
        if not rows:
            raise ValueError(f"{who}: inflow line {f + 1} does not follow a grid edge or a "
                             "nodata border (no boundary face within one cell of it)")
        rows, cols = np.asarray(rows), np.asarray(cols)
        if used[rows, cols].any():
            raise ValueError(f"{who}: inflow line {f + 1} shares cells with another line")
        used[rows, cols] = True
        out.append((rows, cols, np.asarray(sides)))
    return out


def _segments_distance(x, y, seg):
    """Distance from (x, y) to the nearest of the segments (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = seg.T
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(length2 > 0.0, ((x - x0) * dx + (y - y0) * dy) / length2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return float(np.min(np.hypot(x0 + t * dx - x, y0 + t * dy - y)))


def _fusw_feature_value(lines, f, name, default):
    """Feature attribute ``name`` of feature ``f`` when numeric and set, else
    ``default``."""
    values = lines.feature_attrs.get(name)
    if values is None:
        return float(default)
    try:
        v = float(values[f])
    except (TypeError, ValueError):
        return float(default)
    return float(default) if math.isnan(v) else v


def _fusw_edge_params():
    """Per grid edge: its boundary code and the values the codes read."""
    out = []
    for side in _FUSW_EDGES:
        name = side.capitalize()
        out += [
            Param(f"{side}_boundary", "string", label=f"{name} edge", default="neumann",
                  choices=_FUSW_EDGE_CODES, choice_labels=_FUSW_EDGE_LABELS,
                  doc=f"Boundary condition of the {side} grid edge (FullSWOF): wall; free "
                      "(zero gradient); imposed water level; critical outfall; normal depth "
                      "(uniform flow); rating curve; imposed discharge; sink."),
            Param(f"{side}_stage", "float", label=f"{name}: water level", unit="m",
                  default=0.0,
                  doc="Imposed water level: water-surface elevation (m) beyond the edge."),
            Param(f"{side}_discharge", "float", label=f"{name}: discharge", unit="m³/s",
                  default=0.0,
                  doc="Imposed discharge: total discharge (m³/s) into the domain through "
                      "the edge, the same per metre on each of its faces (times the inflow "
                      "series factor; negative: out)."),
            Param(f"{side}_outlet_slope", "float", label=f"{name}: energy slope",
                  default=0.0, min=0.0,
                  doc="Normal depth: energy slope of the uniform flow (0: the bed slope "
                      "towards the edge)."),
            Param(f"{side}_outlet_manning", "float", label=f"{name}: Manning n",
                  default=0.033, min=0.0,
                  doc="Normal depth: Manning n of the uniform flow."),
            Param(f"{side}_rating_a", "float", label=f"{name}: rating a", default=1.0,
                  min=0.0,
                  doc="Rating curve q = a (stage - sill)^b (m²/s), stage counted from the "
                      "sill or the cell bed, whichever is higher."),
            Param(f"{side}_rating_b", "float", label=f"{name}: rating b", default=1.5,
                  min=0.0, doc="Rating curve exponent b."),
            Param(f"{side}_rating_eta0", "float", label=f"{name}: rating sill", unit="m",
                  default=0.0, doc="Rating curve sill elevation (m)."),
        ]
    return out


_FUSW_FIELD_PORTS = [
    Port("dem", "georaster", label="Topography",
         doc="Bed elevation (projected, metres); nodata cells are closed."),
    Port("h_init", "georaster", label="Start from water depth", unit="m", optional=True,
         doc="Water depth (m) to start from, at rest. Same grid as the DEM. Ignored when "
             "previous_state is given."),
    Port("previous_state", "pyfastflow.FuswFloodState", optional=True,
         doc="A previous run's state: continues it exactly (depth, velocities, "
             "infiltration, model time, budget, max depth)."),
    Port("precipitation_field", "georaster", label="Rainfall field", unit="mm/h",
         optional=True, doc="Rainfall per cell (mm/h), replaces Rainfall; same grid as the DEM."),
    Port("friction_field", "georaster", label="Friction coefficient field", optional=True,
         doc="Friction coefficient per cell (the law's: Manning n, Darcy f, laminar), "
             "replaces Friction coefficient; same grid as the DEM."),
    Port("source_field", "georaster", label="Source field", unit="m³/s per cell",
         optional=True,
         doc="Water added to each cell (m³/s; negative: taken out), times the source "
             "series factor; same grid as the DEM."),
    Port("inflow_lines", "geolines", label="Inflow lines", optional=True,
         doc="Lines along the grid edge or a nodata border, each carrying Inflow discharge "
             "(or its feature attribute 'discharge', m³/s) into the domain, shared by its "
             "faces as uniform flow under one water level (energy slope Inflow slope or "
             "the attribute 'slope'). A line takes the boundary faces of the cells within "
             "one cell of it."),
    _sources_port(),
    Port("probes", "geopoints", label="Probe points", optional=True,
         doc="Cells whose depth and velocities are recorded over time (output probe_table)."),
] + [
    Port(f"{name}_field", "georaster", label=f"Green-Ampt {name} field",
         unit=unit, optional=True,
         doc=f"Green-Ampt {name} per cell ({unit or 'ratio'}), replaces the scalar; same "
             "grid as the DEM.")
    for name, (_d, unit, _r) in _FUSW_GREEN_AMPT.items()
]

_FUSW_RETURNS = ("velocity", "unit_discharge", "discharge", "water_surface", "max_depth",
                 "infiltrated", "shear_friction", "shear_surface", "froude",
                 "acceleration", "state", "report")


@process(
    id="pyfastflow.fusw_flood",
    category="Flood",
    label="FullSWOF flood (shallow-water finite volumes)",
    inputs=_FUSW_FIELD_PORTS,
    params=[
        Param("duration", "float", unit="s", default=3600.0, min=0.0,
              doc="Model time to simulate (s), from the start (or previous_state's time)."),
        Param("rainfall", "float", unit="mm/h", default=50.0, min=0.0,
              doc="Rainfall rate (mm/h), when no precipitation field is given; times the "
                  "rain series factor."),
        Param("rain_series", "string", label="Rain factor over time", default="",
              doc="Rain factor over model time, \"t:factor, t:factor, ...\" (s); empty: 1. "
                  "Before the first time and after the last, the end factors hold."),
        Param("inflow_series", "string", label="Inflow factor over time", default="",
              doc="Factor of the imposed edge discharges and the inflow lines over model "
                  "time, as Rain factor over time."),
        Param("source_series", "string", label="Source factor over time", default="",
              doc="Factor of the source field over model time, as Rain factor over time."),
        Param("series_interp", "string", label="Between series times", default="step",
              choices=["step", "linear"],
              choice_labels=["Hold each factor until the next time", "Linear"],
              doc="step: a factor holds from its time on; linear: interpolated."),
        Param("friction", "string", label="Friction law", default="manning",
              choices=_FUSW_FRICTIONS, choice_labels=_FUSW_FRICTION_LABELS,
              doc="Semi-implicit bed friction: none, Manning (coefficient n), "
                  "Darcy-Weisbach (friction factor f) or laminar."),
        Param("friction_coefficient", "float", label="Friction coefficient", default=0.033,
              min=0.0, doc="Coefficient of the friction law (Manning n in s/m^(1/3), "
                           "Darcy-Weisbach f, laminar), when no field is given."),
        Param("infiltration", "string", default="none", choices=["none", "green_ampt"],
              choice_labels=["None", "Green-Ampt"],
              doc="Infiltration: none, or Green-Ampt with an optional crust (FullSWOF)."),
        Param("ks", "float", label="Ks (saturated conductivity)", unit="mm/h",
              default=10.0, min=0.0, doc="Green-Ampt saturated hydraulic conductivity (mm/h)."),
        Param("kc", "float", label="Kc (crust conductivity)", unit="mm/h", default=0.0,
              min=0.0, doc="Green-Ampt conductivity of the crust (mm/h)."),
        Param("dtheta", "float", label="Δθ (moisture deficit)", default=0.3, min=0.0,
              max=1.0, doc="Green-Ampt water content deficit (saturated minus initial)."),
        Param("psi", "float", label="ψ (suction at the front)", unit="m", default=0.1,
              min=0.0, doc="Green-Ampt suction head at the wetting front (m)."),
        Param("zcrust", "float", label="Crust thickness (0 = none)", unit="m",
              default=0.0, min=0.0, doc="Thickness of the crust (m); 0: no crust."),
        Param("imax", "float", label="Max infiltration rate", unit="mm/h", default=1000.0,
              min=0.0, doc="Upper bound of the infiltration rate (mm/h)."),
        Param("boundary", "string", label="Periodic edges", default="normal",
              choices=["normal", "periodic_EW", "periodic_NS"],
              choice_labels=["None", "East-west", "North-south"],
              doc="Periodic grid edges (their boundary conditions are then unused)."),
        *_fusw_edge_params(),
        Param("nodata_boundary", "string", label="Nodata borders", default="neumann",
              choices=_FUSW_NODATA_CODES, choice_labels=_FUSW_NODATA_LABELS,
              doc="Boundary condition of the faces between valid cells and nodata cells. "
                  "A cell with faces on the grid edge and on nodata takes its edge's "
                  "condition (when not a wall); its other faces act as walls."),
        Param("inflow_discharge", "float", label="Inflow line discharge", unit="m³/s",
              default=10.0, min=0.0,
              doc="Total discharge (m³/s) of each inflow line without a 'discharge' "
                  "attribute, times the inflow series factor."),
        Param("inflow_slope", "float", label="Inflow line energy slope", default=1e-3,
              min=0.0,
              doc="Energy slope of the uniform flow sharing an inflow line's discharge "
                  "among its faces (lines without a 'slope' attribute); > 0 with friction."),
        Param("order", "int", label="Order", default=2, choices=[1, 2],
              choice_labels=["1 (cell values, Euler)", "2 (MUSCL + Heun)"],
              doc="Order of the scheme: 1, or 2 (MUSCL reconstruction, Heun time stepping)."),
        Param("limiter", "string", default="minmod", choices=_FUSW_LIMITERS,
              choice_labels=["Minmod", "Van Albada", "Van Leer"],
              doc="Slope limiter of the MUSCL reconstruction (order 2)."),
        Param("flux", "string", label="Riemann solver", default="hll", choices=_FUSW_FLUXES,
              choice_labels=["Rusanov", "HLL", "HLL2", "HLLC", "HLLC2"],
              doc="Numerical flux of the faces."),
        Param("precision", "string", default="f32", choices=["f32", "f64"],
              choice_labels=["Single (f32)", "Double (f64)"],
              doc="Precision of the state and fluxes (parameters stay f32)."),
        Param("time_step", "string", default="cfl", choices=["cfl", "fixed"],
              choice_labels=["Automatic (CFL)", "Fixed"],
              doc="cfl: dt = min(dt_max, cfl dx / c_max) every step; fixed: dt (the run "
                  "stops with an error when it breaks the CFL condition)."),
        Param("cfl", "float", label="Courant number", default=0.5, min=0.0, max=1.0,
              doc="Courant number of the automatic time step."),
        Param("dt", "float", label="Fixed dt", unit="s", default=0.1, min=0.0,
              doc="Fixed time step (s)."),
        Param("dt_max", "float", label="Largest dt (0 = cfl dx)", unit="s", default=0.0,
              min=0.0, doc="Upper bound of the automatic time step (s); 0: cfl dx."),
        Param("max_restarts", "int", label="Restarts per step", default=2, min=0,
              doc="Order 2: times a step is restarted with a smaller dt when its second "
                  "stage needs one."),
        Param("gravity", "float", unit="m/s²", default=9.81, min=0.0, doc="Gravity (m/s²)."),
        Param("he_ca", "float", label="Dry depth", unit="m", default=1e-12, min=0.0,
              doc="Depths below this are dry (m)."),
        Param("ve_ca", "float", label="Still speed", unit="m/s", default=1e-12, min=0.0,
              doc="Velocities below this are cut to 0 (m/s)."),
        Param("ie_ca", "float", label="Smallest infiltrated depth", unit="m", default=1e-8,
              min=0.0, doc="Green-Ampt: smallest cumulative infiltrated depth (m)."),
        Param("newton_iterations", "int", default=100, min=1,
              doc="Imposed discharge: Newton iterations of the ghost depth."),
        Param("surface_slope", "string", label="Water-surface slope", default="gradient",
              choices=["gradient", "d8"], choice_labels=["Central gradient", "Steepest (D8)"],
              doc="Water-surface slope of the surface shear stress and the acceleration "
                  "ratio."),
        Param("rho", "float", label="Water density", unit="kg/m³", default=1000.0, min=0.0,
              doc="Water density of the shear stress (kg/m³)."),
        Param("check_steps", "int", label="Steps between reads", default=50, min=1,
              doc="Steps between two reads of the solver (time, max depth sample, "
                  "progress)."),
        Param("probe_every", "int", label="Probe every", unit="steps", default=10, min=1,
              doc="Probe points: one row of the probes table every this many steps."),
        Param("report_every", "float", label="Report every (0 = end)", unit="s",
              default=0.0, min=0.0,
              doc="Report row every this much model time (s); 0: at the end only."),
        Param("return_velocity", "bool", label="Also velocity", default=False,
              doc="Also return the flow speed and the east / north velocities (m/s)."),
        Param("return_unit_discharge", "bool", label="Also unit discharge", default=False,
              doc="Also return the unit discharge h |u| (m²/s)."),
        Param("return_discharge", "bool", label="Also discharge", default=False,
              doc="Also return the discharge h |u| dx (m³/s)."),
        Param("return_water_surface", "bool", label="Also water surface", default=False,
              doc="Also return the water-surface elevation z + h (m)."),
        Param("return_max_depth", "bool", label="Also max depth", default=False,
              doc="Also return the maximum depth reached (m)."),
        Param("return_infiltrated", "bool", label="Also infiltrated depth", default=False,
              doc="Also return the cumulative infiltrated depth (m; Green-Ampt)."),
        Param("return_shear_friction", "bool", label="Also shear stress (friction)",
              default=False,
              doc="Also return the bed shear stress (Pa) of the friction law."),
        Param("return_shear_surface", "bool", label="Also shear stress (surface)",
              default=False,
              doc="Also return the bed shear stress rho g h |S_w| (Pa) from the "
                  "water-surface slope."),
        Param("return_froude", "bool", label="Also Froude number", default=False,
              doc="Also return the Froude number |u| / sqrt(g h)."),
        Param("return_acceleration", "bool", label="Also acceleration ratio", default=False,
              doc="Also return the acceleration ratio from the momentum balance, "
                  "|tau_surface - tau_friction| / |tau_surface| (NaN where dry or flat)."),
        Param("return_state", "bool", default=False,
              doc="Also return the state, to continue the run."),
        Param("return_report", "bool", label="Also the report table", default=False,
              doc="Also return the report table (time, dt, max depth, water budget)."),
    ],
    outputs=[
        Output("water_depth", "georaster", doc="Water depth (m) at the end; NaN on nodata."),
        Output("speed", "georaster", optional=True, switch="return_velocity", doc="Flow speed |u| (m/s)."),
        Output("velocity_east", "georaster", optional=True, switch="return_velocity",
               doc="Velocity towards the east (m/s)."),
        Output("velocity_north", "georaster", optional=True, switch="return_velocity",
               doc="Velocity towards the north (m/s)."),
        Output("unit_discharge", "georaster", optional=True,
               doc="Unit discharge h |u| (m²/s)."),
        Output("discharge", "georaster", optional=True, doc="Discharge h |u| dx (m³/s)."),
        Output("water_surface", "georaster", optional=True,
               doc="Water-surface elevation z + h (m)."),
        Output("max_depth", "georaster", optional=True,
               doc="Maximum depth (m) over the run (and the chain it continues), sampled "
                   "every Steps between reads."),
        Output("infiltrated", "georaster", optional=True,
               doc="Cumulative infiltrated depth (m)."),
        Output("shear_stress_friction", "georaster", optional=True, switch="return_shear_friction",
               doc="Bed shear stress magnitude (Pa) of the friction law."),
        Output("shear_stress_surface", "georaster", optional=True, switch="return_shear_surface",
               doc="Bed shear stress magnitude rho g h |S_w| (Pa)."),
        Output("froude", "georaster", optional=True, doc="Froude number (0 when dry)."),
        Output("acceleration_ratio", "georaster", optional=True, switch="return_acceleration",
               doc="Acceleration ratio from the momentum balance; NaN where dry or flat."),
        Output("state", "pyfastflow.FuswFloodState", optional=True,
               doc="State to continue the run (input `previous_state`)."),
        Output("report", "datatable", optional=True,
               doc="Per row: model time, last dt, max depth, stored volume, and since the "
                   "start of the chain the rain, source, infiltrated, sink and cut volumes, "
                   "the volume out per boundary condition and the balance error (m³)."),
        Output("probe_table", "datatable", optional=True,
               doc="With probe points: per row the model time and each probe's depth and "
                   "east / north velocities."),
    ],
    impl="library",
)
def fusw_flood(dem, h_init=None, previous_state=None, precipitation_field=None,
               friction_field=None, source_field=None, sources=None, inflow_lines=None, probes=None,
               ks_field=None, kc_field=None, dtheta_field=None, psi_field=None,
               zcrust_field=None, imax_field=None, **params):
    """Transient shallow-water flood after FullSWOF 2D (pyfastflow
    ``FuswProgram``, GPU, CuPy) on ``dem`` for ``duration`` seconds of model
    time.

    Finite volumes on D4 faces: hydrostatic reconstruction, a Riemann
    solver, order 1 or 2 (MUSCL + Heun), semi-implicit friction, rain,
    sources, Green-Ampt infiltration; boundary conditions per grid edge, on
    nodata borders and on inflow lines. Nodata cells are closed and stay NaN.
    """
    from ..core import PROCESSES

    spec = PROCESSES.get("pyfastflow.fusw_flood").spec
    args = {p.name: p.default_value for p in spec.params if p.has_default}
    unknown = set(params) - set(args)
    if unknown:
        raise TypeError(f"fusw_flood: unknown params {sorted(unknown)}")
    args.update(params)
    inputs = {k: v for k, v in dict(locals()).items()
              if k in {p.name for p in spec.inputs}}
    return run_once(FuswFloodRun, inputs, args)


@runner("pyfastflow.fusw_flood", axis="time", config=(
    "rain_series", "inflow_series", "source_series", "series_interp", "friction",
    "infiltration", "boundary", *(f"{s}_boundary" for s in _FUSW_EDGES), "nodata_boundary",
    "order", "limiter", "flux", "precision", "time_step", "max_restarts", "surface_slope"))
class FuswFloodRun(Runner):
    """Live FullSWOF flood: one program across advances. Each advance
    simulates ``duration`` more seconds; model time, budget, max depth and
    the report and probe rows continue. Every non-config param (rain rate,
    friction, infiltration, edge values, line discharges, time stepping,
    thresholds) can change between advances."""

    who = "FullSWOF flood"

    def __init__(self, inputs, p):
        import cupy as cp

        who = self.who
        for name, choices in (("friction", _FUSW_FRICTIONS), ("flux", _FUSW_FLUXES),
                              ("limiter", _FUSW_LIMITERS), ("nodata_boundary",
                                                            _FUSW_NODATA_CODES)):
            if p[name] not in choices:
                raise ValueError(f"{who}: unknown {name} {p[name]!r}")
        self.edge_codes = {}
        for side in _FUSW_EDGES:
            code = p[f"{side}_boundary"]
            if code not in _FUSW_EDGE_CODES:
                raise ValueError(f"{who}: unknown {side}_boundary {code!r}")
            self.edge_codes[side] = code
        self.nodata_code = p["nodata_boundary"]
        self.dem = dem = inputs["dem"]
        self.rain_field = inputs.get("precipitation_field")
        self.dx, valid, _ = _flood_grid(dem, who)
        self.valid = valid
        shape = self.shape = valid.shape
        self.real = np.float64 if p["precision"] == "f64" else np.float32
        z = np.where(valid, dem.z, 0.0).astype(self.real)
        exposed = _fusw_exposed(valid, p["boundary"])
        codes, faces, self.edge_of = _fusw_codes(exposed, self.edge_codes, p["nodata_boundary"])
        self.lines = inputs.get("inflow_lines")
        line_faces = (_fusw_line_faces(self.lines, dem, exposed, who)
                      if self.lines is not None else [])
        probe_points = inputs.get("probes")
        self.probe_cells = None
        if probe_points is not None:
            self.probe_cells = self._cells_of(probe_points)
        series = {name: _fusw_series(p[f"{name}_series"], name, who) for name in _FUSW_SERIES}

        # Spatial params are fields only when an array goes to the constructor.
        fields = {}
        for name, (port, mm_h) in {"rain": ("precipitation_field", True),
                                   "friction_coefficient": ("friction_field", False),
                                   "source": ("source_field", False)}.items():
            if inputs.get(port) is not None:
                value = _field(inputs[port], 0.0, shape, port, who)
                fields[name] = np.where(valid, value * (_MM_PER_H if mm_h else 1.0), 0.0)
        for name, (_d, _u, rate) in _FUSW_GREEN_AMPT.items():
            if inputs.get(f"{name}_field") is not None:
                value = _field(inputs[f"{name}_field"], 0.0, shape, f"{name}_field", who)
                fields[name] = value * (_MM_PER_H if rate else 1.0)
        # Discharge sources go to the source field (m³/s per cell); a linked
        # one follows the rain rate, sent again in _set_params.
        # With linked ones the source series is applied here (the program's
        # source factor would scale them too): the field is sent again before
        # each run of steps, linked q x the rain factor, the rest x the source
        # factor.
        self.stamps = _source_stamps(inputs.get("sources"), dem, valid, who)
        self.source_base = fields.get("source")
        self.series, self.series_interp = series, p["series_interp"]
        self.linked = any(area > 0.0 for _c, _w, _q, area in self.stamps)
        if self.stamps:
            fields["source"] = self._source(p, 0.0)
        self.edge_params = sorted({prog for prog, used in _FUSW_EDGE_VALUES.values()
                                   if any(c in used for c in self.edge_codes.values())})
        edge_values = self._edge_values(p)
        for name in self.edge_params:
            fields[name] = edge_values[name]
        self.field_names = set(fields)
        flood = self.flood = FuswProgram(
            _backend(), nx=shape[1], ny=shape[0], dx=self.dx, boundary=p["boundary"],
            nodata=not valid.all(), precision=p["precision"], order=int(p["order"]),
            limiter=p["limiter"], flux=p["flux"], friction=p["friction"],
            max_restarts=int(p["max_restarts"]),
            infiltration=p["infiltration"], time_step=p["time_step"],
            inflow_unit="unit", series_interp=p["series_interp"],
            surface_slope=p["surface_slope"], inflow_lines=len(line_faces),
            inflow_line_faces=sum(len(r) for r, _c, _s in line_faces),
            n_probes=0 if self.probe_cells is None else len(self.probe_cells[0]),
            record_capacity=0 if self.probe_cells is None else _FUSW_RECORD_ROWS,
            **{f"{name}_series": 0 if name == "source" and self.linked else len(series[name][0])
               for name in _FUSW_SERIES},
            **{k: v.astype(np.float32) for k, v in fields.items()})
        try:
            if not valid.all():
                flood.nodata_mask.from_numpy((~valid).astype(np.uint8))
            flood.z.from_numpy(z)
            flood.initialize()
            flood.boundary_type.from_numpy(codes)
            flood.boundary_faces.from_numpy(faces)
            for k, (rows, cols, sides) in enumerate(line_faces):
                flood.set_inflow_line(k, rows, cols, sides, *self._line_values(k, p))
            for name in _FUSW_SERIES:
                if len(series[name][0]) and not (name == "source" and self.linked):
                    flood.set_series(name, *series[name])
            if self.probe_cells is not None:
                flood.set_probes(*self.probe_cells)
            self.max_depth = cp.asarray(self._start(inputs.get("h_init"),
                                                    inputs.get("previous_state")))
        except Exception:
            flood.close()
            raise
        self.h = flood.h.array
        self.rows, self.probe_rows = {}, {}
        self.next_row = None
        # dt of the last full step read (not the shorter one landing on a stop).
        self.step_dt = 0.0

    def _source(self, p, t):
        """The source field (m³/s per cell) at model time ``t``: the field
        input plus the sources. With linked sources, (field + fixed q) x the
        source factor + linked q (rain x area) x the rain factor; without, the
        program applies the source factor."""
        rain = (_field(self.rain_field, 0.0, self.shape, "precipitation_field", self.who) * _MM_PER_H
                if self.rain_field is not None else float(p["rainfall"]) * _MM_PER_H)
        if not self.linked:
            out = _source_discharge(self.stamps, rain, self.shape)
            if self.source_base is not None:
                out = out + self.source_base
            return np.where(self.valid, out, 0.0)
        fixed = _source_discharge([(c, w, q, 0.0) for c, w, q, _a in self.stamps], 0.0, self.shape)
        linked = _source_discharge([(c, w, 0.0, a) for c, w, _q, a in self.stamps], rain, self.shape)
        if self.source_base is not None:
            fixed = fixed + self.source_base
        out = fixed * self._factor("source", t) + linked * self._factor("rain", t)
        return np.where(self.valid, out, 0.0)

    def _factor(self, name, t):
        """The factor of series ``name`` at model time ``t`` (1 without one;
        the end factors hold outside its times)."""
        times, factors = self.series[name]
        if not len(times):
            return 1.0
        if self.series_interp == "linear":
            return float(np.interp(t, times, factors))
        return float(factors[max(int(np.searchsorted(times, t, side="right")) - 1, 0)])

    def _cells_of(self, points):
        dem = self.dem
        xy = points.in_crs(dem.crs_wkt or dem.epsg).xy
        if len(xy) == 0:
            raise ValueError(f"{self.who}: no probe point given")
        cols = np.floor((xy[:, 0] - dem.x_min) / dem.cell_size).astype(np.int64)
        rows = np.floor((dem.y_max - xy[:, 1]) / dem.cell_size).astype(np.int64)
        off = (rows < 0) | (rows >= self.shape[0]) | (cols < 0) | (cols >= self.shape[1])
        if off.any():
            raise ValueError(f"{self.who}: {int(off.sum())} probe point(s) fall outside the grid")
        return rows, cols

    def _edge_values(self, p):
        """Per program edge param: the (ny, nx) values of the cells taking
        each edge's condition (0 elsewhere). An edge discharge (m³/s) is
        shared by its faces as a unit discharge (m²/s)."""
        out = {}
        for suffix, (prog, _used) in _FUSW_EDGE_VALUES.items():
            if prog not in self.edge_params:
                continue
            table = np.zeros(len(_FUSW_EDGES) + 1, dtype=np.float64)
            for k, side in enumerate(_FUSW_EDGES):
                value = float(p[f"{side}_{suffix}"])
                if suffix == "discharge":
                    n = int((self.edge_of == k).sum())
                    value = value / (n * self.dx) if n else 0.0
                table[k] = value
            out[prog] = table[self.edge_of].astype(np.float32)  # index -1: the 0 at the end
        return out

    def _line_values(self, k, p):
        return (_fusw_feature_value(self.lines, k, "discharge", p["inflow_discharge"]),
                _fusw_feature_value(self.lines, k, "slope", p["inflow_slope"]))

    def _start(self, h_init, state):
        """Writes the start state; returns the running max depth (host)."""
        flood, valid, shape, who = self.flood, self.valid, self.shape, self.who
        if state is not None:
            if state.h.shape != shape:
                raise ValueError(f"{who}: the state is {state.h.shape[0]} x {state.h.shape[1]} "
                                 f"cells, the DEM {shape[0]} x {shape[1]}")
            if abs(state.dx - self.dx) > 1e-6 * self.dx:
                raise ValueError(f"{who}: the state's cell size is {state.dx} m, the DEM's "
                                 f"{self.dx} m")
            for name in ("h", "u", "v"):
                getattr(flood, name).from_numpy(
                    np.where(valid, getattr(state, name), 0.0).astype(self.real))
            if flood.infiltration != "none" and state.infiltrated is not None:
                flood.infiltrated.from_numpy(state.infiltrated.astype(self.real))
            clock = flood._handle("clock")
            values = clock.to_numpy()
            values[_FUSW_TIME] = float(state.time)
            clock.from_numpy(values)
            flood._handle("budget").from_numpy(np.asarray(state.budget, dtype=np.float64))
            return state.max_depth.astype(np.float32)
        if h_init is not None:
            if h_init.z.shape != shape:
                raise ValueError(f"{who}: h_init is {h_init.z.shape[0]} x {h_init.z.shape[1]} "
                                 f"cells, the DEM {shape[0]} x {shape[1]}")
            h = np.where(valid, np.maximum(np.nan_to_num(h_init.z, nan=0.0), 0.0), 0.0)
            flood.h.from_numpy(h.astype(self.real))
        return flood.h.to_numpy().astype(np.float32)

    def _set_params(self, p):
        flood = self.flood
        scalars = {"gravity": p["gravity"], "cfl": p["cfl"], "dt": p["dt"],
                   "dt_max": p["dt_max"], "he_ca": p["he_ca"], "ve_ca": p["ve_ca"],
                   "ie_ca": p["ie_ca"], "rho": p["rho"]}
        if "rain" not in self.field_names:
            scalars["rain"] = float(p["rainfall"]) * _MM_PER_H
        else:  # the field again: it may have changed since (see FloodRun)
            rain = _field(self.rain_field, 0.0, self.shape, "precipitation_field", self.who)
            flood.rain.from_numpy(np.where(self.valid, rain * _MM_PER_H, 0.0).astype(self.real))
        if "source" in self.field_names:
            flood.source.from_numpy(self._source(p, flood.time).astype(self.real))
        if "friction_coefficient" not in self.field_names:
            scalars["friction_coefficient"] = p["friction_coefficient"]
        for name, (_d, _u, rate) in _FUSW_GREEN_AMPT.items():
            if name not in self.field_names:
                scalars[name] = float(p[name]) * (_MM_PER_H if rate else 1.0)
        _flood_set(flood, **{k: float(v) for k, v in scalars.items()})
        _flood_set(flood, newton_iterations=int(p["newton_iterations"]),
                   record_interval=int(p["probe_every"]))
        for name, values in self._edge_values(p).items():
            getattr(flood, name).from_numpy(values)
        for k in range(int(flood.inflow_lines)):
            flood.set_line_discharge(k, *self._line_values(k, p))

    def advance(self, p):
        import cupy as cp

        who, flood = self.who, self.flood
        duration = float(p["duration"])
        if duration <= 0.0:
            raise ValueError(f"{who}: duration must be > 0")
        if p["time_step"] == "fixed" and float(p["dt"]) <= 0.0:
            raise ValueError(f"{who}: the fixed time step must be > 0")
        if p["friction"] != "none" and self.lines is not None and not any(
                _fusw_feature_value(self.lines, k, "slope", p["inflow_slope"]) > 0.0
                for k in range(self.lines.n_features)):
            raise ValueError(f"{who}: the inflow lines need an energy slope > 0")
        self._set_params(p)
        chunk = int(p["check_steps"])
        if self.probe_cells is not None:
            chunk = min(chunk, _FUSW_RECORD_ROWS * int(p["probe_every"]))
        t0 = flood.time
        end = t0 + duration
        report_dt = float(p["report_every"])
        if self.next_row is None or self.next_row <= t0:
            self.next_row = t0 + report_dt if report_dt > 0.0 else end
        self.next_row = min(self.next_row, end)
        eps = 1e-9 * max(duration, 1.0)
        clock = flood._handle("clock")
        bound = (float(p["dt"]) if p["time_step"] == "fixed" else
                 float(p["dt_max"]) if float(p["dt_max"]) > 0.0 else float(p["cfl"]) * self.dx)
        progress.report(0.0, duration, "shallow-water flow")
        try:
            while flood.time < end - eps:
                stop = self.next_row
                values = clock.to_numpy()
                values[_FUSW_T_END] = stop
                clock.from_numpy(values)
                while flood.time < stop - eps:
                    before = flood.time
                    # Steps to the stop at the last dt (or its bound): no
                    # zero-length steps after landing on it.
                    n = max(1, min(chunk, int(np.ceil((stop - before) / (self.step_dt or bound)))))
                    if self.linked:
                        flood.source.from_numpy(self._source(p, before).astype(self.real))
                    flood.run(n)
                    if flood.time < stop - eps:
                        self.step_dt = flood.last_dt
                    elif not self.step_dt:
                        self.step_dt = (flood.time - before) / n
                    cp.maximum(self.max_depth, self.h.reshape(self.shape).astype(cp.float32),
                               out=self.max_depth)
                    self._drain_probes()
                    progress.report(min(flood.time, end) - t0, duration, "shallow-water flow")
                    if flood.time <= before:
                        raise RuntimeError(f"{who}: the time step fell to 0 at t = "
                                           f"{flood.time:.6g} s")
                self._row()
                self.next_row = min(self.next_row + report_dt, end) if report_dt > 0.0 else end
        finally:
            values = clock.to_numpy()
            values[_FUSW_T_END] = 0.0
            clock.from_numpy(values)
        return self._outputs(p)

    def _drain_probes(self):
        if self.probe_cells is None:
            return
        rec = self.flood.records()
        if rec["overflow"]:
            raise RuntimeError(f"{self.who}: probe rows were dropped (record capacity)")
        if not len(rec["time"]):
            return
        rows = self.probe_rows
        rows.setdefault("time", []).extend(rec["time"].tolist())
        for k in range(rec["h"].shape[1]):
            rows.setdefault(f"h_{k + 1}", []).extend(rec["h"][:, k].tolist())
            rows.setdefault(f"u_east_{k + 1}", []).extend(rec["u"][:, k].tolist())
            rows.setdefault(f"v_north_{k + 1}", []).extend((-rec["v"][:, k]).tolist())

    def _row(self):
        flood = self.flood
        b = flood.budget()
        row = {"time": flood.time, "dt": self.step_dt, "h_max": float(self.h.max()),
               "stored": b["stored"], "rain": b["rain"], "source": b["source"],
               "infiltration": b["infiltration"], "sink": b["sink"],
               "cut": b["threshold"], "outflow": b["outflow"]}
        used = set(self.edge_codes.values()) | {self.nodata_code}
        if self.lines is not None:
            used.add("inflow")
        for code in FUSW_CODES:
            if code in used and code != "wall":
                row[f"out_{code}"] = b["boundary"][code]["volume"]
        row["error"] = b["error"]
        for k, v in row.items():
            self.rows.setdefault(k, []).append(float(v))

    def _outputs(self, p):
        import cupy as cp

        flood, valid, dem = self.flood, self.valid, self.dem
        nan = lambda a: _like(dem, np.where(valid, a, np.nan))
        h = flood.field("h")
        out = {"water_depth": nan(h)}
        if p["return_velocity"]:
            u, v = flood.field("u"), flood.field("v")
            out["speed"] = nan(np.hypot(u, v))
            out["velocity_east"] = nan(u)
            out["velocity_north"] = nan(-v)
        if p["return_unit_discharge"]:
            out["unit_discharge"] = nan(flood.field("q"))
        if p["return_discharge"]:
            out["discharge"] = nan(flood.field("Q"))
        if p["return_water_surface"]:
            out["water_surface"] = nan(flood.field("eta"))
        max_depth = cp.asnumpy(self.max_depth)
        if p["return_max_depth"]:
            out["max_depth"] = nan(max_depth)
        if p["return_infiltrated"]:
            if flood.infiltration == "none":
                raise ValueError(f"{self.who}: the infiltrated depth needs infiltration on")
            out["infiltrated"] = nan(flood.field("infiltrated"))
        if p["return_shear_friction"]:
            out["shear_stress_friction"] = nan(np.hypot(*flood.shear_stress("friction")))
        if p["return_shear_surface"]:
            out["shear_stress_surface"] = nan(np.hypot(*flood.shear_stress("surface")))
        if p["return_froude"]:
            out["froude"] = nan(flood.froude())
        if p["return_acceleration"]:
            out["acceleration_ratio"] = nan(flood.acceleration_ratio_balance())
        if p["return_state"]:
            infiltrated = flood.field("infiltrated") if flood.infiltration != "none" else None
            out["state"] = FuswFloodState(
                h=h, u=flood.field("u"), v=flood.field("v"), infiltrated=infiltrated,
                time=flood.time, max_depth=max_depth,
                budget=flood._handle("budget").to_numpy().copy(), dx=self.dx)
        if p["return_report"]:
            units = {k: "m³" for k in self.rows}
            units.update(time="s", dt="s", h_max="m")
            out["report"] = DataTable(
                {k: np.asarray(v, dtype=np.float64) for k, v in self.rows.items()},
                units=units, roles={"time": "x", "dt": "aux"}, title="FullSWOF flood")
        if self.probe_cells is not None and self.probe_rows:
            units = {k: ("m" if k.startswith("h_") else "m/s") for k in self.probe_rows}
            units["time"] = "s"
            out["probe_table"] = DataTable(
                {k: np.asarray(v, dtype=np.float64) for k, v in self.probe_rows.items()},
                units=units, roles={"time": "x"}, title="FullSWOF probes")
        return out

    def coordinate(self):
        return {"time": float(self.flood.time)}

    def status(self):
        rows = self.rows
        if not rows.get("time"):
            return {"text": "", "stopped": False}
        return {"text": f"t = {_num(rows['time'][-1], 's')}, max depth "
                        f"{_num(rows['h_max'][-1], 'm')}, out {_num(rows['outflow'][-1], 'm³')}, "
                        f"balance error {_num(rows['error'][-1], 'm³')}.",
                "stopped": False}

    def close(self):
        self.flood.close()


if GraphFloodRelax is None:
    from ..core.process import DEFAULT_PROCESSES
    for _pid in ("pyfastflow.graphflood", "pyfastflow.graphflood_particle"):
        DEFAULT_PROCESSES._procs.pop(_pid, None)
if InertialFloodProgram is None:
    from ..core.process import DEFAULT_PROCESSES
    DEFAULT_PROCESSES._procs.pop("pyfastflow.inertial_flood", None)
if FuswProgram is None:
    from ..core.process import DEFAULT_PROCESSES
    DEFAULT_PROCESSES._procs.pop("pyfastflow.fusw_flood", None)
