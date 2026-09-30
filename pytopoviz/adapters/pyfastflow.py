"""pyfastflow adapter.

Registration unit for pyfastflow (DESIGN.md §8). pyfastflow's programs run on a
CuPy (GPU) backend, created once per python session on first use. Each process
builds its programs, runs them and closes them within the call, so no device
state outlives a call; results cross back as numpy-backed types.

- ``pyfastflow.saleve_steady``: Salève steady-state landscape generator, a port
  of ``compute_landscape`` in pyfastflow's ``examples/saleve_app.py`` (Perlin
  uplift / erodibility fields, white-noise initial surface, one steady
  multigrid solve).

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import pyfastflow  # noqa: F401  (hard import: the adapter is skipped without it)
from pyfastflow.core import Backend
from pyfastflow.flow import BOUNDARIES, LOCAL_MINIMA, RECEIVER_MODES, TOPOLOGIES
from pyfastflow.noise import PerlinNoiseProgram
from pyfastflow.saleve import HILLSLOPE_MODELS, SLOPE_CORRECTIONS, VALLEY_MODELS, SaleveProgram

from ..core import Output, Param, process
from ..georaster import GeoRaster

_BACKEND = None


def _backend():
    """The session's CuPy backend, created on first use (fails there, with
    CuPy's own message, when no GPU is usable)."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = Backend.from_name("cupy")
    return _BACKEND


def _perlin_field(n, base, span, freq, octaves, persistence, seed):
    """``base * 2 ** (span * noise)`` on an (n, n) grid, noise normalised to
    [-1, 1]; the scalar ``base`` when ``span`` is 0."""
    import cupy as cp

    if span == 0.0:
        return float(base)
    # The noise program works on half-integer frequencies.
    freq = round(2.0 * freq) / 2.0
    with PerlinNoiseProgram(_backend(), nx=n, ny=n, amplitude=1.0,
                            frequency=freq, octaves=int(octaves),
                            persistence=persistence, seed=int(seed)) as noise:
        noise.generate()
        v = noise.noise.array
        v = v / cp.maximum(cp.abs(v).max(), 1e-12)
        return cp.asnumpy(base * cp.exp2(span * v)).astype(np.float32).reshape(n, n)


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
        Param("erodibility", "float", default=2e-5, min=0.0,
              doc="Erodibility K, mean of the erodibility field."),
        # Uplift field (Perlin)
        Param("u_span", "float", default=1.0, min=0.0,
              doc="Uplift field contrast: U varies by 2^(± span); 0 = uniform."),
        Param("u_freq", "float", default=3.0, min=0.5,
              doc="Uplift field frequency (features per domain)."),
        Param("u_octaves", "int", default=4, min=1,
              doc="Uplift field octaves."),
        Param("u_persistence", "float", default=0.5, min=0.0, max=1.0,
              doc="Uplift field persistence (amplitude kept per octave)."),
        # Erodibility field (Perlin)
        Param("k_span", "float", default=1.0, min=0.0,
              doc="Erodibility field contrast: K varies by 2^(± span); 0 = uniform."),
        Param("k_freq", "float", default=5.0, min=0.5,
              doc="Erodibility field frequency (features per domain)."),
        Param("k_octaves", "int", default=4, min=1,
              doc="Erodibility field octaves."),
        Param("k_persistence", "float", default=0.5, min=0.0, max=1.0,
              doc="Erodibility field persistence (amplitude kept per octave)."),
        # Hillslope
        Param("hillslope_model", "string", default="hack", choices=HILLSLOPE_MODELS,
              doc="How hillslope erosion enters the link slope: hack (area proxy), "
                  "divide_linear or divide_roering (distance from the divide, the "
                  "latter saturating at the critical slope)."),
        Param("hillslope_erosion", "float", default=0.4, min=0.0,
              doc="Hillslope diffusivity D; 0 = off."),
        Param("hack_constant", "float", default=1.5, min=0.0,
              doc="Hack's law constant c (hack model)."),
        Param("hack_exponent", "float", default=0.6, min=0.0,
              doc="Hack's law exponent h (hack model)."),
        Param("channel_threshold", "bool", default=False,
              doc="Switch between hillslope and fluvial laws at Channel area instead "
                  "of adding them."),
        Param("channel_area", "float", default=2.5e4, min=0.0,
              doc="Drainage area (m²) of the hillslope / channel switch; used when "
                  "Channel threshold is on."),
        Param("diffusion_iterations", "int", default=0, min=0,
              doc="Jacobi sweeps of the 2D diffusion / stream-power balance after "
                  "each solve; 0 = off."),
        # Thermal (talus)
        Param("thermal_erosion", "float", default=0.0, min=0.0,
              doc="Thermal (talus) coefficient kt; 0 = off."),
        Param("critical_slope", "float", default=0.5, min=0.0,
              doc="Critical slope Sc of thermal erosion (and of the Roering model)."),
        # Valleys
        Param("valley_model", "string", default="none", choices=VALLEY_MODELS,
              doc="hand: lower cells close above their channel onto a valley floor."),
        Param("valley_area", "float", default=1e6, min=0.0,
              doc="Drainage area (m²) from which a channel gets a valley."),
        Param("valley_height", "float", default=10.0, min=0.0,
              doc="Floodplain height (m) above the channel at Valley area."),
        Param("valley_exponent", "float", default=0.5, min=0.0,
              doc="Exponent of the floodplain height's growth with drainage area."),
        Param("valley_slope", "float", default=1e-3, min=0.0,
              doc="Transverse slope of the valley floor."),
        Param("valley_transition", "float", default=0.5, min=0.0,
              doc="Width over which the valley wall is smoothed."),
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
    ],
    outputs=[Output("grid", "georaster")],
    impl="library",
)
def saleve_steady(n=1024, dx=50.0, seed=1, m=0.45, uplift=1e-3, erodibility=2e-5,
                  u_span=1.0, u_freq=3.0, u_octaves=4, u_persistence=0.5,
                  k_span=1.0, k_freq=5.0, k_octaves=4, k_persistence=0.5,
                  hillslope_model="hack", hillslope_erosion=0.4,
                  hack_constant=1.5, hack_exponent=0.6,
                  channel_threshold=False, channel_area=2.5e4, diffusion_iterations=0,
                  thermal_erosion=0.0, critical_slope=0.5,
                  valley_model="none", valley_area=1e6, valley_height=10.0,
                  valley_exponent=0.5, valley_slope=1e-3, valley_transition=0.5,
                  topology="D8", boundary="periodic_EW", local_minima="cordonnier_carve",
                  epsilon=1e-3, jitter=10.0, receiver_mode="steepest", receiver_seed=0,
                  slope_correction="gradient", min_link_slope=1e-6,
                  max_slope_correction=100.0,
                  cliff_optimization=False, cliff_iterations=50,
                  cliff_learning_rate=0.01, cliff_river_weight=1.0 / 3.0,
                  levels=5, iterations=24, relaxation=0.25):
    """Generate a steady-state landscape with pyfastflow's Salève stream-power solver.

    Perlin uplift and erodibility fields over a white-noise initial surface, one
    steady multigrid solve on the GPU (CuPy). Square grid of n x n cells of dx
    metres, origin (0, 0), no CRS.
    """
    n = int(n)
    seed = int(seed)
    backend = _backend()
    uplift_field = _perlin_field(n, uplift, u_span, u_freq, u_octaves, u_persistence,
                                 seed + 1000)
    erodibility_field = _perlin_field(n, erodibility, k_span, k_freq, k_octaves,
                                      k_persistence, seed + 2000)
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
        hillslope_model=hillslope_model,
        channel_area=channel_area if channel_threshold else 0.0,
        diffusion_iterations=int(diffusion_iterations),
        valley_model=valley_model, valley_area=valley_area,
        valley_height=valley_height, valley_exponent=valley_exponent,
        valley_slope=valley_slope, valley_transition=valley_transition,
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
    return GeoRaster(z=z, cell_size=float(dx), x_min=0.0, y_min=0.0)
