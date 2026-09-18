"""DEM preparation routines (composite processes).

Author: B.G.
"""

from __future__ import annotations

from ..core import Output, Param, Process, process
from ..core.registries import PROCESSES


@process(
    id="pytopoviz.load_and_smooth",
    label="Load sample DEM and smooth",
    params=[
        Param("name", "string", default="bigtujunga"),
        Param("sigma", "float", default=2.0, min=0.0),
    ],
    outputs=[Output("smoothed", "topotoolbox.GridObject")],
    impl="composite",
)
def load_and_smooth(name="bigtujunga", sigma=2.0):
    """Compose two registered processes: load a sample DEM, then Gaussian-smooth it."""
    load: Process = PROCESSES.get("topotoolbox.load_dem")
    smooth: Process = PROCESSES.get("topotoolbox.gaussian_smooth")

    grid = load(name=name)["grid"]
    return smooth(dem=grid, sigma=sigma)["smoothed"]
