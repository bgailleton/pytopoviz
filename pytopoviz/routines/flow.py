"""Flow routines (composite processes).

Author: B.G.
"""

from __future__ import annotations

from ..core import Output, Param, Port, process
from ..core.registries import PROCESSES


@process(
    id="pytopoviz.flow_accumulation_lsdtt3",
    label="Flow accumulation",
    inputs=[
        Port("dem", "lsdtt3.Raster", doc="Elevation to route water over."),
        Port("filled_dem", "lsdtt3.Raster", optional=True,
             doc="Your own depression-free DEM; skips the filling step."),
        Port("flow_directions", "lsdtt3.FlowInfo", optional=True,
             doc="Flow directions already computed; skips filling and routing."),
        Port("weights", "lsdtt3.Raster", optional=True,
             doc="Amount each cell contributes (e.g. rainfall); replaces the units choice."),
    ],
    params=[
        Param("units", "enum", default="area", choices=["area", "pixels"],
              doc="Upstream area in map units squared, or number of upstream cells."),
        Param("boundary_conditions", "lsdtt3.boundary_conditions", default="oooo"),
        Param("min_slope", "float", default=1e-4, min=0.0,
              doc="Small slope given to filled areas so water keeps flowing."),
        Param("allow_pits_as_outlets", "bool", default=False,
              doc="Let leftover pits end the flow instead of failing."),
    ],
    outputs=[
        Output("accumulation", "lsdtt3.Raster", doc="Accumulated flow."),
        Output("filled", "lsdtt3.Raster", optional=True,
               doc="The depression-free DEM used (not produced when flow directions are given)."),
        Output("flow", "lsdtt3.FlowInfo", doc="Flow directions, reusable by later runs."),
    ],
    impl="composite",
)
def flow_accumulation_lsdtt3(
    dem, filled_dem=None, flow_directions=None, weights=None, units="area",
    boundary_conditions="oooo", min_slope=1e-4, allow_pits_as_outlets=False,
):
    """How much water flows through every cell of a DEM.

    Chains lsdtt3.fill_pits_boundary_aware -> lsdtt3.flow_info ->
    lsdtt3.drainage_area | contributing_pixels | accumulate. Each stage can be
    supplied instead of computed: ``filled_dem`` skips the fill,
    ``flow_directions`` skips fill and routing (``dem`` is then unused).
    """
    flow = flow_directions
    if flow is None:
        if filled_dem is None:
            filled_dem = PROCESSES.get("lsdtt3.fill_pits_boundary_aware")(
                raster=dem, boundary_conditions=boundary_conditions, min_slope=min_slope
            )["filled"]
        flow = PROCESSES.get("lsdtt3.flow_info")(
            dem=filled_dem,
            boundary_conditions=boundary_conditions,
            allow_pits_as_outlets=allow_pits_as_outlets,
        )["flow"]

    if weights is not None:
        acc = PROCESSES.get("lsdtt3.accumulate")(flow=flow, weights=weights)["accumulation"]
    elif units == "area":
        acc = PROCESSES.get("lsdtt3.drainage_area")(flow=flow)["area"]
    else:
        acc = PROCESSES.get("lsdtt3.contributing_pixels")(flow=flow)["pixels"]
    return {"accumulation": acc, "filled": filled_dem, "flow": flow}
