"""lsdtt3 adapter: raster IO, D8 flow topology, and converters.

Registration unit for lsdtt3 (DESIGN.md §8). Hard-imports lsdtt3; the guard for a
missing library lives in ``adapters/__init__``. The GridObject->Raster bridge is
registered only when the topotoolbox adapter is also present, so this module stays
usable on its own.

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import lsdtt3

from ..core import Output, Param, Port, process, register_converter, register_type
from ..core.registries import TYPES

# ---- types ------------------------------------------------------------------

register_type("lsdtt3.Raster", "grid", lsdtt3.Raster)
register_type("lsdtt3.FlowInfo", "graph", lsdtt3.FlowInfo)


# ---- converters -------------------------------------------------------------
# Raster -> raw 2D float field. (field->Raster needs a RasterMeta; see the
# GridObject bridge below for the metadata-preserving direction.)

register_converter("lsdtt3.Raster", "field2d", lambda r: np.asarray(r.to_numpy()))


def _grid_to_raster(grid):
    """topotoolbox.GridObject -> lsdtt3.Raster, carrying origin and cell size."""
    z = np.asarray(grid.z, dtype=float)
    n_rows, n_cols = z.shape
    b = grid.bounds
    meta = lsdtt3.RasterMeta.from_shape(
        n_rows, n_cols, x_min=b.left, y_min=b.bottom, cell_size=float(grid.cellsize)
    )
    return lsdtt3.Raster.from_numpy(z, np.isfinite(z), meta)


# Cross-library bridge: only if topotoolbox surfaced its type.
if TYPES.has("topotoolbox.GridObject"):
    register_converter("topotoolbox.GridObject", "lsdtt3.Raster", _grid_to_raster)


# ---- processes --------------------------------------------------------------

@process(
    id="lsdtt3.read_raster",
    label="Read raster (lsdtt3)",
    params=[Param("path", "path")],
    outputs=[Output("raster", "lsdtt3.Raster")],
    impl="library",
)
def read_raster(path):
    return lsdtt3.Raster.read(path)


@process(
    id="lsdtt3.flow_info",
    label="D8 flow topology",
    inputs=[Port("dem", "lsdtt3.Raster")],
    params=[
        Param("boundary_conditions", "string", default="oooo"),
        Param("allow_pits_as_outlets", "bool", default=True),
    ],
    outputs=[Output("flow", "lsdtt3.FlowInfo")],
    impl="library",
)
def flow_info(dem, boundary_conditions="oooo", allow_pits_as_outlets=True):
    return lsdtt3.FlowInfo(
        dem,
        boundary_conditions,
        True,  # interior_nodata_is_outlet
        None,  # forced_outlets
        allow_pits_as_outlets,
    )


@process(
    id="lsdtt3.drainage_area",
    label="Drainage area",
    inputs=[Port("flow", "lsdtt3.FlowInfo")],
    outputs=[Output("area", "lsdtt3.Raster")],
    impl="library",
)
def drainage_area(flow):
    return flow.drainage_area()
