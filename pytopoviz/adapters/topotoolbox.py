"""topotoolbox adapter: surface types, converters and processes.

Registration unit for topotoolbox (DESIGN.md §8). Hard-imports topotoolbox; the
guard for a missing library is in ``adapters/__init__``.

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import topotoolbox as ttb
from scipy.ndimage import gaussian_filter

from ..core import Output, Param, Port, process, register_converter, register_type

# ---- types ------------------------------------------------------------------

register_type("topotoolbox.GridObject", "grid", ttb.GridObject)


# ---- converters -------------------------------------------------------------
# GridObject -> raw 2D float field (conceptual `field2d`, registered in
# adapters/_conceptual), so array-consuming processes/frontends can accept a grid.

register_converter(
    "topotoolbox.GridObject", "field2d", lambda g: np.asarray(g.z)
)


# ---- processes --------------------------------------------------------------

@process(
    id="topotoolbox.read_tif",
    label="Read GeoTIFF",
    params=[Param("path", "path")],
    outputs=[Output("grid", "topotoolbox.GridObject")],
    impl="library",
)
def read_tif(path):
    return ttb.read_tif(path)


@process(
    id="topotoolbox.load_dem",
    label="Load sample DEM",
    params=[Param("name", "string", default="bigtujunga")],
    outputs=[Output("grid", "topotoolbox.GridObject")],
    impl="library",
)
def load_dem(name="bigtujunga"):
    return ttb.load_dem(name)


@process(
    id="topotoolbox.gaussian_smooth",
    label="Gaussian smooth",
    inputs=[Port("dem", "topotoolbox.GridObject")],
    params=[
        Param("sigma", "float", default=2.0, min=0.0),
        Param("mode", "string", default="nearest"),
    ],
    outputs=[Output("smoothed", "topotoolbox.GridObject")],
    impl="library",
)
def gaussian_smooth(dem, sigma=2.0, mode="nearest"):
    """NaN-aware 2D Gaussian smoothing of a GridObject's data."""
    data = np.asarray(dem.z, dtype=float)
    finite = np.isfinite(data)
    filled = np.where(finite, data, 0.0)
    smoothed = gaussian_filter(filled, sigma=sigma, mode=mode)
    weights = gaussian_filter(finite.astype(float), sigma=sigma, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = smoothed / weights
    result[weights == 0.0] = np.nan
    return dem.duplicate_with_new_data(result.astype(np.float32))


@process(
    id="topotoolbox.fillsinks",
    label="Fill sinks",
    inputs=[Port("dem", "topotoolbox.GridObject")],
    params=[Param("hybrid", "bool", default=True)],
    outputs=[Output("filled", "topotoolbox.GridObject")],
    impl="library",
)
def fillsinks(dem, hybrid=True):
    return dem.fillsinks(hybrid=hybrid)


@process(
    id="topotoolbox.filter",
    label="Filter DEM",
    inputs=[Port("dem", "topotoolbox.GridObject")],
    params=[
        Param(
            "method",
            "enum",
            default="mean",
            choices=["mean", "average", "median", "sobel", "scharr", "wiener", "std"],
        ),
        Param("kernelsize", "int", default=3, min=1),
    ],
    outputs=[Output("filtered", "topotoolbox.GridObject")],
    impl="library",
)
def filter_dem(dem, method="mean", kernelsize=3):
    return dem.filter(method=method, kernelsize=kernelsize)
