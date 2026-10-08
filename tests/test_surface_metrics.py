"""Surface metrics tests: lsdtt3.polyfit_metrics on analytic surfaces (skipped
without lsdtt3).

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lsdtt3")

import pytopoviz.adapters  # noqa: F401,E402  (registers library adapters)
from pytopoviz.core import CONVERTERS, PROCESSES  # noqa: E402
from pytopoviz.georaster import GeoRaster  # noqa: E402


def _raster(z, cell_size=10.0):
    return CONVERTERS.convert(GeoRaster(z=z, cell_size=cell_size, x_min=0.0, y_min=0.0),
                              "lsdtt3.Raster", from_type="georaster")


def _values(r):
    return CONVERTERS.convert(r, "field2d", from_type="lsdtt3.Raster")


def test_polyfit_metrics_plane_and_bowl():
    proc = PROCESSES.get("lsdtt3.polyfit_metrics")
    n, dx = 41, 10.0
    x = np.arange(n) * dx
    plane = np.tile(0.1 * x, (n, 1))
    out = proc(raster=_raster(plane), return_slope=True, return_laplacian=True)
    assert set(out) == {"slope", "laplacian"}
    inner = (slice(5, -5), slice(5, -5))
    assert np.allclose(_values(out["slope"])[inner], 0.1, atol=1e-6)
    assert np.allclose(_values(out["laplacian"])[inner], 0.0, atol=1e-8)

    # z = a (x² + y²): laplacian 4a everywhere, concave-up positive.
    a = 1e-3
    c = (n - 1) * dx / 2
    xx, yy = np.meshgrid(x - c, x - c)
    bowl = a * (xx ** 2 + yy ** 2)
    out = proc(raster=_raster(bowl), lambda_metres=50.0, return_slope=False,
               return_laplacian=True, return_mean_curvature=True)
    assert np.allclose(_values(out["laplacian"])[inner], 4 * a, rtol=1e-4)
    assert _values(out["mean_curvature"])[n // 2, n // 2] == pytest.approx(2 * a, rel=1e-4)


def test_polyfit_metrics_needs_a_metric():
    with pytest.raises(ValueError, match="no metric"):
        PROCESSES.get("lsdtt3.polyfit_metrics")(raster=_raster(np.zeros((20, 20))),
                                                return_slope=False)
