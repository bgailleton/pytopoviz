"""2D filtering processing utilities.

Author: B.G.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .map_object import MapObject
from .processing import ProcessingFunction, ProcessorFactory


def gaussian_smooth(sigma: float = 1.0, mode: str = "nearest") -> ProcessingFunction:
    """
    Return processor that applies a 2D Gaussian filter to MapObject values.

    NaNs are preserved by weighting the Gaussian filter by the valid-data mask.
    Author: B.G.
    """

    def process(self: ProcessingFunction, mapper: MapObject):
        data = mapper.value
        finite_mask = np.isfinite(data)

        # Weighting approach: smooth both data and mask, then renormalize
        filled = np.where(finite_mask, data, 0.0)
        smoothed_data = gaussian_filter(filled, sigma=self.sigma, mode=self.mode)
        weights = gaussian_filter(finite_mask.astype(float), sigma=self.sigma, mode=self.mode)

        with np.errstate(invalid="ignore", divide="ignore"):
            result = smoothed_data / weights
        result[weights == 0.0] = np.nan

        mapper.value = result.astype(np.float32)
        return None

    return ProcessorFactory.build(
        "gaussian_smooth",
        process,
        recursive=True,
        sigma=sigma,
        mode=mode,
    )


BUILTIN_FILTERS = {
    "gaussian_smooth": gaussian_smooth,
}

