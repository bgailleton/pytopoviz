"""Mask/NaN processing utilities.

Author: B.G.
"""

import numpy as np

from .processing import ProcessingFunction, ProcessorFactory
from .map_object import MapObject


def nan_equal(target: float) -> ProcessingFunction:
    """Return processor that masks values equal to ``target`` to NaN. Author: B.G."""

    def process(self: ProcessingFunction, mapper: MapObject):
        mapper.value[mapper.value == self.target] = np.nan
        return None

    return ProcessorFactory.build("nan_equal", process, recursive=True, target=target)


def nan_below(threshold: float=0.) -> ProcessingFunction:
    """Return processor that masks values <= threshold to NaN. Author: B.G."""

    def process(self: ProcessingFunction, mapper: MapObject):
        mapper.value[mapper.value <= self.threshold] = np.nan
        return None

    return ProcessorFactory.build("nan_below", process, recursive=True, threshold=threshold)


def nan_above(threshold: float=0.) -> ProcessingFunction:
    """Return processor that masks values >= threshold to NaN. Author: B.G."""

    def process(self: ProcessingFunction, mapper: MapObject):
        mapper.value[mapper.value >= self.threshold] = np.nan
        return None

    return ProcessorFactory.build("nan_above", process, recursive=True, threshold=threshold)

def nan_mask(mask: np.ndarray) -> ProcessingFunction:
    """Return processor that masks values using a 2D boolean/int mask. Author: B.G."""

    def process(self: ProcessingFunction, mapper: MapObject):
        mask_arr = np.asarray(self.mask)
        if mask_arr.shape != mapper.value.shape:
            raise ValueError("nan_mask expects a mask with the same shape as mapper values.")
        mapper.value[mask_arr.astype(bool)] = np.nan
        return None

    return ProcessorFactory.build("nan_mask", process, recursive=True, mask=mask)


BUILTIN_MASK_NAN = {
    "nan_equal": nan_equal,
    "nan_below": nan_below,
    "nan_above": nan_above,
    "nan_mask": nan_mask,
}
