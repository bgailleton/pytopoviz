"""Helpers to create hillshade MapObjects.

Author: B.G.
"""

from functools import partial
from typing import Iterable, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from topotoolbox import GridObject

from .map_object import MapObject

__all__ = ["hillshade", "multishade", "smooth_hillshade", "smooth_multishade"]


def _nan_gaussian_smooth(values: np.ndarray, sigma: float, mode: str) -> np.ndarray:
    """Gaussian smooth while preserving NaNs."""
    data = np.asarray(values, dtype=np.float32)
    finite_mask = np.isfinite(data)

    filled = np.where(finite_mask, data, 0.0)
    smoothed_data = gaussian_filter(filled, sigma=sigma, mode=mode)
    weights = gaussian_filter(finite_mask.astype(float), sigma=sigma, mode=mode)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = smoothed_data / weights
    result[weights == 0.0] = np.nan
    return result.astype(np.float32)


def _hillshade_from_values(
    mapper: MapObject,
    values: np.ndarray,
    azimuth: float,
    altitude: float,
    exaggerate: float,
    fused: bool,
    alpha: float,
) -> MapObject:
    values = np.asarray(values, dtype=np.float32)
    mask = np.isnan(values)
    has_finite_values = np.isfinite(values).any()

    grid_values = np.asarray(mapper.grid.z, dtype=np.float32)
    finite_grid = np.isfinite(grid_values)
    fallback_val = float(grid_values[finite_grid].mean()) if finite_grid.any() else 0.0

    clean_values = np.where(np.isfinite(values), values, fallback_val)
    temp_grid: GridObject = mapper.grid.duplicate_with_new_data(clean_values)

    hs_grid: GridObject = temp_grid.hillshade(
        azimuth=azimuth, altitude=altitude, exaggerate=exaggerate, fused=fused
    )

    shaded_values = np.array(hs_grid.z, copy=True)
    if mask.any():
        if has_finite_values:
            shaded_values[mask] = np.nan
        else:
            base_mask = np.isnan(grid_values)
            shaded_values[base_mask] = np.nan

    result = MapObject(
        hs_grid,
        cmap='gray',
        alpha=alpha,
    )
    result.draped = True
    result.value = shaded_values
    return result


def hillshade(
    mapper: MapObject,
    azimuth: float = 315.0,
    altitude: float = 50.0,
    exaggerate: float = 1.0,
    fused: bool = True,
    alpha: float = 0.45,
) -> MapObject:
    """Return a MapObject with the hillshade of the mapper's grid.

    NaN values from the source mapper are propagated to the hillshaded result.
    """
    return _hillshade_from_values(
        mapper,
        mapper.value,
        azimuth=azimuth,
        altitude=altitude,
        exaggerate=exaggerate,
        fused=fused,
        alpha=alpha,
    )


def smooth_hillshade(
    mapper: MapObject,
    sigma: float = 1.0,
    mode: str = "nearest",
    azimuth: float = 315.0,
    altitude: float = 50.0,
    exaggerate: float = 1.0,
    fused: bool = True,
    alpha: float = 0.45,
) -> MapObject:
    """Return hillshade computed on a Gaussian-smoothed copy of mapper values."""
    smoothed = _nan_gaussian_smooth(mapper.value, sigma=sigma, mode=mode)
    return _hillshade_from_values(
        mapper,
        smoothed,
        azimuth=azimuth,
        altitude=altitude,
        exaggerate=exaggerate,
        fused=fused,
        alpha=alpha,
    )


def multishade(
    mapper: MapObject,
    azimuths: Union[Iterable[float], Tuple[float, float]] = (315.0, 135.0),
    altitude: float = 50.0,
    exaggerate: float = 1.0,
    fused: bool = True,
    alpha: float = 0.45,
) -> MapObject:
    """Average two hillshades from different azimuths."""
    azimuth_list = tuple(azimuths)
    if len(azimuth_list) != 2:
        raise ValueError("azimuths must contain exactly two angles.")

    shade1 = hillshade(
        mapper,
        azimuth=azimuth_list[0],
        altitude=altitude,
        exaggerate=exaggerate,
        fused=fused,
    )
    shade2 = hillshade(
        mapper,
        azimuth=azimuth_list[1],
        altitude=altitude,
        exaggerate=exaggerate,
        fused=fused,
    )

    stacked = np.stack([shade1.value, shade2.value])
    valid_counts = np.isfinite(stacked).sum(axis=0)
    summed = np.nansum(stacked, axis=0, dtype=np.float32)
    averaged = np.divide(
        summed,
        valid_counts,
        out=np.full_like(summed, np.nan, dtype=np.float32),
        where=valid_counts > 0,
    )

    new_grid = mapper.grid.astype(np.float32)
    new_grid.z = averaged.astype(np.float32)

    result = MapObject(
        new_grid,
        cmap='gray',
        alpha=alpha,
    )
    result.draped = True
    result.value = averaged
    return result


def smooth_multishade(
    mapper: MapObject,
    azimuths: Union[Iterable[float], Tuple[float, float]] = (315.0, 135.0),
    sigma: float = 1.0,
    mode: str = "nearest",
    altitude: float = 50.0,
    exaggerate: float = 1.0,
    fused: bool = True,
    alpha: float = 0.45,
) -> MapObject:
    """Average two hillshades computed on a Gaussian-smoothed copy."""
    smoother = partial(smooth_hillshade, sigma=sigma, mode=mode)

    azimuth_list = tuple(azimuths)
    if len(azimuth_list) != 2:
        raise ValueError("azimuths must contain exactly two angles.")

    shade1 = smoother(
        mapper,
        azimuth=azimuth_list[0],
        altitude=altitude,
        exaggerate=exaggerate,
        fused=fused,
        alpha=alpha,
    )
    shade2 = smoother(
        mapper,
        azimuth=azimuth_list[1],
        altitude=altitude,
        exaggerate=exaggerate,
        fused=fused,
        alpha=alpha,
    )

    stacked = np.stack([shade1.value, shade2.value])
    valid_counts = np.isfinite(stacked).sum(axis=0)
    summed = np.nansum(stacked, axis=0, dtype=np.float32)
    averaged = np.divide(
        summed,
        valid_counts,
        out=np.full_like(summed, np.nan, dtype=np.float32),
        where=valid_counts > 0,
    )

    new_grid = mapper.grid.astype(np.float32)
    new_grid.z = averaged.astype(np.float32)

    result = MapObject(
        new_grid,
        cmap='gray',
        alpha=alpha,
    )
    result.draped = True
    result.value = averaged
    return result
