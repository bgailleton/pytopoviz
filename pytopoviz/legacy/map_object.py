"""Container for visualization settings tied to a GridObject.

Author: B.G.
"""

import secrets
import string
from typing import List, Optional, Union

import numpy as np
from matplotlib.colors import Colormap

from topotoolbox import GridObject

__all__ = ["MapObject"]


class MapObject:
    """Store visualization defaults for a GridObject."""

    def __init__(
        self,
        grid: GridObject,
        cmap: Union[str, Colormap] = "terrain",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha: float = 1.0,
        cbar: Optional[str] = None,
        name: Optional[str] = None,
        processors: Optional[list] = None,
        draped: bool = False,
        ambient: float = 0.15,
        diffuse: float = 0.8,
        specular: float = 0.1,
        specular_power: float = 10.0,
        smooth_shading: Optional[bool] = None,
        eye_dome_lighting: Optional[bool] = None,
        light_azimuth: Optional[float] = None,
        light_elevation: Optional[float] = None,
        light_intensity: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        grid : GridObject
            The grid to visualize.
        cmap : str or matplotlib.colors.Colormap, optional
            The colormap to apply when displaying the grid.
        vmin : float, optional
            Lower bound for normalization. Defaults to the minimum finite value.
        vmax : float, optional
            Upper bound for normalization. Defaults to the maximum finite value.
        alpha : float, optional
            Transparency value in [0, 1]. Defaults to 1.0 (opaque).
        cbar : str or None, optional
            If a string, use it as the colorbar label when plotting.
        name : str, optional
            Identifier for this MapObject. Defaults to a random 8-character string.
        processors : list, optional
            ProcessingFunction instances to apply when rendering.
        draped : bool, optional
            If True, use the parent surface geometry in 3D and only affect color.
        ambient : float, optional
            3D lighting ambient component.
        diffuse : float, optional
            3D lighting diffuse component.
        specular : float, optional
            3D lighting specular component.
        specular_power : float, optional
            3D lighting specular exponent.
        smooth_shading : bool or None, optional
            Override smooth shading for 3D rendering when set.
        eye_dome_lighting : bool or None, optional
            Override eye dome lighting for 3D rendering when set.
        light_azimuth : float or None, optional
            Scene light azimuth override (degrees).
        light_elevation : float or None, optional
            Scene light elevation override (degrees).
        light_intensity : float or None, optional
            Scene light intensity override.
        """
        self._grid = grid
        self._cmap = cmap
        self._alpha = alpha
        self._cbar = cbar
        self._name = self._generate_name() if name is None else self._validate_name(name)
        self.processors: List = list(processors) if processors is not None else []
        self._draped = bool(draped)
        self._z_scale_factor = 1.0
        self._ambient = float(ambient)
        self._diffuse = float(diffuse)
        self._specular = float(specular)
        self._specular_power = float(specular_power)
        self._smooth_shading = smooth_shading
        self._eye_dome_lighting = eye_dome_lighting
        self._light_azimuth = light_azimuth
        self._light_elevation = light_elevation
        self._light_intensity = light_intensity

        self._value = self._prepare_value(grid.z)

        if vmin is None or vmax is None:
            auto_min, auto_max = self._nan_aware_minmax(self._value)
            self._vmin = auto_min if vmin is None else vmin
            self._vmax = auto_max if vmax is None else vmax
        else:
            self._vmin = vmin
            self._vmax = vmax

    @staticmethod
    def _prepare_value(value: np.ndarray) -> np.ndarray:
        """Return float32 array with non-finite entries as NaN."""
        val = np.array(value, dtype=np.float32, copy=True)
        val[~np.isfinite(val)] = np.nan
        return val

    @staticmethod
    def _nan_aware_minmax(arr: np.ndarray) -> tuple[float, float]:
        """Return (min, max) ignoring NaNs; (nan, nan) if no finite values."""
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            return float(np.nanmin(arr)), float(np.nanmax(arr))
        return np.nan, np.nan

    @staticmethod
    def _generate_name() -> str:
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(8))

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("MapObject name must be a non-empty string.")
        return name

    @property
    def grid(self) -> GridObject:
        return self._grid

    @grid.setter
    def grid(self, grid: GridObject) -> None:
        self._grid = grid
        self.value = grid.z

    @property
    def cmap(self) -> Union[str, Colormap]:
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: Union[str, Colormap]) -> None:
        self._cmap = cmap

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = alpha

    @property
    def vmin(self) -> float:
        return self._vmin

    @vmin.setter
    def vmin(self, vmin: float) -> None:
        self._vmin = vmin

    @property
    def vmax(self) -> float:
        return self._vmax

    @vmax.setter
    def vmax(self, vmax: float) -> None:
        self._vmax = vmax

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        self._value = self._prepare_value(value)
        self._vmin, self._vmax = self._nan_aware_minmax(self._value)

    @property
    def cbar(self) -> Optional[str]:
        return self._cbar

    @cbar.setter
    def cbar(self, label: Optional[str]) -> None:
        self._cbar = label

    @property
    def name(self) -> str:
        return self._name

    @property
    def draped(self) -> bool:
        return self._draped

    @draped.setter
    def draped(self, draped: bool) -> None:
        self._draped = bool(draped)

    @property
    def z_scale_factor(self) -> float:
        return self._z_scale_factor

    @z_scale_factor.setter
    def z_scale_factor(self, value: float) -> None:
        self._z_scale_factor = float(value)

    @property
    def ambient(self) -> float:
        return self._ambient

    @ambient.setter
    def ambient(self, value: float) -> None:
        self._ambient = max(0.0, min(1.0, float(value)))

    @property
    def diffuse(self) -> float:
        return self._diffuse

    @diffuse.setter
    def diffuse(self, value: float) -> None:
        self._diffuse = max(0.0, min(1.0, float(value)))

    @property
    def specular(self) -> float:
        return self._specular

    @specular.setter
    def specular(self, value: float) -> None:
        self._specular = max(0.0, min(1.0, float(value)))

    @property
    def specular_power(self) -> float:
        return self._specular_power

    @specular_power.setter
    def specular_power(self, value: float) -> None:
        self._specular_power = max(0.0, float(value))

    @property
    def smooth_shading(self) -> Optional[bool]:
        return self._smooth_shading

    @smooth_shading.setter
    def smooth_shading(self, value: Optional[bool]) -> None:
        self._smooth_shading = None if value is None else bool(value)

    @property
    def eye_dome_lighting(self) -> Optional[bool]:
        return self._eye_dome_lighting

    @eye_dome_lighting.setter
    def eye_dome_lighting(self, value: Optional[bool]) -> None:
        self._eye_dome_lighting = None if value is None else bool(value)

    @property
    def light_azimuth(self) -> Optional[float]:
        return self._light_azimuth

    @light_azimuth.setter
    def light_azimuth(self, value: Optional[float]) -> None:
        self._light_azimuth = None if value is None else float(value)

    @property
    def light_elevation(self) -> Optional[float]:
        return self._light_elevation

    @light_elevation.setter
    def light_elevation(self, value: Optional[float]) -> None:
        self._light_elevation = None if value is None else float(value)

    @property
    def light_intensity(self) -> Optional[float]:
        return self._light_intensity

    @light_intensity.setter
    def light_intensity(self, value: Optional[float]) -> None:
        self._light_intensity = None if value is None else float(value)

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MapObject):
            return False
        return self._name == other._name
