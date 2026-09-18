"""3D-only processing helpers.

Author: B.G.
"""

from .map_object import MapObject
from .processing import ProcessingFunction, ProcessorFactory

_DEFAULT_LIGHT_AZIMUTH = 315.0
_DEFAULT_LIGHT_ELEVATION = 45.0


def scale(factor: float) -> ProcessingFunction:
    """Return a 3D-only processor that scales 3D surface height."""

    def process(self: ProcessingFunction, mapper: MapObject):
        mapper.z_scale_factor = mapper.z_scale_factor * self.factor
        return None

    return ProcessorFactory.build(
        "scale",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
        factor=float(factor),
    )


def double_scale() -> ProcessingFunction:
    """Convenience processor: double surface height."""
    return scale(2.0)


def halve_scale() -> ProcessingFunction:
    """Convenience processor: halve surface height."""
    return scale(0.5)


def tenfold() -> ProcessingFunction:
    """Convenience processor: multiply surface height by 10."""
    return scale(10.0)


def tenthfold() -> ProcessingFunction:
    """Convenience processor: multiply surface height by 0.1."""
    return scale(0.1)


def lighting_control(
    *,
    enabled: bool | None = None,
    ambient: float | None = None,
    diffuse: float | None = None,
    specular: float | None = None,
    specular_power: float | None = None,
    smooth_shading: bool | None = None,
    eye_dome_lighting: bool | None = None,
    light_azimuth: float | None = None,
    light_elevation: float | None = None,
    light_intensity: float | None = None,
) -> ProcessingFunction:
    """Return a 3D-only processor that configures lighting safely."""

    def process(self: ProcessingFunction, mapper: MapObject):
        if self.enabled is False:
            mapper.ambient = 1.0
            mapper.diffuse = 0.0
            mapper.specular = 0.0
            mapper.specular_power = 1.0
            mapper.smooth_shading = False
        else:
            if self.ambient is not None:
                mapper.ambient = self.ambient
            if self.diffuse is not None:
                mapper.diffuse = self.diffuse
            if self.specular is not None:
                mapper.specular = self.specular
            if self.specular_power is not None:
                mapper.specular_power = self.specular_power
            if self.smooth_shading is not None:
                mapper.smooth_shading = self.smooth_shading

        if self.eye_dome_lighting is not None:
            mapper.eye_dome_lighting = self.eye_dome_lighting
        if self.light_azimuth is not None:
            mapper.light_azimuth = self.light_azimuth
        if self.light_elevation is not None:
            mapper.light_elevation = self.light_elevation
        if self.light_intensity is not None:
            mapper.light_intensity = self.light_intensity
        return None

    return ProcessorFactory.build(
        "lighting_control",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
        enabled=enabled,
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        specular_power=specular_power,
        smooth_shading=smooth_shading,
        eye_dome_lighting=eye_dome_lighting,
        light_azimuth=light_azimuth,
        light_elevation=light_elevation,
        light_intensity=light_intensity,
    )


def matte_lighting() -> ProcessingFunction:
    """Preset: low specular highlights for a matte look."""
    return lighting_control(ambient=0.2, diffuse=0.85, specular=0.02, specular_power=5.0)


def glossy_lighting() -> ProcessingFunction:
    """Preset: stronger specular highlights for a glossy look."""
    return lighting_control(ambient=0.12, diffuse=0.75, specular=0.35, specular_power=20.0)


def flat_lighting() -> ProcessingFunction:
    """Preset: flatten shading for a cartographic look."""
    return lighting_control(ambient=1.0, diffuse=0.0, specular=0.0, specular_power=1.0)


def dramatic_lighting() -> ProcessingFunction:
    """Preset: higher contrast lighting."""
    return lighting_control(ambient=0.08, diffuse=0.95, specular=0.15, specular_power=25.0)


def heightmap_lighting() -> ProcessingFunction:
    """Preset: balanced lighting for large heightmaps."""
    return lighting_control(
        ambient=0.12,
        diffuse=0.9,
        specular=0.08,
        specular_power=18.0,
        smooth_shading=True,
        eye_dome_lighting=True,
        light_azimuth=_DEFAULT_LIGHT_AZIMUTH,
        light_elevation=35.0,
        light_intensity=1.1,
    )


def lighting_intensity_up() -> ProcessingFunction:
    """Relative: increase lighting intensity by 20%."""

    def process(self: ProcessingFunction, mapper: MapObject):
        mapper.ambient = mapper.ambient * 1.2
        mapper.diffuse = mapper.diffuse * 1.2
        mapper.specular = mapper.specular * 1.2
        return None

    return ProcessorFactory.build(
        "lighting_intensity_up",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
    )


def lighting_intensity_down() -> ProcessingFunction:
    """Relative: decrease lighting intensity by 20%."""

    def process(self: ProcessingFunction, mapper: MapObject):
        mapper.ambient = mapper.ambient * 0.8
        mapper.diffuse = mapper.diffuse * 0.8
        mapper.specular = mapper.specular * 0.8
        return None

    return ProcessorFactory.build(
        "lighting_intensity_down",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
    )


def lighting_brighten() -> ProcessingFunction:
    """Relative: brighten by boosting ambient and diffuse."""

    def process(self: ProcessingFunction, mapper: MapObject):
        mapper.ambient = mapper.ambient * 1.3
        mapper.diffuse = mapper.diffuse * 1.15
        return None

    return ProcessorFactory.build(
        "lighting_brighten",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
    )


def lighting_darken() -> ProcessingFunction:
    """Relative: darken by reducing ambient and diffuse."""

    def process(self: ProcessingFunction, mapper: MapObject):
        mapper.ambient = mapper.ambient * 0.7
        mapper.diffuse = mapper.diffuse * 0.85
        return None

    return ProcessorFactory.build(
        "lighting_darken",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
    )


def light_rotate_left(degrees: float = 20.0) -> ProcessingFunction:
    """Relative: rotate light azimuth left (counter-clockwise)."""

    def process(self: ProcessingFunction, mapper: MapObject):
        azim = mapper.light_azimuth if mapper.light_azimuth is not None else _DEFAULT_LIGHT_AZIMUTH
        mapper.light_azimuth = azim - self.degrees
        if mapper.light_elevation is None:
            mapper.light_elevation = _DEFAULT_LIGHT_ELEVATION
        return None

    return ProcessorFactory.build(
        "light_rotate_left",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
        degrees=float(degrees),
    )


def light_rotate_right(degrees: float = 20.0) -> ProcessingFunction:
    """Relative: rotate light azimuth right (clockwise)."""

    def process(self: ProcessingFunction, mapper: MapObject):
        azim = mapper.light_azimuth if mapper.light_azimuth is not None else _DEFAULT_LIGHT_AZIMUTH
        mapper.light_azimuth = azim + self.degrees
        if mapper.light_elevation is None:
            mapper.light_elevation = _DEFAULT_LIGHT_ELEVATION
        return None

    return ProcessorFactory.build(
        "light_rotate_right",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
        degrees=float(degrees),
    )


def light_raise(degrees: float = 10.0) -> ProcessingFunction:
    """Relative: raise light elevation."""

    def process(self: ProcessingFunction, mapper: MapObject):
        elev = mapper.light_elevation if mapper.light_elevation is not None else _DEFAULT_LIGHT_ELEVATION
        mapper.light_elevation = elev + self.degrees
        if mapper.light_azimuth is None:
            mapper.light_azimuth = _DEFAULT_LIGHT_AZIMUTH
        return None

    return ProcessorFactory.build(
        "light_raise",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
        degrees=float(degrees),
    )


def light_lower(degrees: float = 10.0) -> ProcessingFunction:
    """Relative: lower light elevation."""

    def process(self: ProcessingFunction, mapper: MapObject):
        elev = mapper.light_elevation if mapper.light_elevation is not None else _DEFAULT_LIGHT_ELEVATION
        mapper.light_elevation = elev - self.degrees
        if mapper.light_azimuth is None:
            mapper.light_azimuth = _DEFAULT_LIGHT_AZIMUTH
        return None

    return ProcessorFactory.build(
        "light_lower",
        process,
        recursive=True,
        compatible_2d=False,
        compatible_3d=True,
        degrees=float(degrees),
    )


BUILTIN_3D = {
    "scale": scale,
    "double_scale": double_scale,
    "halve_scale": halve_scale,
    "tenfold": tenfold,
    "tenthfold": tenthfold,
    "lighting_control": lighting_control,
    "matte_lighting": matte_lighting,
    "glossy_lighting": glossy_lighting,
    "flat_lighting": flat_lighting,
    "dramatic_lighting": dramatic_lighting,
    "heightmap_lighting": heightmap_lighting,
    "lighting_intensity_up": lighting_intensity_up,
    "lighting_intensity_down": lighting_intensity_down,
    "lighting_brighten": lighting_brighten,
    "lighting_darken": lighting_darken,
    "light_rotate_left": light_rotate_left,
    "light_rotate_right": light_rotate_right,
    "light_raise": light_raise,
    "light_lower": light_lower,
}
