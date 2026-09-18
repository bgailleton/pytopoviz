"""Processing helpers to transform MapObjects before plotting.

Author: B.G.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .map_object import MapObject


@dataclass
class ProcessingFunction:
    """Callable that mutates a MapObject or returns new MapObjects.

    Attributes
    ----------
    name : str
        Identifier for the processor (e.g., "hillshade").
    apply : callable
        Function taking (self, mapper) that either mutates mapper in-place,
        returns a MapObject, a list/tuple of MapObjects, or None.
    recursive : bool
        When True, this processor is propagated to any child MapObjects
        produced during processing.
    compatible_2d : bool
        Whether the processor applies to 2D plotting.
    compatible_3d : bool
        Whether the processor applies to 3D plotting.
    """
    name: str
    apply: Callable[["ProcessingFunction", MapObject], MapObject | Sequence[MapObject] | None]
    recursive: bool = True
    compatible_2d: bool = True
    compatible_3d: bool = True

    def __call__(self, mapper: MapObject):
        return self.apply(self, mapper)

    def __post_init__(self) -> None:
        # Keep string-keyed compatibility flags for easy getattr usage.
        setattr(self, "2d_compatible", bool(self.compatible_2d))
        setattr(self, "3d_compatible", bool(self.compatible_3d))


class ProcessorFactory:
    """Helper to build ProcessingFunction instances.

    Author: B.G.
    """

    @staticmethod
    def build(
        name: str,
        apply: Callable[["ProcessingFunction", MapObject], MapObject | Sequence[MapObject] | None],
        recursive: bool = True,
        compatible_2d: bool = True,
        compatible_3d: bool = True,
        **params,
    ) -> ProcessingFunction:
        """Create a ProcessingFunction and attach provided params as attributes."""
        proc = ProcessingFunction(
            name=name,
            apply=apply,
            recursive=recursive,
            compatible_2d=compatible_2d,
            compatible_3d=compatible_3d,
        )
        for key, val in params.items():
            setattr(proc, key, val)
        return proc


def is_plottable(obj) -> bool:
    """Return True when object can be rendered by topoviz."""
    return isinstance(obj, MapObject)

def _processor_is_compatible(proc: ProcessingFunction, mode: str | None) -> bool:
    if mode is None:
        return True
    if mode == "2d":
        return bool(getattr(proc, "2d_compatible", proc.compatible_2d))
    if mode == "3d":
        return bool(getattr(proc, "3d_compatible", proc.compatible_3d))
    raise ValueError("mode must be '2d', '3d', or None.")

def expand_plottables(mapper: MapObject, mode: str | None = None) -> list[MapObject]:
    """Apply processors in-order and collect plottables depth-first.

    The walk honors the order of `mapper.processors`. Each processor can:
    - mutate the current MapObject in-place and return None
    - return one or more new MapObjects to be plotted (children)

    Children are immediately walked before proceeding, so the plotting order is:
    parent (post-mutation) then its spawned children (post-processing), in the
    sequence the processors define.
    """

    def walk(current: MapObject) -> list[MapObject]:
        collected: list[MapObject] = [current]
        for proc in current.processors:
            if not _processor_is_compatible(proc, mode):
                continue
            produced = proc(current)
            if produced is None:
                continue
            produced_list: Iterable = produced if isinstance(produced, (list, tuple)) else (produced,)
            for item in produced_list:
                if not is_plottable(item):
                    continue
                collected.extend(walk(item))
        return collected

    return walk(mapper)


class _ProcessorNamespace:
    """Namespace-style accessor for built-in processors grouped by module."""

    def __init__(self):
        from . import filter2d, helper3d, masknan, shading2d  # local import to avoid cycles

        self.masknan = masknan
        self.shading2d = shading2d
        self.filter2d = filter2d
        self.helper3d = helper3d
        # Convenient top-level aliases
        self.nan_equal = masknan.nan_equal
        self.nan_below = masknan.nan_below
        self.nan_above = masknan.nan_above
        self.nan_mask = masknan.nan_mask
        self.hillshade = shading2d.hillshade_processor
        self.multishade = shading2d.multishade_processor
        self.gaussian_smooth = filter2d.gaussian_smooth
        self.scale = helper3d.scale
        self.double_scale = helper3d.double_scale
        self.halve_scale = helper3d.halve_scale
        self.tenfold = helper3d.tenfold
        self.tenthfold = helper3d.tenthfold
        self.lighting_control = helper3d.lighting_control
        self.matte_lighting = helper3d.matte_lighting
        self.glossy_lighting = helper3d.glossy_lighting
        self.flat_lighting = helper3d.flat_lighting
        self.dramatic_lighting = helper3d.dramatic_lighting
        self.heightmap_lighting = helper3d.heightmap_lighting
        self.lighting_intensity_up = helper3d.lighting_intensity_up
        self.lighting_intensity_down = helper3d.lighting_intensity_down
        self.lighting_brighten = helper3d.lighting_brighten
        self.lighting_darken = helper3d.lighting_darken
        self.light_rotate_left = helper3d.light_rotate_left
        self.light_rotate_right = helper3d.light_rotate_right
        self.light_raise = helper3d.light_raise
        self.light_lower = helper3d.light_lower


# Expose a shared namespace instance
# This enables convenient access such as tpz.processor.multishade()
processor = _ProcessorNamespace()
