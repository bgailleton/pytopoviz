"""Shading-related processing utilities.

Author: B.G.
"""

from typing import Sequence

from .hillshading import hillshade, multishade
from .map_object import MapObject
from .processing import ProcessingFunction, ProcessorFactory


def hillshade_processor(
    azimuth: float = 315.0,
    altitude: float = 50.0,
    exaggerate: float = 1.0,
    fused: bool = True,
    alpha: float = 0.45,
) -> ProcessingFunction:
    """Return processor generating a hillshade MapObject. Author: B.G."""

    def process(self: ProcessingFunction, mapper: MapObject):
        return hillshade(
            mapper,
            azimuth=self.azimuth,
            altitude=self.altitude,
            exaggerate=self.exaggerate,
            fused=self.fused,
            alpha=self.alpha,
        )

    return ProcessorFactory.build(
        "hillshade",
        process,
        recursive=False,
        azimuth=azimuth,
        altitude=altitude,
        exaggerate=exaggerate,
        fused=fused,
        alpha=alpha,
    )


def multishade_processor(
    azimuths: Sequence[float] = (315.0, 135.0),
    altitude: float = 50.0,
    exaggerate: float = 1.0,
    fused: bool = True,
    alpha: float = 0.45,
) -> ProcessingFunction:
    """Return processor generating averaged dual-azimuth hillshade. Author: B.G."""

    def process(self: ProcessingFunction, mapper: MapObject):
        return multishade(
            mapper,
            azimuths=tuple(self.azimuths),
            altitude=self.altitude,
            exaggerate=self.exaggerate,
            fused=self.fused,
            alpha=self.alpha,
        )

    return ProcessorFactory.build(
        "multishade",
        process,
        recursive=False,
        azimuths=tuple(azimuths),
        altitude=altitude,
        exaggerate=exaggerate,
        fused=fused,
        alpha=alpha,
    )


BUILTIN_SHADING = {
    "hillshade": hillshade_processor,
    "multishade": multishade_processor,
}
