"""pytopoviz — plumbing framework between science libraries and any frontend.

The framework lives in :mod:`pytopoviz.core` (types, processes, converters,
sessions, workflows, contract). See ``DESIGN.md``.

The legacy 2D/3D figure code (``MapObject``, ``Fig2DObject``, processors, styles)
lives in :mod:`pytopoviz.legacy` — pre-alpha throwaway kept importable for reference,
being rethought from scratch as ``figmaker`` (DESIGN.md §9). It is loaded lazily and
optional heavy deps (e.g. pyvista) never gate importing the framework.
"""

from __future__ import annotations

__version__ = "0.0.1"

from . import core  # framework — always available

__all__ = ["core", "__version__"]


# ---- legacy figure API (optional, lazy) -------------------------------------
# Exposed on attribute access so a missing viz dependency does not break
# `import pytopoviz` or `import pytopoviz.core`.

_LEGACY_EXPORTS = {
    "MapObject": ".legacy.map_object",
    "hillshade": ".legacy.hillshading",
    "multishade": ".legacy.hillshading",
    "smooth_hillshade": ".legacy.hillshading",
    "smooth_multishade": ".legacy.hillshading",
    "Fig2DObject": ".legacy.fig2d",
    "quickmap": ".legacy.fig2d",
    "quickmap3d": ".legacy.fig3d",
    "Fig3DObject": ".legacy.fig3d",
    "ProcessorFactory": ".legacy.processing",
    "ProcessingFunction": ".legacy.processing",
    "expand_plottables": ".legacy.processing",
    "is_plottable": ".legacy.processing",
    "processor": ".legacy.processing",
    "nan_above": ".legacy.masknan",
    "nan_below": ".legacy.masknan",
    "nan_equal": ".legacy.masknan",
    "nan_mask": ".legacy.masknan",
    "hillshade_processor": ".legacy.shading2d",
    "multishade_processor": ".legacy.shading2d",
    "gaussian_smooth": ".legacy.filter2d",
    "set_style": ".legacy.style2d",
    "get_style": ".legacy.style2d",
    "convert_ticks_to_km": ".legacy.helper2d",
    "add_grid_crosses": ".legacy.helper2d",
    "add_colorbar": ".legacy.helper2d",
}


def __getattr__(name: str):
    module_path = _LEGACY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'pytopoviz' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)
