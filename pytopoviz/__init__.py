"""pytopoviz placeholder package."""

__version__ = "0.0.1"

from .map_object import MapObject
from .hillshading import hillshade, multishade, smooth_hillshade, smooth_multishade
from .fig2d import Fig2DObject, quickmap
from .fig3d import quickmap3d, Fig3DObject
from .processing import (
    ProcessorFactory,
    ProcessingFunction,
    expand_plottables,
    is_plottable,
    processor,
)
from .helper3d import (
    scale,
    double_scale,
    halve_scale,
    tenfold,
    tenthfold,
    lighting_control,
    matte_lighting,
    glossy_lighting,
    flat_lighting,
    dramatic_lighting,
    heightmap_lighting,
    lighting_intensity_up,
    lighting_intensity_down,
    lighting_brighten,
    lighting_darken,
    light_rotate_left,
    light_rotate_right,
    light_raise,
    light_lower,
    BUILTIN_3D,
)
from .masknan import nan_above, nan_below, nan_equal, nan_mask, BUILTIN_MASK_NAN
from .shading2d import hillshade_processor, multishade_processor, BUILTIN_SHADING
from .filter2d import gaussian_smooth, BUILTIN_FILTERS
from .style2d import (
    apply_dark_pres_mono_style,
    apply_color_pres_style,
    apply_paper_style,
    apply_bw_paper_style,
    apply_nothing_style,
    set_style,
    get_style,
)
from .helper2d import convert_ticks_to_km, add_grid_crosses, add_colorbar
from .helper2d_text import set_font, set_font_size, set_font_style, set_font_color

__all__ = [
    "MapObject",
    "ProcessingFunction",
    "ProcessorFactory",
    "expand_plottables",
    "is_plottable",
    "processor",
    "scale",
    "double_scale",
    "halve_scale",
    "tenfold",
    "tenthfold",
    "lighting_control",
    "matte_lighting",
    "glossy_lighting",
    "flat_lighting",
    "dramatic_lighting",
    "heightmap_lighting",
    "lighting_intensity_up",
    "lighting_intensity_down",
    "lighting_brighten",
    "lighting_darken",
    "light_rotate_left",
    "light_rotate_right",
    "light_raise",
    "light_lower",
    "nan_equal",
    "nan_below",
    "nan_above",
    "nan_mask",
    "hillshade_processor",
    "multishade_processor",
    "gaussian_smooth",
    "BUILTIN_MASK_NAN",
    "BUILTIN_SHADING",
    "BUILTIN_FILTERS",
    "BUILTIN_3D",
    "Fig2DObject",
    "Fig3DObject",
    "hillshade",
    "multishade",
    "smooth_hillshade",
    "smooth_multishade",
    "quickmap",
    "quickmap3d",
    "apply_dark_pres_mono_style",
    "apply_color_pres_style",
    "apply_paper_style",
    "apply_bw_paper_style",
    "apply_nothing_style",
    "set_style",
    "get_style",
    "convert_ticks_to_km",
    "add_grid_crosses",
    "set_font",
    "set_font_size",
    "set_font_style",
    "set_font_color",
    "__version__",
]
