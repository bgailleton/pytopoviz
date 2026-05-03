# pytopoviz

Quantitative geomorphology heavily relies on map visualisations, with unavoidable arbitrary choices for parameters — e.g. colormap limits, line style, which watershed to select, etc.

`pytopoviz` is a visualisation framework that addresses this by providing a way to craft reproducible recipes to make 2D and 3D figures from [`pytopotoolbox`](https://github.com/TopoToolbox/pytopotoolbox) or your own data. It also provides common helpers to streamline sometimes obscure `matplotlib` features (e.g. cleanly changing font size and color for a given axis, automatically converting axis ticks from m to km, etc.).

2D figures are built on top of `matplotlib`, 3D figures on `pyvista`.

**_The software is still in alpha but under heavy development since November 2025._**

## Quick start

**Installation:**

```bash
git clone https://github.com/TopoToolbox/pytopoviz
cd pytopoviz
pip install .
```

(_Once stable, there will be a pip package_)

**Dependencies** (installed automatically via pip):

| Package | Role |
|---|---|
| `topotoolbox >= 0.0.7` | Core grid and analysis objects |
| `numpy >= 1.23.5` | Array operations |
| `matplotlib` | 2D figure rendering |
| `pyvista` | 3D figure rendering |
| `scipy` | Spatial filters (e.g. Gaussian smooth) |
| `numba` | JIT-compiled processing kernels |
| `cmcrameri` | Perceptually uniform scientific colormaps |
| `rasterio` | Raster I/O |
| `geopandas` / `shapely` | Vector geometry support |
| `clarabel` | Quadratic programming solver |

Optional extras: `pip install "pytopoviz[docs]"` (Sphinx) or `pip install "pytopoviz[test]"` (pytest).

**2D figure:**

```python
import pytopoviz as ptv
from topotoolbox import GridObject

dem = GridObject("my_dem.tif")

# Wrap the grid with visualisation settings and attach a hillshade processor
m = ptv.MapObject(dem, cmap="terrain", cbar="Elevation (m)")
m.processors.append(ptv.hillshade_processor())

# Build the figure, apply a style, adjust text
ptv.set_style("dark_pres_mono")
fig = ptv.quickmap(m)
ptv.set_font_size(fig.fig, 11, which="tick")
ptv.convert_ticks_to_km(fig.ax)
fig.save(fname="map2d.png", dpi=300)
```

**3D figure:**

```python
m3d = ptv.MapObject(dem, cmap="terrain")
m3d.processors.append(ptv.dramatic_lighting())

fig3d = ptv.quickmap3d(m3d)
fig3d.show()
```

## Features

### Ready

**Core**
- `MapObject` — wraps a `GridObject` with colormap, colorbar, alpha, and processor settings
- `Fig2DObject` / `quickmap` — matplotlib-based 2D figure builder with layered map support
- `Fig3DObject` / `quickmap3d` — pyvista-based 3D figure builder

**Processors** (composable, attach to `MapObject.processors`)
- Hillshading: `hillshade_processor`, `multishade_processor`
- Gaussian smoothing: `gaussian_smooth`
- NaN masking: `nan_above`, `nan_below`, `nan_equal`, `nan_mask`
- 3D scale control: `scale`, `double_scale`, `halve_scale`, `tenfold`, `tenthfold`
- 3D lighting presets: `matte_lighting`, `glossy_lighting`, `flat_lighting`, `dramatic_lighting`, `heightmap_lighting`
- 3D lighting adjustments: `lighting_control`, `lighting_brighten/darken`, `lighting_intensity_up/down`, `light_rotate_left/right`, `light_raise/lower`

**matplotlib helpers**
- `convert_ticks_to_km` — reformat axis tick labels from metres to kilometres
- `add_colorbar` — add an inset colorbar with consistent styling
- `add_grid_crosses` — overlay grid reference crosses
- `set_font` / `set_font_size` / `set_font_style` / `set_font_color` — fine-grained font control per figure, axis, or category (title, label, tick, legend, colorbar)

**Style presets** (`set_style` / `get_style`)
- `dark_pres_mono` — dark background, monochrome presentation style
- `color_pres` — dark background with colour
- `paper` — light background for publication figures
- `bw_paper` — black-and-white publication style
- `nothing` — fully transparent, no axes, no text (for compositing)

### WIP

We plan to maintain a high pace of updates. Here is an unordered list of future additions:

- More helpers for `matplotlib` style and axis manipulation
- Tutorials, especially on adding custom processors
- Beyond `python` scripting: `JSON` serialisation of recipes for maps, reproducible without writing code
- Topographic analysis steps within recipes
- Minimalist GUI

## How does it work?

The central concept is the **`MapObject`**: a thin wrapper around a `topotoolbox` `GridObject` that stores visualisation parameters (colormap, colorbar label, alpha, lighting settings) and a list of **processors**.

**Processors** are composable transforms attached to a `MapObject`. When a figure is built, each processor is applied in order — a processor can mutate the map in place, replace it, or produce additional derived layers (e.g., a hillshade overlay on top of the elevation). Built-in processors cover hillshading, smoothing, NaN masking, and 3D lighting/scale control. Custom processors can be registered with the `@processor` decorator. The order of processors matters: `nan` masking for example will impact the next processors by adding mask to the `MapObject`.

**`Fig2DObject`** wraps a matplotlib `Figure` and its axes. `add_maps(ax, *maps)` expands processors for 2D, calls `imshow` for each plottable layer, and optionally attaches colorbars. The convenience function `quickmap(*maps)` creates a single-axis figure in one call.

**`Fig3DObject`** wraps a pyvista plotter. `quickmap3d(*maps)` builds a structured surface mesh from the DEM, applies 3D processors (scale, lighting), and returns an interactive or offscreen render.

**Style presets** set matplotlib `rcParams` globally so every figure created afterwards inherits the chosen look. They can be combined with the **`helper2d_text`** functions (`set_font`, `set_font_size`, `set_font_style`, `set_font_color`) to fine-tune typography per figure or axis, or globally via `target=None`.

## `matplotlib` helpers

### Axis ticks and layout

**`convert_ticks_to_km(ax, axes="both")`**  
Reformats tick labels from metres to kilometres and updates axis labels accordingly. `axes` can be `"x"`, `"y"`, or `"both"`.

**`finalize_figsize(mappers, base_height=6.0, ...)`**  
Computes a sensible figure size in inches from the spatial extents of one or more `MapObject`s, optionally reserving extra width for a colorbar.

### Colorbar

**`add_colorbar(ax, mappable, label=None, location="right", size="5%", pad=0.05, shrink=1.0)`**  
Attaches a colorbar to an axes using `inset_axes` so it is anchored to the axes edge rather than the figure. Main usage is to create a colorbar the size of the imshow it's attached to - a surprisingly uninstinctive thing in `matplotlib`. `location` accepts `"right"`, `"left"`, `"top"`, or `"bottom"`. `shrink` controls the fraction of the edge that the colorbar spans (1.0 = full length).

### Grid decoration

**`add_grid_crosses(ax, color="black", size=5, linewidth=1, alpha=0.47, include_minor=True)`**  
Draws `+` markers at every tick intersection — a cleaner alternative to a full grid. Minor ticks produce smaller, more transparent crosses.

### Typography

**`set_font(target, family)`**  
Sets the font family for every `Text` object in a figure, or globally when `target=None`. Accepts a font family name (e.g. `"Arial"`) or a path to a `.ttf`/`.otf` file.

**`set_font_size(target, size, which="all")`**  
Sets font size for selected text categories. `which` can be `"title"`, `"label"`, `"tick"`, `"legend"`, `"colorbar"`, `"all"`, or a list of these. `target=None` applies globally via rcParams.

**`set_font_style(target, style, which="all")`**  
Sets font style/weight. `style` is one of `"normal"`, `"italic"`, `"bold"`, `"bold italic"`. Same `which` and `target=None` semantics as above.

**`set_font_color(target, color, which="all")`**  
Sets font color for selected text categories. `color` is any matplotlib-compatible value (named colour, hex, RGB, RGBA). Same `which` and `target=None` semantics as above.

## Authors

Boris Gailleton (boris.gailleton@univ-rennes.fr)

## Funding

Financial support for this research was provided by the Centre national d’études spatiales (CNES), France (ROR: https://ror.org/04h1h0y33) and by the EU H2020 European Research Council (grant no. 803721).
