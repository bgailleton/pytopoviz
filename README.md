# pytopoviz

`pytopoviz` is visualisation framework for crafting/using reproduciple recipes to make 2D and 3D figure from [`pytopotoolbox`](https://github.com/TopoToolbox/pytopotoolbox) and associated packages. 2D figures are built on top of `matplotlib`, 3D on `pyvista`.

The software is still in alpha but under heavy development since November 2025.

## Features

- style sheets
- automatically handles geographic extent
- 2D composite figure object
- 3D quick figure
- built-in processors:
  + conditional nan masking (e.g. removing data below elevation)
  + hillshading, smoothing other filtering
  + WIP
- (WIP) per-application recipes (e.g. graphflood, slope, ...)
- (WIP) topographic analyses processors (e.g. stream network)

## Usage

Two levels of usage:

### python scripts

Create 2D and 3D figures by chaining processor and data. 

```
TODO
```

## Processors

Processors are small, composable steps attached to a `MapObject` that run in order before plotting. They can modify the map values in place (e.g. smoothing, masking) or spawn derived layers (e.g. hillshade). This lets you build repeatable visual recipes where the same source data produces multiple stacked outputs in 2D or 3D. Processors can also be flagged as 2D- or 3D-only so they are ignored in the other context.

### Masking

- `nan_below`, `nan_above`, `nan_equal` (conditional NaN masking)

### Shading

- `hillshade`, `multishade` (and smooth variants via `smooth_hillshade`, `smooth_multishade`)

### Filtering

- `gaussian_smooth`

### 3D helpers

- `scale`, `double_scale`, `halve_scale`, `tenfold`, `tenthfold`
- `lighting_control`
- `matte_lighting`, `glossy_lighting`, `flat_lighting`, `dramatic_lighting`
- `lighting_intensity_up`, `lighting_intensity_down`, `lighting_brighten`, `lighting_darken`
- `light_rotate_left`, `light_rotate_right`, `light_raise`, `light_lower`

## Requirements

- Python >= 3.10
- Core dependencies: `topotoolbox`, `numpy`, `matplotlib`, `scipy`, `rasterio`, `geopandas`, `shapely`, `clarabel`
- Extra dependencies for this extension: `pyvista`, `numba`

## Installation

Once published to PyPI:

```bash
pip install --upgrade pytopoviz
```

For local development from this repository:

```bash
pip install -e ".[test,docs]"
```

## Development

- Source lives in `pytopoviz/`.
- Tests live in `tests/` and use `pytest`.
- Docs use Sphinx; starter files live in `docs/`.
- Linting follows the same `pylint` configuration used in `pytopotoolbox`.

## Authors

Boris Gailleton (boris.gailleton@univ-rennes.fr)
