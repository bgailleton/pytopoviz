import numpy as np

from topotoolbox import GridObject

from pytopoviz import MapObject, expand_plottables
from pytopoviz.legacy.masknan import nan_above, nan_below


def test_map_object_defaults_and_nan_handling():
    grid = GridObject()
    grid.z = np.array([[1.0, 2.0], [np.nan, 4.0]])

    mapper = MapObject(grid, cmap="viridis", alpha=0.5, cbar="elevation")

    assert mapper.grid is grid
    assert mapper.cmap == "viridis"
    assert mapper.alpha == 0.5
    assert mapper.cbar == "elevation"
    assert mapper.value.dtype == np.float32
    assert mapper.vmin == 1.0
    assert mapper.vmax == 4.0
    assert np.isnan(mapper.value[1, 0])


def test_set_nan_filters_via_processor():
    grid = GridObject()
    grid.z = np.array([[0.0, 5.0], [10.0, 15.0]])

    mapper = MapObject(grid)
    processor = nan_below(2.5)
    mapper.processors.append(processor)
    mapper.value[mapper.value == 5.0] = np.nan

    processed = expand_plottables(mapper)[0]

    assert np.isnan(processed.value[0, 0])
    assert np.isnan(processed.value[0, 1])
    assert np.isnan(processed.value[1, 1])
    assert processed.value[1, 0] == np.float32(10.0)


def test_processor_params_are_editable():
    grid = GridObject()
    grid.z = np.array([[1.0, 2.0]])

    mapper = MapObject(grid)
    proc = nan_above(5.0)
    proc.threshold = 1.5  # adjust after creation
    mapper.processors.append(proc)

    processed = expand_plottables(mapper)[0]
    assert np.isnan(processed.value[0, 1])


def test_getters_setters_refresh_values():
    grid = GridObject()
    grid.z = np.array([[1.0, 2.0], [3.0, 4.0]])

    mapper = MapObject(grid)
    mapper.cmap = "gray"
    mapper.alpha = 0.25
    mapper.vmin = 0.0
    mapper.vmax = 5.0
    mapper.cbar = "label"
    mapper.value = [[np.nan, 1.0], [2.0, 3.0]]

    assert mapper.cmap == "gray"
    assert mapper.alpha == 0.25
    assert mapper.cbar == "label"
    assert mapper.vmin == 1.0
    assert mapper.vmax == 3.0
    assert np.isnan(mapper.value[0, 0])

    new_grid = GridObject()
    new_grid.z = np.array([[5.0, 6.0]])
    mapper.grid = new_grid

    assert mapper.grid is new_grid
    assert mapper.vmin == 5.0
    assert mapper.vmax == 6.0


def test_map_object_name_and_hashability():
    grid = GridObject()
    grid.z = np.array([[1.0]])

    mapper = MapObject(grid)
    assert isinstance(mapper.name, str)
    assert len(mapper.name) == 8

    store = {mapper: "stored"}
    assert store[mapper] == "stored"
