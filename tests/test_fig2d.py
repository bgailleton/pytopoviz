import matplotlib

# Use non-interactive backend for tests
matplotlib.use("Agg")

import pytest
import numpy as np

from topotoolbox import GridObject

from pytopoviz import Fig2DObject, MapObject, quickmap
from pytopoviz.legacy.masknan import nan_above
from pytopoviz.legacy.shading2d import hillshade_processor
from pytopoviz.legacy.processing import ProcessingFunction


def _grid_with_values(values):
    grid = GridObject()
    grid.z = np.array(values, dtype=np.float32)
    return grid


def test_quickmap_single_and_cbar_label():
    grid = _grid_with_values([[1, 2], [3, 4]])
    mapper = MapObject(grid, cbar="elev")

    fig, ax = quickmap(mapper)

    assert fig is not None
    assert ax is not None
    # Single axis + colorbar
    assert len(fig.axes) == 2  # imshow axis + colorbar axis
    assert fig.axes[1].get_label() == "<colorbar>"
    assert fig.axes[1].get_ylabel() == "elev"
    # Extent should follow GridObject.extent
    im = fig.axes[0].images[0]
    assert tuple(im.get_extent()) == grid.extent


def test_quickmap_stacks_multiple_maps_on_same_axis():
    grid1 = _grid_with_values([[0, 1]])
    grid2 = _grid_with_values([[1, 0]])

    mapper1 = MapObject(grid1, cbar="one")
    mapper2 = MapObject(grid2, cmap="gray", alpha=0.4)

    fig, ax = quickmap(mapper1, mapper2)

    # Both images plotted on same axis
    assert len(ax.images) == 2
    # Only one colorbar attached (from mapper1)
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "one"


def test_quickmap_labels_ticks_and_crosses():
    grid = _grid_with_values([[0, 1], [2, 3]])
    mapper = MapObject(grid, cbar="elev")

    fig, ax = quickmap(mapper)

    assert ax.get_xlabel() == "Easting (km)"
    assert ax.get_ylabel() == "Northing (km)"

    # Tick formatter should convert meters to km (1000 -> 1.0)
    x_formatter = ax.xaxis.get_major_formatter()
    assert x_formatter(1000) == "1.0"

    # Cross markers added
    assert len(ax.lines) > 0


def test_fig2d_object_adds_and_tracks_maps():
    grid = _grid_with_values([[1, 2], [3, 4]])
    mapper = MapObject(grid, cbar="cb")

    fig_obj = Fig2DObject(figsize=(2, 2))
    fig_obj.add_maps(fig_obj.ax, mapper)

    assert mapper in fig_obj.images
    assert mapper in fig_obj.colorbars
    assert fig_obj.colorbars[mapper].ax.get_ylabel() == "cb"


def test_fig2d_object_ax_property_only_single_axis():
    fig_obj = Fig2DObject(nrows=1, ncols=2)
    with pytest.raises(ValueError):
        _ = fig_obj.ax


def test_quickmap_applies_processors_and_collects_outputs():
    grid = _grid_with_values([[1, 2], [3, 4]])
    mapper = MapObject(grid, cbar="base")
    mapper.processors.append(nan_above(3))
    mapper.processors.append(hillshade_processor())

    fig, ax = quickmap(mapper)

    assert len(ax.images) == 2  # base + hillshade
    cmap_names = [im.get_cmap().name for im in ax.images]
    assert "gray" in cmap_names
    # colorbar only for base
    assert len(fig.axes) == 2


def test_processors_run_in_order_and_spawned_maps_follow():
    grid = _grid_with_values([[0, 0], [0, 0]])
    mapper = MapObject(grid)

    def add_one(_: ProcessingFunction, m: MapObject):
        m.value = m.value + 1

    def spawn_child(_: ProcessingFunction, m: MapObject):
        child = MapObject(m.grid, cmap="gray", alpha=0.5)
        child.value = m.value * 2

        def add_five(_: ProcessingFunction, cm: MapObject):
            cm.value = cm.value + 5

        child.processors.append(ProcessingFunction("add_five", add_five, recursive=False))
        return child

    mapper.processors.append(ProcessingFunction("add_one", add_one, recursive=False))
    mapper.processors.append(ProcessingFunction("spawn_child", spawn_child, recursive=False))

    fig, ax = quickmap(mapper)

    assert len(ax.images) == 2  # base then child
    base_data = ax.images[0].get_array()
    child_data = ax.images[1].get_array()
    np.testing.assert_array_equal(base_data, np.ones_like(base_data))
    np.testing.assert_array_equal(child_data, np.full_like(child_data, 7))
