"""Quick matplotlib helpers for MapObject/GridObject.

Author: B.G.
"""

from typing import Any, Iterable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from topotoolbox import GridObject

from .helper2d import add_colorbar, add_grid_crosses, convert_ticks_to_km, finalize_figsize
from .map_object import MapObject
from .processing import expand_plottables, is_plottable

__all__ = ["quickmap", "Fig2DObject"]


class Fig2DObject:
    """
    Wrapper around a matplotlib figure and axes to manage 2D map layers.
    Stores plotted artists keyed by MapObject for convenient access.
    """

    def __init__(self, **subplots_kwargs):
        fig, axes = plt.subplots(**subplots_kwargs)
        if isinstance(axes, np.ndarray):
            axes_list = axes.ravel().tolist()
        elif axes is None:
            axes_list = []
        else:
            axes_list = [axes]

        self.fig = fig
        self.axes = axes_list
        self.images: dict[MapObject, Any] = {}
        self.colorbars: dict[MapObject, Any] = {}
        self.base_maps: list[MapObject] = []

    @property
    def ax(self):
        if len(self.axes) != 1:
            raise ValueError("Fig2DObject.ax is only available when there is a single axis.")
        return self.axes[0]

    def save(self, **kwargs):
        """Forward to matplotlib's savefig."""
        return self.fig.savefig(**kwargs)

    def add_maps(self, axis, *maps: Union[MapObject, GridObject]):
        """Add one or more MapObjects to the provided axis."""
        if not maps:
            raise ValueError("Provide at least one MapObject or GridObject to add.")

        map_list: Tuple[MapObject, ...] = tuple(_ensure_map(m) for m in maps)
        self.base_maps.extend(map_list)
        plot_list: list[MapObject] = []
        for mapper in map_list:
            # Expand processors so derived layers (e.g., hillshade) get plotted too.
            plot_list.extend(expand_plottables(mapper, mode="2d"))

        for mapper in plot_list:
            if not is_plottable(mapper):
                continue
            im = axis.imshow(
                mapper.value,
                cmap=mapper.cmap,
                alpha=mapper.alpha,
                extent=mapper.grid.extent,
            )
            self.images[mapper] = im
            if isinstance(mapper.cbar, str):
                cbar = add_colorbar(axis, im, label=mapper.cbar)
                self.colorbars[mapper] = cbar
        return axis


def _ensure_map(obj: Union[MapObject, GridObject]) -> MapObject:
    """Convert GridObject to MapObject when needed."""
    if isinstance(obj, MapObject):
        return obj
    if isinstance(obj, GridObject):
        return MapObject(obj)
    raise TypeError("quickmap accepts MapObject or GridObject instances.")


def quickmap(*maps: Union[MapObject, GridObject]):
    """Plot one or more MapObjects on a single axis, in order.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    if len(maps) == 0:
        raise ValueError("Provide at least one MapObject or GridObject.")

    map_list: Tuple[MapObject, ...] = tuple(_ensure_map(m) for m in maps)

    figsize = finalize_figsize(map_list)
    fig_obj = Fig2DObject(figsize=figsize, constrained_layout=True, layout=None)
    fig_obj.fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.02, hspace=0.02)

    ax = fig_obj.ax
    fig_obj.add_maps(ax, *map_list)

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    convert_ticks_to_km(ax)
    add_grid_crosses(ax)
    return fig_obj.fig, ax
