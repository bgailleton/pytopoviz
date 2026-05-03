"""Miscellaneous 2D matplotlib helpers.

Author: B.G.
"""

from typing import Literal

__all__ = [
    "convert_ticks_to_km",
    "add_grid_crosses",
    "add_colorbar",
    "finalize_figsize",
]


def convert_ticks_to_km(ax, axes: Literal["x", "y", "both"] = "both") -> None:
    """
    Convert axis tick labels from meters to kilometers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to modify.
    axes : {'x', 'y', 'both'}
        Which axes to convert.
    """
    from matplotlib.ticker import FuncFormatter

    def km_formatter(x, _pos):
        return f"{x/1000:.1f}"

    if axes in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(km_formatter))
        xlabel = ax.get_xlabel()
        if xlabel and "m" in xlabel.lower() and "km" not in xlabel.lower():
            ax.set_xlabel(xlabel.replace("m", "km").replace("M", "km"))

    if axes in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(km_formatter))
        ylabel = ax.get_ylabel()
        if ylabel and "m" in ylabel.lower() and "km" not in ylabel.lower():
            ax.set_ylabel(ylabel.replace("m", "km").replace("M", "km"))


def add_grid_crosses(
    ax,
    color="black",
    size=5,
    linewidth=1,
    alpha=0.47,
    include_minor: bool = True,
) -> None:
    """
    Add + signs at grid junction points defined by axis ticks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to decorate.
    color : str or tuple
        Marker color.
    size : float
        Marker size.
    linewidth : float
        Line width of crosses.
    alpha : float
        Transparency of crosses.
    """
    xticks = list(ax.get_xticks())
    yticks = list(ax.get_yticks())

    if include_minor:
        xticks += list(ax.get_xticks(minor=True))
        yticks += list(ax.get_yticks(minor=True))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xticks = [x for x in xticks if min(xlim) <= x <= max(xlim)]
    yticks = [y for y in yticks if min(ylim) <= y <= max(ylim)]

    minor_xticks = set(ax.get_xticks(minor=True)) if include_minor else set()
    minor_yticks = set(ax.get_yticks(minor=True)) if include_minor else set()

    for x in xticks:
        for y in yticks:
            is_minor = (x in minor_xticks) or (y in minor_yticks)
            marker_size = size * 0.6 if is_minor else size
            marker_alpha = alpha * 0.2 if is_minor else alpha
            ax.plot(
                x,
                y,
                "+",
                color=color,
                markersize=marker_size,
                markeredgewidth=linewidth,
                alpha=marker_alpha,
                zorder=100,
            )

def add_colorbar(ax, mappable, label=None, location="right", size="5%", pad=0.05, labelpad=5, shrink=1.0):
    """
    Attach a colorbar to an axes, sized to match the axes edge.

    Uses ``inset_axes`` so the colorbar is locked to the axes (not the figure)
    and supports fractional shrinking along its long axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to attach the colorbar to.
    mappable : matplotlib.cm.ScalarMappable
        The mappable (e.g. return value of ``imshow``) to describe.
    label : str, optional
        Colorbar label.
    location : {'right', 'left', 'top', 'bottom'}
        Which edge of the axes to place the colorbar on.
    size : str, optional
        Thickness of the colorbar as a percentage of the axes short dimension,
        e.g. ``"5%"``.
    pad : float, optional
        Padding between the axes and the colorbar in axes-fraction units.
    labelpad : float, optional
        Padding between the colorbar and its label.
    shrink : float, optional
        Fraction of the axes long dimension to use for the colorbar length.
        1.0 (default) matches the full axes edge; 0.5 gives half-length centered.

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    vertical = location in ("right", "left")
    orientation = "vertical" if vertical else "horizontal"

    long_pct = f"{shrink * 100:.4g}%"

    if vertical:
        width, height = size, long_pct
    else:
        width, height = long_pct, size

    loc_map = {"right": "center left", "left": "center right", "top": "lower center", "bottom": "upper center"}
    anchor_map = {
        "right":  (1.0 + pad, 0.0, 1.0, 1.0),
        "left":   (-pad,      0.0, 1.0, 1.0),
        "top":    (0.0, 1.0 + pad, 1.0, 1.0),
        "bottom": (0.0, -pad,      1.0, 1.0),
    }

    cax = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc_map[location],
        bbox_to_anchor=anchor_map[location],
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    cbar = ax.figure.colorbar(mappable, cax=cax, orientation=orientation)
    if label is not None:
        cbar.set_label(label, labelpad=labelpad)
    return cbar


def finalize_figsize(
    mappers,
    base_height: float = 6.0,
    base_width: float | None = None,
    min_width: float = 4.0,
    cbar_extra: float = 1.3,
) -> tuple[float, float]:
    """
    Pick a figure size based on mapper extents and colorbars.

    Parameters
    ----------
    mappers : Iterable[MapObject]
        MapObjects being plotted.
    base_height : float
        Default figure height in inches.
    base_width : float or None
        Base width (before aspect) in inches; multiplied by aspect ratio.
        Defaults to 70% of base_height when None.
    min_width : float
        Minimum figure width in inches.
    cbar_extra : float
        Additional width to reserve when a colorbar is present.
    """
    xs: list[float] = []
    ys: list[float] = []
    has_cbar = False
    for mapper in mappers:
        xmin, xmax, ymin, ymax = mapper.grid.extent
        xs.extend([xmin, xmax])
        ys.extend([ymin, ymax])
        has_cbar = has_cbar or isinstance(mapper.cbar, str)

    width = max(xs) - min(xs) if xs else 1.0
    height = max(ys) - min(ys) if ys else 1.0

    # Avoid degenerate aspect ratios
    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0

    aspect = width / height
    fig_height = base_height
    base_w = base_width if base_width is not None else 0.7 * base_height
    fig_width = max(min_width, base_w * aspect)

    if has_cbar:
        fig_width += cbar_extra

    return fig_width, fig_height
