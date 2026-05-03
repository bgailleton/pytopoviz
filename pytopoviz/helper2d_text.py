"""Text property helpers for matplotlib figures and axes."""

from __future__ import annotations

from typing import Union

import matplotlib.axes as maxes
import matplotlib.colorbar as mcolorbar
import matplotlib.figure as mfigure
import matplotlib.text as mtext

__all__ = ["set_font", "set_font_size", "set_font_style", "set_font_color"]

_WHICH_ALL = ("title", "label", "tick", "legend", "colorbar")


def _resolve_target(target):
    if isinstance(target, mfigure.Figure):
        return target, target.axes
    if isinstance(target, maxes.Axes):
        return target.figure, [target]
    raise TypeError(f"target must be a Figure or Axes, got {type(target)}")


def _find_colorbar_axes(fig):
    return {cb.ax for cb in fig.findobj(mcolorbar.Colorbar)}


def _resolve_which(which):
    if which == "all":
        return set(_WHICH_ALL)
    if isinstance(which, str):
        return {which}
    return set(which)


def _collect(axes, cbar_axes, categories):
    texts = []
    for ax in axes:
        if ax in cbar_axes:
            if "colorbar" in categories:
                texts.extend(ax.get_xticklabels())
                texts.extend(ax.get_yticklabels())
                texts.append(ax.xaxis.label)
                texts.append(ax.yaxis.label)
        else:
            if "title" in categories:
                texts.append(ax.title)
            if "label" in categories:
                texts.extend([ax.xaxis.label, ax.yaxis.label])
            if "tick" in categories:
                texts.extend(ax.get_xticklabels())
                texts.extend(ax.get_yticklabels())
                texts.extend(ax.get_xticklabels(minor=True))
                texts.extend(ax.get_yticklabels(minor=True))
            if "legend" in categories:
                legend = ax.get_legend()
                if legend is not None:
                    texts.extend(legend.get_texts())
    return texts


# rcParam keys for each category, per function
_COLOR_RCPARAMS = {
    "title":  ["text.color"],
    "label":  ["axes.labelcolor"],
    "tick":   ["xtick.color", "ytick.color"],
    "legend": ["text.color"],
    # colorbar has no dedicated rcParam — skipped
}

_SIZE_RCPARAMS = {
    "title":   ["axes.titlesize"],
    "label":   ["axes.labelsize"],
    "tick":    ["xtick.labelsize", "ytick.labelsize"],
    "legend":  ["legend.fontsize"],
    # colorbar has no dedicated rcParam — skipped
}

_STYLE_RCPARAMS = {
    "title":   [("axes.titleweight", "weight")],
    "label":   [("axes.labelweight", "weight")],
    "tick":    [],
    "legend":  [],
    # font.style / font.weight are global; mapped below
}


def set_font_color(
    target: Union[mfigure.Figure, maxes.Axes, None],
    color,
    which: Union[str, list[str]] = "all",
) -> None:
    """
    Set font color for selected text categories.

    Parameters
    ----------
    target : Figure, Axes, or None
        When ``None``, sets rcParams globally for future figures.
    color : any matplotlib color
        e.g. ``"white"``, ``"#ff0000"``, ``(1, 0, 0)``, ``(1, 0, 0, 0.5)``.
    which : str or list of str
        One or more of: ``"title"``, ``"label"``, ``"tick"``, ``"legend"``,
        ``"colorbar"``, ``"all"``.
    """
    import matplotlib as mpl

    categories = _resolve_which(which)

    if target is None:
        for cat, keys in _COLOR_RCPARAMS.items():
            if cat in categories:
                for key in keys:
                    mpl.rcParams[key] = color
        return

    fig, axes = _resolve_target(target)
    fig.canvas.draw()
    cbar_axes = _find_colorbar_axes(fig)
    for t in _collect(axes, cbar_axes, categories):
        t.set_color(color)


def set_font(
    target: Union[mfigure.Figure, None],
    family: str,
) -> None:
    """
    Set the font family for every Text object in a figure, or globally.

    Parameters
    ----------
    target : Figure or None
        When ``None``, sets ``font.family`` rcParam globally for future figures.
    family : str
        Font family name (e.g. ``"Arial"``) or path to a ``.ttf``/``.otf`` file.
        When a file path is given, the font is registered and its family name used.
    """
    import os
    import matplotlib as mpl
    from matplotlib import font_manager

    if os.path.isfile(family):
        font_manager.fontManager.addfont(family)
        family = font_manager.FontProperties(fname=family).get_name()

    if target is None:
        mpl.rcParams["font.family"] = family
        return

    target.canvas.draw()
    for obj in target.findobj(mtext.Text):
        obj.set_fontfamily(family)


def set_font_size(
    target: Union[mfigure.Figure, maxes.Axes, None],
    size: float,
    which: Union[str, list[str]] = "all",
) -> None:
    """
    Set font size for selected text categories.

    Parameters
    ----------
    target : Figure, Axes, or None
        When ``None``, sets rcParams globally for future figures.
    size : float
        Font size in points.
    which : str or list of str
        One or more of: ``"title"``, ``"label"``, ``"tick"``, ``"legend"``,
        ``"colorbar"``, ``"all"``.
    """
    import matplotlib as mpl

    categories = _resolve_which(which)

    if target is None:
        for cat, keys in _SIZE_RCPARAMS.items():
            if cat in categories:
                for key in keys:
                    mpl.rcParams[key] = size
        return

    fig, axes = _resolve_target(target)
    fig.canvas.draw()
    cbar_axes = _find_colorbar_axes(fig)
    for t in _collect(axes, cbar_axes, categories):
        t.set_fontsize(size)


def set_font_style(
    target: Union[mfigure.Figure, maxes.Axes, None],
    style: str,
    which: Union[str, list[str]] = "all",
) -> None:
    """
    Set font style and weight for selected text categories.

    Parameters
    ----------
    target : Figure, Axes, or None
        When ``None``, sets rcParams globally for future figures.
    style : {'normal', 'italic', 'bold', 'bold italic'}
    which : str or list of str
        One or more of: ``"title"``, ``"label"``, ``"tick"``, ``"legend"``,
        ``"colorbar"``, ``"all"``.
    """
    import matplotlib as mpl

    _STYLES = {
        "normal":      ("normal", "normal"),
        "italic":      ("italic", "normal"),
        "bold":        ("normal", "bold"),
        "bold italic": ("italic", "bold"),
    }
    if style not in _STYLES:
        raise ValueError(f"style must be one of {list(_STYLES)}; got '{style}'")

    fontstyle, fontweight = _STYLES[style]
    categories = _resolve_which(which)

    if target is None:
        if "title" in categories:
            mpl.rcParams["axes.titleweight"] = fontweight
        if "label" in categories:
            mpl.rcParams["axes.labelweight"] = fontweight
        # font.style and font.weight are global — apply when "all" or no specific axes cat
        if categories >= {"title", "label", "tick", "legend"}:
            mpl.rcParams["font.style"] = fontstyle
            mpl.rcParams["font.weight"] = fontweight
        return

    fig, axes = _resolve_target(target)
    fig.canvas.draw()
    cbar_axes = _find_colorbar_axes(fig)
    for t in _collect(axes, cbar_axes, categories):
        t.set_style(fontstyle)
        t.set_weight(fontweight)
