"""Matplotlib style presets for topoviz."""

from __future__ import annotations


__all__ = [
    "apply_dark_pres_mono_style",
    "apply_color_pres_style",
    "apply_paper_style",
    "apply_bw_paper_style",
    "apply_nothing_style",
    "set_style",
    "get_style",
]

_CURRENT_STYLE: "str | None" = None


def apply_dark_pres_mono_style() -> None:
    """
    Apply dark presentation monospace style to matplotlib.

    Sets rcParams for a dark theme with monospace fonts.
    """
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = [
        "JetBrains Mono",
        "Fira Mono",
        "Consolas",
        "Menlo",
        "DejaVu Sans Mono",
        "Courier New",
    ]
    plt.rcParams["font.size"] = 15
    plt.rcParams["text.color"] = "#e6e6e6"
    plt.rcParams["axes.labelcolor"] = "#ffffff"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = True

    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlelocation"] = "left"
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 14

    plt.rcParams["figure.facecolor"] = "#0e1117"
    plt.rcParams["axes.facecolor"] = "#0e1117"
    plt.rcParams["savefig.facecolor"] = "#0e1117"
    plt.rcParams["figure.edgecolor"] = "#0e1117"
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05

    plt.rcParams["axes.edgecolor"] = "#3a3f4b"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True

    plt.rcParams["axes.grid"] = False

    plt.rcParams["xtick.color"] = "#d8dee9"
    plt.rcParams["ytick.color"] = "#d8dee9"
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2
    plt.rcParams["xtick.minor.size"] = 3.5
    plt.rcParams["ytick.minor.size"] = 3.5
    plt.rcParams["xtick.minor.width"] = 1.0
    plt.rcParams["ytick.minor.width"] = 1.0

    plt.rcParams["lines.linewidth"] = 2.6
    plt.rcParams["lines.solid_capstyle"] = "round"
    plt.rcParams["lines.solid_joinstyle"] = "round"
    plt.rcParams["lines.antialiased"] = True
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["lines.markeredgewidth"] = 0.0
    plt.rcParams["errorbar.capsize"] = 3

    plt.rcParams["patch.edgecolor"] = "#0e1117"
    plt.rcParams["patch.force_edgecolor"] = False

    plt.rcParams["image.cmap"] = "magma"
    plt.rcParams["image.interpolation"] = "antialiased"

    plt.rcParams["axes.prop_cycle"] = cycler(
        color=[
            "#4cc9f0",
            "#f72585",
            "#bde03f",
            "#fca311",
            "#9b5de5",
            "#00f5d4",
            "#ffd166",
            "#e76f51",
            "#56cfe1",
            "#ff006e",
        ]
    )

    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.facecolor"] = "none"
    plt.rcParams["legend.edgecolor"] = "none"
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 0.0
    plt.rcParams["legend.title_fontsize"] = 14
    plt.rcParams["legend.handlelength"] = 2.0
    plt.rcParams["legend.handletextpad"] = 0.6
    plt.rcParams["legend.borderaxespad"] = 0.8

    plt.rcParams["boxplot.flierprops.marker"] = "o"
    plt.rcParams["boxplot.flierprops.markerfacecolor"] = "#f72585"
    plt.rcParams["boxplot.flierprops.markeredgecolor"] = "#0e1117"
    plt.rcParams["boxplot.whiskerprops.linestyle"] = "-"

    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.constrained_layout.h_pad"] = 0.02
    plt.rcParams["figure.constrained_layout.w_pad"] = 0.02

    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def apply_paper_style() -> None:
    """
    Apply clean paper-ready style to matplotlib.

    Sets rcParams for a light theme with professional fonts.
    """
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams["font.size"] = 12
    plt.rcParams["text.color"] = "#000000"
    plt.rcParams["axes.labelcolor"] = "#000000"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = True

    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlelocation"] = "center"
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11

    plt.rcParams["figure.facecolor"] = "#ffffff"
    plt.rcParams["axes.facecolor"] = "#ffffff"
    plt.rcParams["savefig.facecolor"] = "#ffffff"
    plt.rcParams["figure.edgecolor"] = "#ffffff"
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05

    plt.rcParams["axes.edgecolor"] = "#000000"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True

    plt.rcParams["axes.grid"] = False

    plt.rcParams["xtick.color"] = "#000000"
    plt.rcParams["ytick.color"] = "#000000"
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.minor.size"] = 2.5
    plt.rcParams["ytick.minor.size"] = 2.5
    plt.rcParams["xtick.minor.width"] = 0.6
    plt.rcParams["ytick.minor.width"] = 0.6

    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["lines.solid_capstyle"] = "round"
    plt.rcParams["lines.solid_joinstyle"] = "round"
    plt.rcParams["lines.antialiased"] = True
    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["lines.markeredgewidth"] = 0.5
    plt.rcParams["errorbar.capsize"] = 2

    plt.rcParams["patch.edgecolor"] = "#000000"
    plt.rcParams["patch.force_edgecolor"] = False

    plt.rcParams["image.cmap"] = "viridis"
    plt.rcParams["image.interpolation"] = "antialiased"

    plt.rcParams["axes.prop_cycle"] = cycler(
        color=[
            "#0173B2",
            "#DE8F05",
            "#029E73",
            "#CC78BC",
            "#CA9161",
            "#949494",
            "#ECE133",
            "#56B4E9",
            "#F0E442",
            "#D55E00",
        ]
    )

    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.facecolor"] = "#ffffff"
    plt.rcParams["legend.edgecolor"] = "#000000"
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1.0
    plt.rcParams["legend.title_fontsize"] = 10
    plt.rcParams["legend.handlelength"] = 2.0
    plt.rcParams["legend.handletextpad"] = 0.5
    plt.rcParams["legend.borderaxespad"] = 0.5

    plt.rcParams["boxplot.flierprops.marker"] = "o"
    plt.rcParams["boxplot.flierprops.markerfacecolor"] = "#000000"
    plt.rcParams["boxplot.flierprops.markeredgecolor"] = "#000000"
    plt.rcParams["boxplot.whiskerprops.linestyle"] = "-"

    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.constrained_layout.h_pad"] = 0.04
    plt.rcParams["figure.constrained_layout.w_pad"] = 0.04

    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def apply_color_pres_style() -> None:
    """
    Apply a light, high-contrast presentation style.

    Uses a bright background with saturated accents for projection-friendly slides.
    """
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Inter",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams["font.size"] = 15
    plt.rcParams["text.color"] = "#0b0f1a"
    plt.rcParams["axes.labelcolor"] = "#0b0f1a"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = True

    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlelocation"] = "left"
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 14

    plt.rcParams["figure.facecolor"] = "#f6f7fb"
    plt.rcParams["axes.facecolor"] = "#f6f7fb"
    plt.rcParams["savefig.facecolor"] = "#f6f7fb"
    plt.rcParams["figure.edgecolor"] = "#f6f7fb"
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05

    plt.rcParams["axes.edgecolor"] = "#b7c0d8"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True

    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#d8deed"
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["grid.alpha"] = 0.8
    plt.rcParams["grid.linestyle"] = "-"

    plt.rcParams["xtick.color"] = "#0b0f1a"
    plt.rcParams["ytick.color"] = "#0b0f1a"
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["xtick.minor.width"] = 0.8
    plt.rcParams["ytick.minor.width"] = 0.8

    plt.rcParams["lines.linewidth"] = 2.4
    plt.rcParams["lines.solid_capstyle"] = "round"
    plt.rcParams["lines.solid_joinstyle"] = "round"
    plt.rcParams["lines.antialiased"] = True
    plt.rcParams["lines.markersize"] = 7
    plt.rcParams["lines.markeredgewidth"] = 0.0
    plt.rcParams["errorbar.capsize"] = 3

    plt.rcParams["patch.edgecolor"] = "#0b0f1a"
    plt.rcParams["patch.force_edgecolor"] = False

    # Use a widely available, vivid colormap to avoid dependency issues.
    plt.rcParams["image.cmap"] = "turbo"
    plt.rcParams["image.interpolation"] = "antialiased"

    plt.rcParams["axes.prop_cycle"] = cycler(
        color=[
            "#0066ff",
            "#ff3366",
            "#00b386",
            "#ffb000",
            "#7a4fff",
            "#ff6f00",
            "#0096c7",
            "#d7263d",
            "#38b000",
            "#f9844a",
        ]
    )

    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.facecolor"] = "none"
    plt.rcParams["legend.edgecolor"] = "none"
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 0.0
    plt.rcParams["legend.title_fontsize"] = 14
    plt.rcParams["legend.handlelength"] = 2.0
    plt.rcParams["legend.handletextpad"] = 0.6
    plt.rcParams["legend.borderaxespad"] = 0.8

    plt.rcParams["boxplot.flierprops.marker"] = "o"
    plt.rcParams["boxplot.flierprops.markerfacecolor"] = "#ff3366"
    plt.rcParams["boxplot.flierprops.markeredgecolor"] = "#0b0f1a"
    plt.rcParams["boxplot.whiskerprops.linestyle"] = "-"

    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.constrained_layout.h_pad"] = 0.04
    plt.rcParams["figure.constrained_layout.w_pad"] = 0.04

    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def apply_bw_paper_style() -> None:
    """
    Apply monochrome paper style for grayscale/print outputs.
    """
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Georgia",
        "DejaVu Serif",
        "Liberation Serif",
    ]
    plt.rcParams["font.size"] = 11
    plt.rcParams["text.color"] = "#000000"
    plt.rcParams["axes.labelcolor"] = "#000000"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["axes.unicode_minus"] = True

    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlelocation"] = "center"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10

    plt.rcParams["figure.facecolor"] = "#ffffff"
    plt.rcParams["axes.facecolor"] = "#ffffff"
    plt.rcParams["savefig.facecolor"] = "#ffffff"
    plt.rcParams["figure.edgecolor"] = "#ffffff"
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05

    plt.rcParams["axes.edgecolor"] = "#000000"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True

    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#cccccc"
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["grid.alpha"] = 0.8
    plt.rcParams["grid.linestyle"] = "--"

    plt.rcParams["xtick.color"] = "#000000"
    plt.rcParams["ytick.color"] = "#000000"
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.minor.size"] = 2
    plt.rcParams["ytick.minor.size"] = 2
    plt.rcParams["xtick.minor.width"] = 0.6
    plt.rcParams["ytick.minor.width"] = 0.6

    plt.rcParams["lines.linewidth"] = 1.4
    plt.rcParams["lines.solid_capstyle"] = "butt"
    plt.rcParams["lines.solid_joinstyle"] = "miter"
    plt.rcParams["lines.antialiased"] = True
    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["lines.markeredgewidth"] = 0.6
    plt.rcParams["errorbar.capsize"] = 2

    plt.rcParams["patch.edgecolor"] = "#000000"
    plt.rcParams["patch.force_edgecolor"] = True

    plt.rcParams["image.cmap"] = "Greys"
    plt.rcParams["image.interpolation"] = "nearest"

    plt.rcParams["axes.prop_cycle"] = cycler(color=["#111111"])

    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.facecolor"] = "#ffffff"
    plt.rcParams["legend.edgecolor"] = "#000000"
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1.0
    plt.rcParams["legend.title_fontsize"] = 10
    plt.rcParams["legend.handlelength"] = 1.6
    plt.rcParams["legend.handletextpad"] = 0.4
    plt.rcParams["legend.borderaxespad"] = 0.4

    plt.rcParams["boxplot.flierprops.marker"] = "o"
    plt.rcParams["boxplot.flierprops.markerfacecolor"] = "#888888"
    plt.rcParams["boxplot.flierprops.markeredgecolor"] = "#000000"
    plt.rcParams["boxplot.whiskerprops.linestyle"] = "-"

    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.constrained_layout.h_pad"] = 0.02
    plt.rcParams["figure.constrained_layout.w_pad"] = 0.02

    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def apply_nothing_style() -> None:
    """
    Apply a bare style: transparent background, no axes, no text.

    Intended for clean image export or compositing where only the data should be visible.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["savefig.facecolor"] = "none"
    plt.rcParams["figure.edgecolor"] = "none"
    plt.rcParams["savefig.transparent"] = True

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.bottom"] = False

    plt.rcParams["xtick.bottom"] = False
    plt.rcParams["xtick.labelbottom"] = False
    plt.rcParams["ytick.left"] = False
    plt.rcParams["ytick.labelleft"] = False

    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = False

    plt.rcParams["image.interpolation"] = "antialiased"
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.0


def _builtin_style_lookup() -> dict[str, str]:
    """Return a lowercase lookup for matplotlib's available styles."""
    import matplotlib.pyplot as plt

    return {name.lower(): name for name in plt.style.available}


def set_style(style: str) -> None:
    """
    Apply a named matplotlib style preset.

    Parameters
    ----------
    style:
        Name of the style to apply. Custom values: ``"dark_pres_mono"``,
        ``"color_pres"``, ``"paper"``, or ``"bw_paper"`` (case-insensitive). If
        the name matches a matplotlib native style (e.g. ``\"seaborn\"``,
        ``\"ggplot\"``, ``\"classic\"``), that style is applied via
        ``plt.style.use``.
    """
    import matplotlib.pyplot as plt

    normalized = style.strip().lower()
    styles = {
        "dark_pres_mono": apply_dark_pres_mono_style,
        "color_pres": apply_color_pres_style,
        "paper": apply_paper_style,
        "bw_paper": apply_bw_paper_style,
        "nothing": apply_nothing_style,
    }

    global _CURRENT_STYLE
    if normalized in styles:
        styles[normalized]()
        _CURRENT_STYLE = normalized
        return

    builtin_styles = _builtin_style_lookup()
    if normalized in builtin_styles:
        plt.style.use(builtin_styles[normalized])
        _CURRENT_STYLE = builtin_styles[normalized]
        return

    available = ", ".join(sorted(set(styles) | set(builtin_styles)))
    raise ValueError(f"Unknown style '{style}'. Available styles: {available}")


def get_style() -> str | None:
    """Return the last style applied via set_style, if any."""
    return _CURRENT_STYLE
