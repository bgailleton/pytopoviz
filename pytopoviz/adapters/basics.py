"""Small building blocks for workflows (node graphs): numbers, grid math,
masks, statistics and grid information.

Grids are ``georaster`` (any library grid converts into it at the port, and
back out of it where a library process reads one): one block serves every
library. A grid out has the lattice of the grid in; masks are 0 / 1 grids,
NaN where the grid read is nodata. Two grids combined must share a lattice
(shape, cell size, corner, CRS), else the block fails naming both.

numpy only: no optional library, so this module always loads.

Author: B.G.
"""

from __future__ import annotations

import numpy as np

from ..core import Output, Param, Port, process
from ..georaster import GeoRaster

_P = "pytopoviz.basic."

# ---- numbers ----------------------------------------------------------------

_BINARY = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide,
    "pow": np.power, "min": np.fmin, "max": np.fmax,
}
_BINARY_LABELS = ["a + b", "a − b", "a × b", "a ÷ b", "a ^ b", "min(a, b)", "max(a, b)"]
_UNARY = {
    "abs": np.abs, "neg": np.negative, "sqrt": np.sqrt, "log": np.log,
    "log10": np.log10, "exp": np.exp,
}
_UNARY_LABELS = ["|x|", "−x", "√x", "ln x", "log₁₀ x", "eˣ"]
_COMPARE = {
    "<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
    "==": np.equal, "!=": np.not_equal,
}
_COMPARE_LABELS = ["a < b", "a ≤ b", "a > b", "a ≥ b", "a = b", "a ≠ b"]


def _op_param(table, labels, default, doc):
    return Param("op", "string", default=default, choices=list(table), choice_labels=labels,
                 label="Operation", doc=doc)


@process(
    id=_P + "float",
    category="Numbers",
    label="Number",
    params=[Param("value", "float", default=0.0, label="Value")],
    outputs=[Output("number", "float", label="Value")],
)
def number(value=0.0):
    """A number."""
    return float(value)


@process(
    id=_P + "int",
    category="Numbers",
    label="Integer",
    params=[Param("value", "int", default=0, label="Value")],
    outputs=[Output("number", "int", label="Value")],
)
def integer(value=0):
    """An integer."""
    return int(value)


@process(
    id=_P + "math",
    category="Numbers",
    label="Math",
    params=[Param("a", "float", default=0.0, label="a"),
            Param("b", "float", default=0.0, label="b"),
            _op_param(_BINARY, _BINARY_LABELS, "+", "What is done with a and b.")],
    outputs=[Output("value", "float", label="Result")],
)
def math(a=0.0, b=0.0, op="+"):
    """a (op) b."""
    with np.errstate(all="ignore"):
        return float(_BINARY[op](float(a), float(b)))


@process(
    id=_P + "unary",
    category="Numbers",
    label="Function",
    params=[Param("x", "float", default=0.0, label="x"),
            _op_param(_UNARY, _UNARY_LABELS, "abs", "The function of x.")],
    outputs=[Output("value", "float", label="Result")],
)
def unary(x=0.0, op="abs"):
    """f(x)."""
    with np.errstate(all="ignore"):
        return float(_UNARY[op](float(x)))


@process(
    id=_P + "compare",
    category="Numbers",
    label="Compare",
    params=[Param("a", "float", default=0.0, label="a"),
            Param("b", "float", default=0.0, label="b"),
            _op_param(_COMPARE, _COMPARE_LABELS, "<", "The comparison.")],
    outputs=[Output("value", "bool", label="Result")],
)
def compare(a=0.0, b=0.0, op="<"):
    """a (op) b, true or false."""
    return bool(_COMPARE[op](float(a), float(b)))


@process(
    id=_P + "switch",
    category="Numbers",
    label="Switch",
    params=[Param("condition", "bool", default=True, label="Condition"),
            Param("a", "float", default=0.0, label="If true"),
            Param("b", "float", default=0.0, label="If false")],
    outputs=[Output("value", "float", label="Result")],
)
def switch(condition=True, a=0.0, b=0.0):
    """a when the condition holds, else b."""
    return float(a if condition else b)


# ---- grids ------------------------------------------------------------------

def _like(grid: GeoRaster, z) -> GeoRaster:
    """A grid of ``grid``'s lattice holding ``z``."""
    return GeoRaster.from_meta(np.asarray(z, dtype=float), grid.meta())


def _same(a: GeoRaster, b: GeoRaster, what: str) -> None:
    if not a.same_grid(b.meta()):
        raise ValueError(
            f"{what}: the grids are not on the same lattice "
            f"({a.n_cols} x {a.n_rows} cells of {a.cell_size:g} at ({a.x_min:g}, {a.y_min:g}) "
            f"vs {b.n_cols} x {b.n_rows} cells of {b.cell_size:g} at ({b.x_min:g}, {b.y_min:g}))")


def _mask_values(mask: GeoRaster) -> np.ndarray:
    """True where ``mask`` is set (non-zero, not nodata)."""
    m = mask.z
    return np.isfinite(m) & (m != 0)


@process(
    id=_P + "grid_math",
    category="Grid maths",
    label="Grid math (grid, grid)",
    inputs=[Port("a", "georaster", label="a"), Port("b", "georaster", label="b")],
    params=[_op_param(_BINARY, _BINARY_LABELS, "+", "What is done with a and b, cell by cell.")],
    outputs=[Output("result", "georaster", label="Result")],
)
def grid_math(a, b, op="+"):
    """a (op) b, cell by cell; both on the same lattice. Nodata in either is
    nodata out (min / max: the other value)."""
    _same(a, b, "grid math")
    with np.errstate(all="ignore"):
        return _like(a, _BINARY[op](a.z, b.z))


@process(
    id=_P + "grid_scalar_math",
    category="Grid maths",
    label="Grid math (grid, number)",
    inputs=[Port("a", "georaster", label="Grid")],
    params=[Param("b", "float", default=0.0, label="Number"),
            _op_param(_BINARY, _BINARY_LABELS, "*", "What is done with each cell and the number."),
            Param("swap", "bool", default=False, label="Number first",
                  doc="Number (op) grid instead of grid (op) number.")],
    outputs=[Output("result", "georaster", label="Result")],
)
def grid_scalar_math(a, b=0.0, op="*", swap=False):
    """Each cell (op) the number, or the number (op) each cell."""
    b = float(b)
    with np.errstate(all="ignore"):
        z = _BINARY[op](b, a.z) if swap else _BINARY[op](a.z, b)
    return _like(a, z)


@process(
    id=_P + "grid_unary",
    category="Grid maths",
    label="Grid function",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[_op_param(_UNARY, _UNARY_LABELS, "abs", "The function of each cell.")],
    outputs=[Output("result", "georaster", label="Result")],
)
def grid_unary(grid, op="abs"):
    """f(cell), cell by cell (out of the function's domain: nodata)."""
    with np.errstate(all="ignore"):
        return _like(grid, _UNARY[op](grid.z))


@process(
    id=_P + "threshold",
    category="Grid maths",
    label="Threshold",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[Param("value", "float", default=0.0, label="Value"),
            Param("op", "string", default=">", choices=list(_COMPARE),
                  choice_labels=[s.replace("a", "cell").replace("b", "value") for s in _COMPARE_LABELS],
                  label="Keep cells where")],
    outputs=[Output("mask", "georaster", label="Mask")],
)
def threshold(grid, value=0.0, op=">"):
    """A 0 / 1 mask: 1 where the cell compares true with the value; nodata
    where the grid is."""
    z = grid.z
    with np.errstate(invalid="ignore"):
        out = _COMPARE[op](z, float(value)).astype(float)
    out[~np.isfinite(z)] = np.nan
    return _like(grid, out)


@process(
    id=_P + "apply_mask",
    category="Grid maths",
    label="Apply mask",
    inputs=[Port("grid", "georaster", label="Grid"), Port("mask", "georaster", label="Mask")],
    params=[Param("invert", "bool", default=False, label="Invert",
                  doc="Keep the cells outside the mask instead.")],
    outputs=[Output("result", "georaster", label="Result")],
)
def apply_mask(grid, mask, invert=False):
    """The grid where the mask is set (non-zero), nodata elsewhere (inverted:
    the other way round; a nodata mask cell is never kept)."""
    _same(grid, mask, "apply mask")
    keep = _mask_values(mask)
    if invert:
        keep = np.isfinite(mask.z) & ~keep
    return _like(grid, np.where(keep, grid.z, np.nan))


@process(
    id=_P + "where",
    category="Grid maths",
    label="Where",
    inputs=[Port("mask", "georaster", label="Mask"), Port("a", "georaster", label="If set"),
            Port("b", "georaster", label="Else")],
    outputs=[Output("result", "georaster", label="Result")],
)
def where(mask, a, b):
    """a where the mask is set (non-zero), b elsewhere; nodata where the mask
    is."""
    _same(mask, a, "where")
    _same(mask, b, "where")
    z = np.where(_mask_values(mask), a.z, b.z)
    z[~np.isfinite(mask.z)] = np.nan
    return _like(mask, z)


@process(
    id=_P + "clamp",
    category="Grid maths",
    label="Clamp",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[Param("lo", "float", default=0.0, label="Lowest"),
            Param("hi", "float", default=1.0, label="Highest")],
    outputs=[Output("result", "georaster", label="Result")],
)
def clamp(grid, lo=0.0, hi=1.0):
    """Every cell within [lo, hi]."""
    if lo > hi:
        raise ValueError(f"clamp: lowest {lo:g} is above highest {hi:g}")
    return _like(grid, np.clip(grid.z, float(lo), float(hi)))


@process(
    id=_P + "rescale",
    category="Grid maths",
    label="Rescale",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[Param("lo", "float", default=0.0, label="New lowest"),
            Param("hi", "float", default=1.0, label="New highest")],
    outputs=[Output("result", "georaster", label="Result")],
)
def rescale(grid, lo=0.0, hi=1.0):
    """The grid's values mapped linearly from [its min, its max] to [lo, hi]
    (a flat grid: all lo)."""
    z = grid.z
    if not np.isfinite(z).any():
        return _like(grid, z)
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if zmax == zmin:
        return _like(grid, np.where(np.isfinite(z), float(lo), np.nan))
    return _like(grid, float(lo) + (z - zmin) * (float(hi) - float(lo)) / (zmax - zmin))


@process(
    id=_P + "fill_nodata",
    category="Grid maths",
    label="Fill nodata",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[Param("value", "float", default=0.0, label="Value")],
    outputs=[Output("result", "georaster", label="Result")],
)
def fill_nodata(grid, value=0.0):
    """Nodata cells set to the value."""
    return _like(grid, np.where(np.isfinite(grid.z), grid.z, float(value)))


@process(
    id=_P + "set_nodata",
    category="Grid maths",
    label="Value to nodata",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[Param("value", "float", default=0.0, label="Value")],
    outputs=[Output("result", "georaster", label="Result")],
)
def set_nodata(grid, value=0.0):
    """Cells equal to the value become nodata."""
    return _like(grid, np.where(grid.z == float(value), np.nan, grid.z))


# ---- statistics -> number ---------------------------------------------------

_STATS = {
    "min": np.nanmin, "max": np.nanmax, "mean": np.nanmean, "median": np.nanmedian,
    "std": np.nanstd, "sum": np.nansum,
}
_STAT_LABELS = ["Minimum", "Maximum", "Mean", "Median", "Standard deviation", "Sum"]


@process(
    id=_P + "statistic",
    category="Statistics",
    label="Statistic",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[Param("stat", "string", default="mean", choices=list(_STATS),
                  choice_labels=_STAT_LABELS, label="Statistic")],
    outputs=[Output("value", "float", label="Value")],
)
def statistic(grid, stat="mean"):
    """A statistic of the grid's valid cells (none valid: nodata, NaN; sum: 0)."""
    z = grid.z
    if not np.isfinite(z).any():
        return 0.0 if stat == "sum" else float("nan")
    return float(_STATS[stat](z))


@process(
    id=_P + "percentile",
    category="Statistics",
    label="Percentile",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[Param("p", "float", default=50.0, min=0.0, max=100.0, label="Percentile", unit="%")],
    outputs=[Output("value", "float", label="Value")],
)
def percentile(grid, p=50.0):
    """The p-th percentile of the grid's valid cells (none valid: NaN)."""
    z = grid.z
    if not np.isfinite(z).any():
        return float("nan")
    return float(np.nanpercentile(z, float(p)))


@process(
    id=_P + "count",
    category="Statistics",
    label="Count cells",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[Param("what", "string", default="valid", choices=["valid", "nodata"],
                  choice_labels=["Valid cells", "Nodata cells"], label="Count")],
    outputs=[Output("value", "int", label="Cells")],
)
def count(grid, what="valid"):
    """The number of valid (or nodata) cells."""
    valid = int(np.isfinite(grid.z).sum())
    return valid if what == "valid" else int(grid.z.size - valid)


@process(
    id=_P + "mask_area",
    category="Statistics",
    label="Mask area",
    inputs=[Port("mask", "georaster", label="Mask")],
    outputs=[Output("value", "float", label="Area")],
)
def mask_area(mask):
    """The area of the set (non-zero) cells: their count x cell size², in the
    grid's units squared."""
    return float(_mask_values(mask).sum()) * mask.cell_size ** 2


# ---- grid information -> number ---------------------------------------------

_INFO = {
    "cell_size": lambda g: g.cell_size, "n_rows": lambda g: float(g.n_rows),
    "n_cols": lambda g: float(g.n_cols), "x_min": lambda g: g.x_min, "x_max": lambda g: g.x_max,
    "y_min": lambda g: g.y_min, "y_max": lambda g: g.y_max,
}
_INFO_LABELS = ["Cell size", "Rows", "Columns", "x min", "x max", "y min", "y max"]


@process(
    id=_P + "grid_info",
    category="Statistics",
    label="Grid info",
    inputs=[Port("grid", "georaster", label="Grid")],
    params=[Param("what", "string", default="cell_size", choices=list(_INFO),
                  choice_labels=_INFO_LABELS, label="Value")],
    outputs=[Output("value", "float", label="Value")],
)
def grid_info(grid, what="cell_size"):
    """A number describing the grid's lattice (rows and columns as numbers)."""
    return float(_INFO[what](grid))
