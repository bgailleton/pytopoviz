"""Closed kind vocabularies.

A *kind* is a coarse routing tag. The UI must learn to render each one, so adding
a kind is a deliberate edit here (see DESIGN.md §3). Two disjoint sets:

- PARAM_KINDS: scalar values that map 1:1 to a UI widget.
- DATA_KINDS:  convertible things carried as opaque handles.

Author: B.G.
"""

from __future__ import annotations

from typing import FrozenSet

PARAM_KINDS: FrozenSet[str] = frozenset(
    {"int", "float", "bool", "string", "enum", "path", "color"}
)

DATA_KINDS: FrozenSet[str] = frozenset(
    {"grid", "field", "graph", "vector", "table", "file"}
)

KINDS: FrozenSet[str] = PARAM_KINDS | DATA_KINDS


def is_param_kind(kind: str) -> bool:
    return kind in PARAM_KINDS


def is_data_kind(kind: str) -> bool:
    return kind in DATA_KINDS


def is_kind(kind: str) -> bool:
    return kind in KINDS
