"""Process interface primitives: Port, Param, Output.

- **Port**   = a typed *data* socket (an input carried as a handle).
- **Param**  = a typed *value* (rendered as a widget).
- **Output** = a typed produced value.

``type`` on each is one type_id or a union list of type_ids. When it is a union,
all members must share one kind (validated against a TypeRegistry, not here).
``arg`` remaps the interface name onto the wrapped library's parameter name; it
defaults to ``name`` (DESIGN.md §8).

Author: B.G.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

# One type_id, or a union list of them.
TypeRef = Union[str, Sequence[str]]

_MISSING = object()


def as_type_list(type_ref: TypeRef) -> Tuple[str, ...]:
    """Normalise a TypeRef to an ordered tuple of type_ids (port-declared order)."""
    if isinstance(type_ref, str):
        return (type_ref,)
    out = tuple(type_ref)
    if not out:
        raise ValueError("type union must list at least one type_id")
    return out


@dataclass(frozen=True)
class Port:
    """A typed input data socket."""

    name: str
    type: TypeRef
    optional: bool = False
    arg: Optional[str] = None

    @property
    def types(self) -> Tuple[str, ...]:
        return as_type_list(self.type)

    @property
    def target(self) -> str:
        return self.arg or self.name


@dataclass(frozen=True)
class Param:
    """A typed scalar value (a UI widget)."""

    name: str
    type: TypeRef
    default: object = _MISSING
    choices: Optional[Sequence[object]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    optional: bool = False
    arg: Optional[str] = None

    @property
    def types(self) -> Tuple[str, ...]:
        return as_type_list(self.type)

    @property
    def target(self) -> str:
        return self.arg or self.name

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING

    @property
    def default_value(self) -> object:
        return None if self.default is _MISSING else self.default


@dataclass(frozen=True)
class Output:
    """A typed produced value."""

    name: str
    type: TypeRef

    @property
    def types(self) -> Tuple[str, ...]:
        return as_type_list(self.type)
