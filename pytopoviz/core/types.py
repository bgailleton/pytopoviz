"""Type registry.

A *Type* binds a stable ``type_id`` to a ``kind`` and a ``check`` used to decide
whether a runtime value is of that type, plus an optional ``doc`` that interface
members of that type fall back to in the contract. ``check`` is either an isinstance class
(or tuple of classes) or a predicate ``value -> bool`` — the latter disambiguates
variants that share a Python class (e.g. ``field2d_f32`` vs ``field2d_f64``).

The core never imports a science library: adapters register their concrete types
and hand ``check`` in; the registry stores it opaquely.

type_id convention (DESIGN.md §3):
- library-concrete, namespaced: ``topotoolbox.GridObject``, ``lsdtt3.Raster``
- pytopoviz-conceptual, flat:   ``field2d``, ``scalar``, ``river_network``

Author: B.G.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple, Type, Union

from .errors import RegistrationError, TypeError_
from .kinds import is_kind

# An isinstance target, or a value-predicate.
Check = Union[type, Tuple[type, ...], Callable[[object], bool]]


@dataclass(frozen=True)
class TypeSpec:
    """One registered type."""

    type_id: str
    kind: str
    check: Check
    doc: str = ""

    def matches(self, value: object) -> bool:
        chk = self.check
        if isinstance(chk, type) or (
            isinstance(chk, tuple) and all(isinstance(c, type) for c in chk)
        ):
            return isinstance(value, chk)  # type: ignore[arg-type]
        return bool(chk(value))  # type: ignore[operator]


class TypeRegistry:
    """Holds TypeSpecs keyed by type_id."""

    def __init__(self) -> None:
        self._types: Dict[str, TypeSpec] = {}

    def register_type(
        self, type_id: str, kind: str, check: Check, doc: str = ""
    ) -> TypeSpec:
        if not type_id or not isinstance(type_id, str):
            raise RegistrationError("type_id must be a non-empty string")
        if type_id in self._types:
            raise RegistrationError(f"type_id already registered: {type_id!r}")
        if not is_kind(kind):
            raise RegistrationError(f"unknown kind {kind!r} for type {type_id!r}")
        if check is None:
            raise RegistrationError(f"type {type_id!r} needs a check")
        spec = TypeSpec(type_id=type_id, kind=kind, check=check, doc=doc)
        self._types[type_id] = spec
        return spec

    def get(self, type_id: str) -> TypeSpec:
        try:
            return self._types[type_id]
        except KeyError:
            raise TypeError_(f"unknown type_id: {type_id!r}")

    def has(self, type_id: str) -> bool:
        return type_id in self._types

    def kind_of(self, type_id: str) -> str:
        return self.get(type_id).kind

    def identify(self, value: object) -> Optional[str]:
        """Return the type_id whose check matches ``value``, or None.

        First match wins; ambiguous overlapping checks are the registrant's
        responsibility to keep disjoint.
        """
        for spec in self._types.values():
            if spec.matches(value):
                return spec.type_id
        return None

    def clear(self) -> None:
        self._types.clear()

    def __iter__(self) -> Iterator[TypeSpec]:
        return iter(self._types.values())

    def __len__(self) -> int:
        return len(self._types)

    def __contains__(self, type_id: object) -> bool:
        return type_id in self._types
