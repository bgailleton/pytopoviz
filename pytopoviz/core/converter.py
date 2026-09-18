"""Converter registry and port-binding resolution.

Converters are explicit, direct, pairwise ``(from_type_id, to_type_id)`` -> callable.
No multi-hop chaining (DESIGN.md §5). Resolution for a value against a port's
accepted set ``S`` (in port-declared order):

1. identify the value's type_id; if it is already in ``S``, pass through.
2. else pick the first ``x in S`` with a registered converter ``(identified, x)``.
3. else error.

Cross-kind conversion is allowed.

Author: B.G.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

from .errors import ConversionError, RegistrationError
from .types import TypeRegistry

ConverterFn = Callable[[object], object]


@dataclass(frozen=True)
class ConverterSpec:
    from_type: str
    to_type: str
    fn: ConverterFn


class ConverterRegistry:
    """Holds converters keyed by ``(from_type_id, to_type_id)``."""

    def __init__(self, types: TypeRegistry) -> None:
        self._types = types
        self._convs: Dict[Tuple[str, str], ConverterSpec] = {}

    def register_converter(
        self, from_type: str, to_type: str, fn: ConverterFn
    ) -> ConverterSpec:
        if not self._types.has(from_type):
            raise RegistrationError(f"converter from unknown type {from_type!r}")
        if not self._types.has(to_type):
            raise RegistrationError(f"converter to unknown type {to_type!r}")
        if from_type == to_type:
            raise RegistrationError(f"converter from a type to itself: {from_type!r}")
        key = (from_type, to_type)
        if key in self._convs:
            raise RegistrationError(f"converter already registered: {from_type} -> {to_type}")
        spec = ConverterSpec(from_type=from_type, to_type=to_type, fn=fn)
        self._convs[key] = spec
        return spec

    def has(self, from_type: str, to_type: str) -> bool:
        return (from_type, to_type) in self._convs

    def get(self, from_type: str, to_type: str) -> ConverterSpec:
        try:
            return self._convs[(from_type, to_type)]
        except KeyError:
            raise ConversionError(f"no converter {from_type!r} -> {to_type!r}")

    def convert(self, value: object, to_type: str, from_type: Optional[str] = None) -> object:
        """Convert ``value`` to ``to_type``. Identifies ``from_type`` if not given."""
        if from_type is None:
            from_type = self._types.identify(value)
            if from_type is None:
                raise ConversionError(
                    f"cannot identify value of type {type(value).__name__} for conversion"
                )
        if from_type == to_type:
            return value
        return self.get(from_type, to_type).fn(value)

    def resolve(
        self, value: object, accepted: Sequence[str]
    ) -> Tuple[object, str]:
        """Coerce ``value`` to satisfy a port accepting ``accepted`` (in order).

        Returns ``(coerced_value, resolved_type_id)``.
        """
        identified = self._types.identify(value)
        if identified is None:
            raise ConversionError(
                f"cannot identify value of type {type(value).__name__}; "
                f"port accepts {list(accepted)}"
            )
        if identified in accepted:
            return value, identified
        for target in accepted:
            if self.has(identified, target):
                return self.get(identified, target).fn(value), target
        raise ConversionError(
            f"no converter from {identified!r} into any of {list(accepted)}"
        )

    def clear(self) -> None:
        self._convs.clear()

    def __iter__(self) -> Iterator[ConverterSpec]:
        return iter(self._convs.values())

    def __len__(self) -> int:
        return len(self._convs)
