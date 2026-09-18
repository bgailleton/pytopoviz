"""Session and data handles.

A *Session* is the system of record for a single run: an in-memory authoritative
store of values behind opaque handles (DESIGN.md §6). Plain ``@process`` calls in a
script bypass it; the workflow runner uses it.

Materialisation to file/npy is a transport adapter's job, not core's. Caching /
refcount / content-hash dedup are deferred; the API is shaped not to preclude them.

Author: B.G.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from .converter import ConverterRegistry
from .errors import SessionError
from .types import TypeRegistry


@dataclass(frozen=True)
class DataHandle:
    """An opaque reference to a stored value."""

    id: str
    type_id: str
    name: Optional[str] = None
    provenance: Optional[str] = None

    def __repr__(self) -> str:
        tag = f" {self.name!r}" if self.name else ""
        return f"<DataHandle {self.type_id}{tag} {self.id[:8]}>"


@dataclass
class _Entry:
    handle: DataHandle
    value: object


class Session:
    """In-memory store of values behind handles for one run."""

    def __init__(
        self, types: TypeRegistry, converters: ConverterRegistry
    ) -> None:
        self._types = types
        self._converters = converters
        self._store: Dict[str, _Entry] = {}

    def put(
        self,
        value: object,
        type_id: Optional[str] = None,
        name: Optional[str] = None,
        provenance: Optional[str] = None,
    ) -> DataHandle:
        if type_id is None:
            type_id = self._types.identify(value)
            if type_id is None:
                raise SessionError(
                    f"cannot identify value of type {type(value).__name__}; "
                    f"pass type_id explicitly"
                )
        elif not self._types.has(type_id):
            raise SessionError(f"unknown type_id: {type_id!r}")
        handle = DataHandle(
            id=uuid.uuid4().hex, type_id=type_id, name=name, provenance=provenance
        )
        self._store[handle.id] = _Entry(handle=handle, value=value)
        return handle

    def get(self, handle: DataHandle) -> object:
        return self._entry(handle).value

    def has(self, handle: DataHandle) -> bool:
        return handle.id in self._store

    def convert(self, handle: DataHandle, to_type_id: str) -> DataHandle:
        """Convert a handle's value to ``to_type_id`` and store the result."""
        entry = self._entry(handle)
        if handle.type_id == to_type_id:
            return handle
        new_value = self._converters.convert(
            entry.value, to_type_id, from_type=handle.type_id
        )
        return self.put(
            new_value,
            type_id=to_type_id,
            name=handle.name,
            provenance=handle.provenance,
        )

    def release(self, handle: DataHandle) -> None:
        if handle.id not in self._store:
            raise SessionError(f"handle not found or already released: {handle!r}")
        del self._store[handle.id]

    def _entry(self, handle: DataHandle) -> _Entry:
        try:
            return self._store[handle.id]
        except KeyError:
            raise SessionError(f"handle not found or released: {handle!r}")

    def __len__(self) -> int:
        return len(self._store)
