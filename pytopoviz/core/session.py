"""Session and data handles.

A *Session* is the system of record for a single run: an authoritative store of
values behind opaque handles (DESIGN.md §6). Plain ``@process`` calls in a
script bypass it; the workflow runner uses it.

The store is a catalog tree. Each item has a parent (None: a root), a ``role``
("primary", or "auxiliary": kept but not put forward, e.g. a flow graph or a
live program) and an optional ``coord`` (its place along its parent's axis).
A *group* (type_id ``GROUP``, no value) holds items; a group with an ``axis``
is a *series*: its children are frames (groups, or values) ordered by their
``coord[axis]`` (a time step, a model time, a parameter value; see
``core/series.py``).

Storage tiers: a value is in memory, on disk, or both. With a ``store`` (the
disk tier: ``write(key, array, meta) -> bytes``, ``read(key, index=None) ->
(array, meta)``, ``delete(key)``; e.g. ``pytopoviz.zarrstore.ZarrStore``) and a
``memory_budget`` (bytes), the least recently used values that are not pinned
are spilled to disk once the values in memory exceed the budget, and read back
on ``get``. A value spills through its type's codec, or through a converter
pair to and from a type with a codec (e.g. ``lsdtt3.Raster`` via
``georaster``); a value with neither (a live GPU program) stays in memory.
Values are never modified in place, so a disk copy, once written, stays valid.

Author: B.G.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .converter import ConverterRegistry
from .errors import SessionError
from .types import TypeRegistry

#: type_id of a catalog group (holds items, has no value).
GROUP = "group"
ROLES = ("primary", "auxiliary")


@dataclass(frozen=True)
class DataHandle:
    """An opaque reference to a stored value (or a group)."""

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
    value: object = None
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    role: str = "primary"
    coord: Dict[str, Any] = field(default_factory=dict)
    axis: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)
    nbytes: int = 0
    resident: bool = True
    disk_bytes: int = 0  # > 0: a disk copy exists
    pinned: bool = False
    last_used: float = 0.0


class Session:
    """Catalog of values behind handles for one run, in memory and spilled to
    a disk ``store`` past ``memory_budget`` bytes (see module doc)."""

    def __init__(
        self,
        types: TypeRegistry,
        converters: ConverterRegistry,
        store=None,
        memory_budget: Optional[int] = None,
    ) -> None:
        self._types = types
        self._converters = converters
        self._store: Dict[str, _Entry] = {}
        self._roots: List[str] = []
        self._routes: Dict[str, Optional[str]] = {}
        self.disk = store
        self.memory_budget = memory_budget

    # ---- values -----------------------------------------------------------

    def put(
        self,
        value: object,
        type_id: Optional[str] = None,
        name: Optional[str] = None,
        provenance: Optional[str] = None,
        parent: Optional[DataHandle] = None,
        role: str = "primary",
        coord: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
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
        entry = _Entry(handle=handle, value=value, nbytes=self._measure(value, type_id))
        self._add(entry, parent, role, coord, info)
        self._enforce(keep=handle.id)
        return handle

    def group(
        self,
        name: str,
        parent: Optional[DataHandle] = None,
        role: str = "primary",
        axis: Optional[str] = None,
        coord: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
        provenance: Optional[str] = None,
    ) -> DataHandle:
        """A new group (a series when ``axis`` is given) under ``parent``."""
        handle = DataHandle(id=uuid.uuid4().hex, type_id=GROUP, name=name, provenance=provenance)
        entry = _Entry(handle=handle, resident=False, axis=axis)
        self._add(entry, parent, role, coord, info)
        return handle

    def get(self, handle: DataHandle) -> object:
        entry = self._entry(handle)
        if entry.handle.type_id == GROUP:
            raise SessionError(f"{handle!r} is a group: it has no value")
        entry.last_used = time.monotonic()
        if not entry.resident:
            entry.value = self._load(entry)
            entry.resident = True
            self._enforce(keep=entry.handle.id)
        return entry.value

    def read_array(self, handle: DataHandle, index=None) -> Tuple[Any, Dict]:
        """The value's codec array (or ``array[index]``) and meta; from disk
        without loading the whole value when only the disk copy is there."""
        entry = self._entry(handle)
        via = self._route(entry.handle.type_id)
        if via is None:
            raise SessionError(f"{entry.handle.type_id!r} has no array form (no codec)")
        if not entry.resident:
            return self.disk.read(entry.handle.id, index)
        array, meta = self._encode(entry, via)
        return (array if index is None else array[index]), meta

    def has(self, handle: DataHandle) -> bool:
        return handle.id in self._store

    def lookup(self, handle_id: str) -> DataHandle:
        """The current handle of the item with id ``handle_id``."""
        try:
            return self._store[handle_id].handle
        except KeyError:
            raise SessionError(f"unknown handle id: {handle_id!r}")

    def convert(self, handle: DataHandle, to_type_id: str) -> DataHandle:
        """Convert a handle's value to ``to_type_id`` and store the result
        (next to the source, as auxiliary)."""
        if handle.type_id == to_type_id:
            return handle
        entry = self._entry(handle)
        new_value = self._converters.convert(
            self.get(handle), to_type_id, from_type=handle.type_id
        )
        parent = self._store[entry.parent].handle if entry.parent else None
        return self.put(
            new_value,
            type_id=to_type_id,
            name=handle.name,
            provenance=handle.provenance,
            parent=parent,
            role="auxiliary",
        )

    def release(self, handle: DataHandle) -> None:
        """Removes the item (a group with everything under it), and its disk copy."""
        if handle.id not in self._store:
            raise SessionError(f"handle not found or already released: {handle!r}")
        entry = self._store[handle.id]
        for cid in list(entry.children):
            self.release(self._store[cid].handle)
        siblings = self._store[entry.parent].children if entry.parent else self._roots
        siblings.remove(handle.id)
        if entry.disk_bytes and self.disk is not None:
            self.disk.delete(handle.id)
        del self._store[handle.id]

    # ---- catalog ----------------------------------------------------------

    def rename(self, handle: DataHandle, name: str) -> DataHandle:
        entry = self._entry(handle)
        entry.handle = replace(entry.handle, name=name)
        return entry.handle

    def move(self, handle: DataHandle, parent: Optional[DataHandle]) -> None:
        """Re-parents the item (``parent`` None: a root)."""
        entry = self._entry(handle)
        pid = self._group_id(parent)
        up = pid
        while up is not None:
            if up == handle.id:
                raise SessionError(f"cannot move {handle!r} under itself")
            up = self._store[up].parent
        (self._store[entry.parent].children if entry.parent else self._roots).remove(handle.id)
        (self._store[pid].children if pid else self._roots).append(handle.id)
        entry.parent = pid

    def set_role(self, handle: DataHandle, role: str) -> None:
        self._entry(handle).role = _check_role(role)

    def pin(self, handle: DataHandle, pinned: bool = True) -> None:
        """A pinned item (a group: everything under it) is never spilled."""
        for entry in self._walk(self._entry(handle)):
            entry.pinned = pinned

    def spill(self, handle: DataHandle) -> int:
        """Moves the item's value (a group: every value under it) to disk;
        returns the bytes freed. Pinned and codec-less values stay."""
        if self.disk is None:
            raise SessionError("this session has no disk store")
        freed = 0
        for entry in self._walk(self._entry(handle)):
            if entry.resident and not entry.pinned and self.spillable(entry.handle):
                freed += self._spill(entry)
        return freed

    def spillable(self, handle: DataHandle) -> bool:
        return handle.type_id != GROUP and self._route(handle.type_id) is not None

    def children(self, handle: Optional[DataHandle] = None) -> List[DataHandle]:
        """The items directly under ``handle`` (None: the roots), in order;
        a series' frames ordered by their coordinate."""
        ids = self._store[self._group_id(handle)].children if handle else self._roots
        if handle is not None and self._store[handle.id].axis:
            axis = self._store[handle.id].axis
            ids = sorted(ids, key=lambda i: _sort_key(self._store[i].coord.get(axis)))
        return [self._store[i].handle for i in ids]

    def parent(self, handle: DataHandle) -> Optional[DataHandle]:
        pid = self._entry(handle).parent
        return self._store[pid].handle if pid else None

    def item(self, handle: DataHandle) -> Dict[str, Any]:
        """One catalog row (JSON-serialisable but for ``info``'s contents)."""
        entry = self._entry(handle)
        h = entry.handle
        row = {
            "id": h.id, "name": h.name, "type_id": h.type_id, "provenance": h.provenance,
            "parent": entry.parent, "role": entry.role, "coord": dict(entry.coord),
            "info": dict(entry.info), "pinned": entry.pinned,
        }
        if h.type_id == GROUP:
            row.update(axis=entry.axis, n_children=len(entry.children),
                       nbytes=sum(e.nbytes for e in self._walk(entry) if e.resident),
                       disk_bytes=sum(e.disk_bytes for e in self._walk(entry)))
        else:
            row.update(nbytes=entry.nbytes, resident=entry.resident,
                       disk_bytes=entry.disk_bytes, spillable=self.spillable(h))
        return row

    def items(self, handle: Optional[DataHandle] = None) -> List[Dict[str, Any]]:
        """Catalog rows depth-first (parents before their children), from the
        roots or from ``handle``'s children."""
        rows: List[Dict[str, Any]] = []

        def visit(hs):
            for h in hs:
                rows.append(self.item(h))
                if h.type_id == GROUP:
                    visit(self.children(h))

        visit(self.children(handle))
        return rows

    def memory_used(self) -> int:
        """Bytes of the values held in memory (as measured at put)."""
        return sum(e.nbytes for e in self._store.values() if e.resident)

    def disk_used(self) -> int:
        return sum(e.disk_bytes for e in self._store.values())

    def enforce_budget(self) -> int:
        """Spills least recently used values until under ``memory_budget``;
        returns the bytes freed."""
        return self._enforce()

    # ---- internals --------------------------------------------------------

    def _add(self, entry, parent, role, coord, info) -> None:
        entry.role = _check_role(role)
        entry.coord = dict(coord or {})
        entry.info = dict(info or {})
        entry.last_used = time.monotonic()
        pid = self._group_id(parent)
        entry.parent = pid
        self._store[entry.handle.id] = entry
        (self._store[pid].children if pid else self._roots).append(entry.handle.id)

    def _group_id(self, parent: Optional[DataHandle]) -> Optional[str]:
        if parent is None:
            return None
        if self._entry(parent).handle.type_id != GROUP:
            raise SessionError(f"{parent!r} is not a group")
        return parent.id

    def _walk(self, entry: _Entry):
        yield entry
        for cid in entry.children:
            yield from self._walk(self._store[cid])

    def _route(self, type_id: str) -> Optional[str]:
        """The type whose codec carries ``type_id`` to disk (itself, or one
        with converters both ways), or None."""
        if type_id not in self._routes:
            via = None
            if self._types.has(type_id) and self._types.get(type_id).codec is not None:
                via = type_id
            else:
                for conv in self._converters:
                    if (conv.from_type == type_id and self._types.get(conv.to_type).codec
                            and self._converters.has(conv.to_type, type_id)):
                        via = conv.to_type
                        break
            self._routes[type_id] = via
        return self._routes[type_id]

    def _encode(self, entry: _Entry, via: str):
        value = entry.value
        if via != entry.handle.type_id:
            value = self._converters.convert(value, via, from_type=entry.handle.type_id)
        return self._types.get(via).codec.encode(value)

    def _load(self, entry: _Entry) -> object:
        if not entry.disk_bytes or self.disk is None:
            raise SessionError(f"{entry.handle!r} has no value in memory nor on disk")
        via = self._route(entry.handle.type_id)
        array, meta = self.disk.read(entry.handle.id)
        value = self._types.get(via).codec.decode(array, meta)
        if via != entry.handle.type_id:
            value = self._converters.convert(value, entry.handle.type_id, from_type=via)
        return value

    def _spill(self, entry: _Entry) -> int:
        if not entry.disk_bytes:
            array, meta = self._encode(entry, self._route(entry.handle.type_id))
            entry.disk_bytes = max(1, int(self.disk.write(entry.handle.id, array, meta)))
        entry.value = None
        entry.resident = False
        return entry.nbytes

    def _measure(self, value: object, type_id: str) -> int:
        n = _nbytes(value)
        if n == 0 and self._route(type_id) not in (None, type_id):
            # A library object that hides its arrays: measured as its codec array.
            try:
                n = _nbytes(self._encode(_Entry(handle=DataHandle("", type_id), value=value),
                                         self._route(type_id))[0])
            except Exception:  # noqa: BLE001  (a size estimate only)
                n = 0
        return n

    def _enforce(self, keep: Optional[str] = None) -> int:
        if self.disk is None or self.memory_budget is None:
            return 0
        over = self.memory_used() - self.memory_budget
        freed = 0
        if over <= 0:
            return 0
        candidates = sorted(
            (e for e in self._store.values()
             if e.resident and not e.pinned and e.handle.id != keep
             and e.nbytes > 0 and self.spillable(e.handle)),
            key=lambda e: e.last_used,
        )
        for entry in candidates:
            if freed >= over:
                break
            freed += self._spill(entry)
        return freed

    def _entry(self, handle: DataHandle) -> _Entry:
        try:
            return self._store[handle.id]
        except KeyError:
            raise SessionError(f"handle not found or released: {handle!r}")

    def __len__(self) -> int:
        return len(self._store)


def _check_role(role: str) -> str:
    if role not in ROLES:
        raise SessionError(f"unknown role {role!r}; expected one of {ROLES}")
    return role


def _sort_key(v):
    # Numbers in order, then anything else by its text, then missing.
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (0, v, "")
    return (2, 0, "") if v is None else (1, 0, str(v))


def _nbytes(value: object, depth: int = 2) -> int:
    """Bytes of the arrays a value holds: its own ``nbytes``, else those of
    its attributes / dict values / list items, ``depth`` levels down."""
    n = getattr(value, "nbytes", None)
    if isinstance(n, int):
        return n
    if depth == 0:
        return 0
    if isinstance(value, dict):
        parts = value.values()
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        try:
            parts = vars(value).values()
        except TypeError:
            return 0
    return sum(_nbytes(p, depth - 1) for p in parts)
