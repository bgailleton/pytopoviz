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

An item's ``sources`` (input port -> item id) name the items it was made
from (a run's inputs); they stay as recorded when those items are released.
A saved session (``pytopoviz.sessionfile``) comes back through ``restore``:
an item under its saved id whose value is read from the saved store (its
``origin``) on first use, or, for a value with no array form, made again
from its ``info["recipe"]`` by the session's ``rebuild`` callable
(``rebuild(handle, recipe, sources) -> value``). A value whose origin holds
it is dropped from memory without a write when it spills.

``replace`` gives an item a new value under the same id (an edit made
elsewhere, e.g. in Topos), so what refers to it by id follows.

Every change to an item (added, renamed, moved, spilled, a group's content or
sizes changed...) stamps it with the session's next ``revision``; a released
item's id is logged with its revision. ``changes(since)`` returns what changed
after a revision, so a viewer keeps its copy of the catalog without rereading
it. Byte totals (in memory, on disk; overall and per group) are kept up to
date as values come and go, not summed on each read.

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
    sources: Dict[str, str] = field(default_factory=dict)
    last_used: float = 0.0
    rev: int = 0
    # Groups: bytes in memory / on disk of the values under them.
    tree_nbytes: int = 0
    tree_disk: int = 0
    # A read-only store holding the value under the item's id (a saved
    # session), or None.
    origin: object = None
    # Revision at which the value was set (put, replace, restore).
    stamp: int = 0


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
        self.revision = 0
        self._removed: List[Tuple[int, str]] = []  # (revision, id) of released items
        self._memory = 0
        self._disk_total = 0
        #: ``rebuild(handle, recipe, sources) -> value``: makes a restored
        #: item's value again from its ``info["recipe"]`` (see module doc).
        self.rebuild = None

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
        sources: Optional[Dict[str, str]] = None,
        id: Optional[str] = None,
    ) -> DataHandle:
        """Stores ``value``; ``id``: the item's id when the caller chose it
        (e.g. a viewer naming its items before they arrive), else a new one."""
        if id is not None and id in self._store:
            raise SessionError(f"id {id!r} is taken")
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
            id=id or uuid.uuid4().hex, type_id=type_id, name=name, provenance=provenance
        )
        entry = _Entry(handle=handle, value=value, nbytes=self._measure(value, type_id))
        self._add(entry, parent, role, coord, info, sources)
        entry.stamp = entry.rev
        self._enforce(keep=handle.id)
        return handle

    def restore(self, row: Dict[str, Any], origin=None) -> DataHandle:
        """Puts back a saved catalog ``row`` (as ``item`` gives it) under its
        id, its parent restored before it; no value in memory: it is read
        from ``origin`` (a store holding it under the id) or made again from
        ``row["info"]["recipe"]`` (see module doc) on first use."""
        hid = str(row["id"])
        if hid in self._store:
            raise SessionError(f"id {hid!r} is taken")
        type_id = str(row["type_id"])
        if type_id != GROUP and not self._types.has(type_id):
            raise SessionError(f"unknown type_id: {type_id!r}")
        parent = self._store[row["parent"]].handle if row.get("parent") else None
        handle = DataHandle(id=hid, type_id=type_id, name=row.get("name"),
                            provenance=row.get("provenance"))
        entry = _Entry(handle=handle, resident=False, axis=row.get("axis") or None,
                       nbytes=int(row.get("nbytes", 0) or 0) if type_id != GROUP else 0,
                       origin=origin, pinned=bool(row.get("pinned", False)))
        self._add(entry, parent, row.get("role", "primary"), row.get("coord"), row.get("info"),
                  row.get("sources"))
        entry.stamp = entry.rev
        return handle

    def set_origin(self, handle: DataHandle, origin) -> None:
        """``origin`` (a store) now holds the item's value under its id."""
        self._entry(handle).origin = origin

    def origin(self, handle: DataHandle):
        """The store holding the item's current value (``set_origin``), or None."""
        return self._entry(handle).origin

    def stamp(self, handle: DataHandle) -> int:
        """The revision at which the item's value was last set."""
        return self._entry(handle).stamp

    def group(
        self,
        name: str,
        parent: Optional[DataHandle] = None,
        role: str = "primary",
        axis: Optional[str] = None,
        coord: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
        provenance: Optional[str] = None,
        sources: Optional[Dict[str, str]] = None,
    ) -> DataHandle:
        """A new group (a series when ``axis`` is given) under ``parent``."""
        handle = DataHandle(id=uuid.uuid4().hex, type_id=GROUP, name=name, provenance=provenance)
        entry = _Entry(handle=handle, resident=False, axis=axis)
        self._add(entry, parent, role, coord, info, sources)
        return handle

    def replace(self, handle: DataHandle, value: object,
                type_id: Optional[str] = None) -> DataHandle:
        """Gives the item a new value (of ``type_id``, default: identified,
        else the item's type), same id; its disk copy is dropped. Returns the
        current handle (its type_id may have changed)."""
        entry = self._entry(handle)
        if entry.handle.type_id == GROUP:
            raise SessionError(f"{handle!r} is a group: it has no value")
        if type_id is None:
            type_id = self._types.identify(value) or entry.handle.type_id
        elif not self._types.has(type_id):
            raise SessionError(f"unknown type_id: {type_id!r}")
        self._sizes(entry, -(entry.nbytes if entry.resident else 0), -entry.disk_bytes)
        if entry.disk_bytes and self.disk is not None:
            self.disk.delete(entry.handle.id)
        entry.handle = replace(entry.handle, type_id=type_id)
        entry.value = value
        entry.resident = True
        entry.disk_bytes = 0
        entry.origin = None
        entry.nbytes = self._measure(value, type_id)
        entry.last_used = time.monotonic()
        self._sizes(entry, entry.nbytes, 0)
        self._touch(entry)
        entry.stamp = entry.rev
        self._enforce(keep=entry.handle.id)
        return entry.handle

    def get(self, handle: DataHandle) -> object:
        entry = self._entry(handle)
        if entry.handle.type_id == GROUP:
            raise SessionError(f"{handle!r} is a group: it has no value")
        entry.last_used = time.monotonic()
        if not entry.resident:
            entry.value = self._load(entry)
            entry.resident = True
            if not entry.nbytes:
                entry.nbytes = self._measure(entry.value, entry.handle.type_id)
            self._sizes(entry, entry.nbytes, 0)
            self._touch(entry)
            self._enforce(keep=entry.handle.id)
        return entry.value

    def read_array(self, handle: DataHandle, index=None) -> Tuple[Any, Dict]:
        """The value's codec array (or ``array[index]``) and meta; from disk
        without loading the whole value when only the disk copy is there."""
        entry = self._entry(handle)
        via = self._route(entry.handle.type_id)
        if via is None:
            raise SessionError(f"{entry.handle.type_id!r} has no array form (no codec)")
        if not entry.resident and entry.disk_bytes:
            return self.disk.read(entry.handle.id, index)
        if not entry.resident and entry.origin is not None:
            return entry.origin.read(entry.handle.id, index)
        self.get(entry.handle)
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
            sources={"source": handle.id},
        )

    def release(self, handle: DataHandle) -> None:
        """Removes the item (a group with everything under it), and its disk copy."""
        if handle.id not in self._store:
            raise SessionError(f"handle not found or already released: {handle!r}")
        entry = self._store[handle.id]
        for cid in list(entry.children):
            self.release(self._store[cid].handle)
        self._sizes(entry, -(entry.nbytes if entry.resident else 0), -entry.disk_bytes)
        self._unlink(entry)
        if entry.disk_bytes and self.disk is not None:
            self.disk.delete(handle.id)
        del self._store[handle.id]
        self._removed.append((self._bump(), handle.id))

    # ---- catalog ----------------------------------------------------------

    def rename(self, handle: DataHandle, name: str) -> DataHandle:
        entry = self._entry(handle)
        entry.handle = replace(entry.handle, name=name)
        self._touch(entry)
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
        mem, disk = self._own_sizes(entry)
        self._carry(entry.parent, -mem, -disk)
        self._unlink(entry)
        entry.parent = pid
        self._link(entry)
        self._carry(pid, mem, disk)
        self._touch(entry)

    def set_role(self, handle: DataHandle, role: str) -> None:
        entry = self._entry(handle)
        entry.role = _check_role(role)
        self._touch(entry)

    def pin(self, handle: DataHandle, pinned: bool = True) -> None:
        """A pinned item (a group: everything under it) is never spilled."""
        for entry in self._walk(self._entry(handle)):
            entry.pinned = pinned
            self._touch(entry)

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
        """One catalog row (JSON-serialisable but for ``info``'s contents). A
        group's row lists its children's ids, in order (``children``)."""
        return self._row(self._entry(handle))

    def _row(self, entry: _Entry) -> Dict[str, Any]:
        h = entry.handle
        row = {
            "id": h.id, "name": h.name, "type_id": h.type_id, "provenance": h.provenance,
            "parent": entry.parent, "role": entry.role, "coord": dict(entry.coord),
            "info": dict(entry.info), "pinned": entry.pinned, "rev": entry.rev,
            "sources": dict(entry.sources),
        }
        if h.type_id == GROUP:
            row.update(axis=entry.axis, n_children=len(entry.children),
                       children=[c.id for c in self.children(h)],
                       nbytes=entry.tree_nbytes, disk_bytes=entry.tree_disk)
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

    def roots(self) -> List[str]:
        """Ids of the root items, in order."""
        return list(self._roots)

    def changes(self, since: int = 0) -> Dict[str, Any]:
        """What changed after revision ``since``: ``items`` (the rows of the
        items added or changed, parents before children), ``removed`` (ids
        released), ``roots`` (root ids, in order) and ``revision`` (the
        current one). ``since`` 0: every row, nothing removed."""
        rows: List[Dict[str, Any]] = []

        def visit(ids):
            for i in ids:
                entry = self._store[i]
                if entry.rev > since:
                    rows.append(self._row(entry))
                if entry.handle.type_id == GROUP:
                    visit(entry.children)

        visit(self._roots)
        removed = [i for r, i in self._removed if r > since] if since > 0 else []
        return {"items": rows, "removed": removed, "roots": list(self._roots),
                "revision": self.revision}

    def memory_used(self) -> int:
        """Bytes of the values held in memory (as measured at put)."""
        return self._memory

    def disk_used(self) -> int:
        return self._disk_total

    def enforce_budget(self) -> int:
        """Spills least recently used values until under ``memory_budget``;
        returns the bytes freed."""
        return self._enforce()

    # ---- internals --------------------------------------------------------

    def _add(self, entry, parent, role, coord, info, sources=None) -> None:
        entry.role = _check_role(role)
        entry.sources = dict(sources or {})
        entry.coord = dict(coord or {})
        entry.info = dict(info or {})
        entry.last_used = time.monotonic()
        entry.parent = self._group_id(parent)
        self._store[entry.handle.id] = entry
        self._link(entry)
        self._sizes(entry, entry.nbytes if entry.resident else 0, entry.disk_bytes)
        self._touch(entry)

    def _bump(self) -> int:
        self.revision += 1
        return self.revision

    def _touch(self, entry: _Entry) -> None:
        entry.rev = self._bump()

    def _link(self, entry: _Entry) -> None:
        """Appends the item to its parent's children (touched: its row lists them)."""
        if entry.parent:
            self._store[entry.parent].children.append(entry.handle.id)
            self._touch(self._store[entry.parent])
        else:
            self._roots.append(entry.handle.id)
            self._bump()

    def _unlink(self, entry: _Entry) -> None:
        if entry.parent:
            self._store[entry.parent].children.remove(entry.handle.id)
            self._touch(self._store[entry.parent])
        else:
            self._roots.remove(entry.handle.id)
            self._bump()

    def _own_sizes(self, entry: _Entry) -> Tuple[int, int]:
        """Bytes in memory / on disk of the item (a group: of everything under it)."""
        if entry.handle.type_id == GROUP:
            return entry.tree_nbytes, entry.tree_disk
        return (entry.nbytes if entry.resident else 0), entry.disk_bytes

    def _sizes(self, entry: _Entry, mem: int, disk: int) -> None:
        """Adds ``mem`` / ``disk`` bytes held by ``entry``'s value to the
        totals and to every group above it."""
        self._memory += mem
        self._disk_total += disk
        self._carry(entry.parent, mem, disk)

    def _carry(self, up: Optional[str], mem: int, disk: int) -> None:
        """Adds bytes to group ``up`` and every group above it (touched)."""
        if not (mem or disk):
            return
        while up is not None:
            group = self._store[up]
            group.tree_nbytes += mem
            group.tree_disk += disk
            self._touch(group)
            up = group.parent

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
        if entry.disk_bytes and self.disk is not None:
            array, meta = self.disk.read(entry.handle.id)
        elif entry.origin is not None:
            array, meta = entry.origin.read(entry.handle.id)
        elif "recipe" in entry.info and self.rebuild is not None:
            return self.rebuild(entry.handle, entry.info["recipe"], dict(entry.sources))
        else:
            raise SessionError(f"{entry.handle!r} has no value in memory nor on disk")
        via = self._route(entry.handle.type_id)
        value = self._types.get(via).codec.decode(array, meta)
        if via != entry.handle.type_id:
            value = self._converters.convert(value, entry.handle.type_id, from_type=via)
        return value

    def _spill(self, entry: _Entry) -> int:
        written = 0
        if not entry.disk_bytes and entry.origin is None:
            array, meta = self._encode(entry, self._route(entry.handle.type_id))
            entry.disk_bytes = written = max(1, int(self.disk.write(entry.handle.id, array, meta)))
        entry.value = None
        entry.resident = False
        self._sizes(entry, -entry.nbytes, written)
        self._touch(entry)
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
