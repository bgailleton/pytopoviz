"""A Session saved to a folder and opened again.

The folder (e.g. ``name.topos/``) holds ``session.json``, the manifest, and
``data.zarr``, a ZarrStore with one array per saved value under the item's
id. The manifest lists the catalog rows (parents before children) with how
each comes back (``store``):

- ``"group"``: a group (a series keeps its axis, frames their coordinate);
- ``"data"``: a value with an array form, in ``data.zarr``;
- ``"recipe"``: a value with no array form (e.g. a flow graph) whose
  ``info["recipe"]`` ({process, params, output}) and ``sources`` make it
  again, all its sources being saved too; it is made again on first use
  (``Session.rebuild``).

Anything else (a live program, a value whose sources are not saved) is not
saved; ``save`` names it under ``dropped``. A viewer's own state goes in the
manifest as ``view`` (any JSON), handed back by ``load``.

Saves are incremental: a ``SavedSession`` remembers the folder and the
stamp of each value written there, so saving to the same folder again
writes only the values set since, and removes those released; a value
saved in another folder is copied as stored (Save as). After a save
or a load the session reads the saved values from the folder (their
``origin``), so a value in memory spills without being written again.

Author: B.G.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from .core import GROUP, Session
from .zarrstore import ZarrStore

FORMAT = "pytopoviz-session"
VERSION = 1
MANIFEST = "session.json"
DATA = "data.zarr"
#: Row fields kept in the manifest.
_ROW_KEYS = ("id", "name", "type_id", "provenance", "parent", "role", "coord", "info",
             "pinned", "sources", "axis", "nbytes")


class SavedSession:
    """Where a session was saved or opened from, and what is written there."""

    def __init__(self) -> None:
        self.path: Optional[str] = None
        self.store: Optional[ZarrStore] = None
        self.stamps: Dict[str, int] = {}  # item id -> stamp of the value written


def save(session: Session, path: str, view: Any = None,
         saved: Optional[SavedSession] = None) -> Dict[str, Any]:
    """Writes ``session`` (and the viewer's ``view``) to folder ``path``.
    Another session's folder there (or a save of one stopped before its
    manifest: only data.zarr) is replaced; any other non-empty folder is
    refused. Returns {path, written, data, recipes, dropped: [name]}."""
    path = os.path.abspath(path)
    saved = saved if saved is not None else SavedSession()
    same = saved.path == path and saved.store is not None and os.path.isdir(path)
    if not same:
        if os.path.isdir(path) and os.listdir(path):
            # A save stopped before its manifest leaves only these: replaced.
            unfinished = set(os.listdir(path)) <= {DATA, MANIFEST + ".tmp"}
            if not os.path.isfile(os.path.join(path, MANIFEST)) and not unfinished:
                raise FileExistsError(f"{path} is not empty and holds no saved session")
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        saved.stamps = {}
        saved.store = ZarrStore(os.path.abspath(os.path.join(path, DATA)))
    store = saved.store

    rows = session.items()
    by_id = {r["id"]: r for r in rows}
    how: Dict[str, str] = {}
    for r in rows:
        if r["type_id"] == GROUP:
            how[r["id"]] = "group"
        elif r.get("spillable"):
            how[r["id"]] = "data"
    # Recipes: their sources saved (as data, or as recipes themselves).
    changed = True
    while changed:
        changed = False
        for r in rows:
            if r["id"] in how or "recipe" not in (r.get("info") or {}):
                continue
            if all(how.get(src) in ("data", "recipe") for src in r.get("sources", {}).values()):
                how[r["id"]] = "recipe"
                changed = True
    dropped = [r.get("name") or r["id"] for r in rows if r["id"] not in how]

    written = 0
    for hid, kind in how.items():
        if kind != "data":
            continue
        h = session.lookup(hid)
        stamp = session.stamp(h)
        if saved.stamps.get(hid) != stamp or hid not in store._root:
            origin = session.origin(h)
            if isinstance(origin, ZarrStore) and os.path.abspath(origin.path) != store.path \
                    and os.path.isdir(os.path.join(origin.path, hid)):
                store.copy_from(origin, hid)  # as stored, not decoded again
            else:
                array, meta = session.read_array(h)
                store.write(hid, array, meta)
            written += 1
        saved.stamps[hid] = stamp
    for key in list(store._root.array_keys()):
        if how.get(key) != "data":
            store.delete(key)
            saved.stamps.pop(key, None)

    items = []
    for r in rows:
        if r["id"] not in how:
            continue
        row = {k: r[k] for k in _ROW_KEYS if k in r}
        # A parent not saved (cannot happen for groups) puts the item at the root.
        if row.get("parent") and row["parent"] not in how:
            row["parent"] = None
        row["store"] = how[r["id"]]
        items.append(row)
    doc = {"format": FORMAT, "version": VERSION,
           "saved": datetime.datetime.now().isoformat(timespec="seconds"),
           "items": items, "view": view}
    tmp = os.path.join(path, MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, default=_json_default)
    os.replace(tmp, os.path.join(path, MANIFEST))

    for hid, kind in how.items():
        if kind == "data":
            session.set_origin(session.lookup(hid), store)
    saved.path = path
    counts = list(how.values())
    return {"path": path, "written": written, "data": counts.count("data"),
            "recipes": counts.count("recipe"), "dropped": dropped}


def load(session: Session, path: str,
         saved: Optional[SavedSession] = None) -> Tuple[Any, Dict[str, Any]]:
    """Restores the items saved in folder ``path`` into ``session`` (which
    should hold none of their ids), values left on disk until used. Returns
    (view, {path, items, skipped: [name]}): items of a type this session
    does not know (a library not installed) are skipped, with what is
    under them."""
    path = os.path.abspath(path)
    doc = read_manifest(path)
    store = ZarrStore(os.path.abspath(os.path.join(path, DATA)))
    saved = saved if saved is not None else SavedSession()
    saved.path, saved.store, saved.stamps = path, store, {}
    skipped: List[str] = []
    gone = set()
    n = 0
    for row in doc.get("items", []):
        if row.get("parent") in gone:
            gone.add(row["id"])
            continue
        try:
            h = session.restore(row, origin=store if row.get("store") == "data" else None)
        except Exception:  # noqa: BLE001  (an unknown type: left out)
            skipped.append(row.get("name") or row["id"])
            gone.add(row["id"])
            continue
        if row.get("store") == "data":
            saved.stamps[h.id] = session.stamp(h)
        n += 1
    return doc.get("view"), {"path": path, "items": n, "skipped": skipped}


def read_manifest(path: str) -> Dict[str, Any]:
    """The manifest of the session saved in folder ``path``; ValueError
    when there is none, or one this version cannot read."""
    try:
        with open(os.path.join(path, MANIFEST)) as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"{path} holds no readable saved session ({e})")
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path} is not a saved session")
    if int(doc.get("version", 0)) > VERSION:
        raise ValueError(f"{path} was saved by a newer version (format {doc['version']})")
    return doc


def is_saved_session(path: str) -> bool:
    return os.path.isfile(os.path.join(path, MANIFEST))


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(value)
