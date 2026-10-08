"""Zarr disk tier of a Session (``Session(store=ZarrStore(path), ...)``).

One zarr v3 group on disk; each spilled value is one array named by its
handle id, its codec meta in the array's attributes. Arrays are chunked by
tiles of ``chunk`` x ``chunk`` cells over the last two axes (1D: ``chunk**2``
values), compressed with Blosc zstd and bit shuffling, so reading a window
or one cell per frame touches few chunks.

Author: B.G.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, Tuple

import numpy as np
import zarr
from zarr.codecs import BloscCodec


class ZarrStore:
    def __init__(self, path: str, chunk: int = 256, clevel: int = 3) -> None:
        self.path = path
        self.chunk = int(chunk)
        self._compressor = BloscCodec(cname="zstd", clevel=int(clevel),
                                      shuffle="bitshuffle")
        self._root = zarr.open_group(path, mode="a")

    def write(self, key: str, array, meta: Dict) -> int:
        """Writes ``array`` under ``key``; returns its bytes on disk."""
        a = np.ascontiguousarray(array)
        if a.ndim == 0:
            a = a.reshape(1)
        if a.ndim == 1:
            chunks = (max(1, min(a.shape[0], self.chunk * self.chunk)),)
        else:
            chunks = (1,) * (a.ndim - 2) + tuple(max(1, min(n, self.chunk)) for n in a.shape[-2:])
        z = self._root.create_array(key, shape=a.shape, dtype=a.dtype, chunks=chunks,
                                    compressors=self._compressor, overwrite=True)
        z[...] = a
        z.attrs["meta"] = meta
        return int(z.nbytes_stored())

    def read(self, key: str, index=None) -> Tuple[np.ndarray, Dict]:
        """The array under ``key`` (or ``array[index]``) and its meta."""
        z = self._root[key]
        return (z[...] if index is None else z[index]), dict(z.attrs["meta"])

    def delete(self, key: str) -> None:
        del self._root[key]

    def clear(self) -> None:
        """Removes the store from disk."""
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)


def export_tree(session, handle, path: str, chunk: int = 256, clevel: int = 3):
    """Writes ``handle`` (a group: its whole tree, as nested zarr groups named
    after the items) to a new zarr store at ``path``. Each array keeps its
    codec meta and the item's catalog row (type, role, coord, axis) in its
    attributes. Returns the names of the values with no array form (left out)."""
    from .core import GROUP

    if os.path.exists(path):
        raise FileExistsError(f"{path} exists")
    out = ZarrStore(path, chunk=chunk, clevel=clevel)
    skipped = []

    def write(h, group, key):
        row = session.item(h)
        attrs = {k: row[k] for k in ("name", "type_id", "role", "coord", "axis") if k in row}
        if h.type_id == GROUP:
            sub = group.create_group(key)
            sub.attrs["item"] = attrs
            used = set()
            for c in session.children(h):
                write(c, sub, _unique(c.name or c.type_id, used))
            return
        if not session.spillable(h):
            skipped.append(h.name or h.id)
            return
        array, meta = session.read_array(h)
        store = ZarrStore.__new__(ZarrStore)
        store.chunk, store._compressor, store._root = out.chunk, out._compressor, group
        store.write(key, array, meta)
        group[key].attrs["item"] = attrs

    write(handle, out._root, _unique(handle.name or "item", set()))
    return skipped


def _unique(name: str, used: set) -> str:
    key = "".join(c if c.isalnum() or c in "-_.= " else "_" for c in name).strip() or "item"
    base, i = key, 1
    while key in used:
        i += 1
        key = f"{base} ({i})"
    used.add(key)
    return key
