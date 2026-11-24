"""Session saved to a folder and opened again (pytopoviz.sessionfile).

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("zarr")

import pytopoviz.adapters  # noqa: F401,E402  (registers library adapters)
from pytopoviz import sessionfile  # noqa: E402
from pytopoviz.core import CONVERTERS, GROUP, TYPES, Session, frames, record  # noqa: E402
from pytopoviz.georaster import GeoRaster  # noqa: E402
from pytopoviz.zarrstore import ZarrStore  # noqa: E402


def _geo(v, n=32):
    return GeoRaster(z=np.full((n, n), float(v)), cell_size=10.0)


def _session(tmp_path, name="cache.zarr", budget=None):
    return Session(TYPES, CONVERTERS, store=ZarrStore(str(tmp_path / name)), memory_budget=budget)


def test_round_trip_lazy_values_and_recipes(tmp_path):
    s = _session(tmp_path)
    dem = s.put(_geo(1), name="dem")
    run = s.group("run", info={"auto": True}, sources={"dem": dem.id})
    series = s.group("frames", parent=run, axis="time")
    for t in (2.0, 1.0):
        record(s, series, {"h": _geo(t)}, t)
    graph = s.put("flow of dem", type_id="string", name="flow", parent=run, role="auxiliary",
                  sources={"dem": dem.id},
                  info={"recipe": {"process": "p", "params": {"a": 1}, "output": "flow"}})
    s.put("live program", type_id="string", name="live", parent=run)
    out = sessionfile.save(s, str(tmp_path / "a.topos"), view={"camera": [1, 2, 3]})
    assert out["data"] == 3 and out["recipes"] == 1 and out["dropped"] == ["live"]

    t = _session(tmp_path, "cache2.zarr")
    calls = []

    def rebuild(handle, recipe, sources):
        calls.append((handle.name, recipe, sources, t.get(t.lookup(sources["dem"])).z[0, 0]))
        return "flow again"

    t.rebuild = rebuild
    view, info = sessionfile.load(t, str(tmp_path / "a.topos"))
    assert view == {"camera": [1, 2, 3]} and info["items"] == 8 and info["skipped"] == []
    assert [r["name"] for r in t.items()] == [r["name"] for r in s.items() if r["name"] != "live"]
    h = t.lookup(dem.id)
    assert not t.item(h)["resident"] and t.item(h)["nbytes"] == _geo(1).z.nbytes
    assert t.read_array(h, (0, slice(0, 2)))[0].tolist() == [1.0, 1.0]
    assert np.all(t.get(h).z == 1.0)
    assert [c for c, _ in frames(t, t.lookup(series.id))] == [1.0, 2.0]
    assert t.item(t.lookup(series.id))["axis"] == "time"
    assert t.item(t.lookup(run.id))["info"] == {"auto": True}
    assert t.get(t.lookup(graph.id)) == "flow again"
    assert calls == [("flow", {"process": "p", "params": {"a": 1}, "output": "flow"},
                      {"dem": dem.id}, 1.0)]
    assert t.item(t.lookup(graph.id))["role"] == "auxiliary"


def test_incremental_saves(tmp_path):
    s = _session(tmp_path)
    a = s.put(_geo(1), name="a")
    b = s.put(_geo(2), name="b")
    saved = sessionfile.SavedSession()
    path = str(tmp_path / "s.topos")
    assert sessionfile.save(s, path, saved=saved)["written"] == 2
    assert sessionfile.save(s, path, saved=saved)["written"] == 0
    s.rename(a, "renamed")  # not a value change
    s.replace(b, _geo(5))
    assert sessionfile.save(s, path, saved=saved)["written"] == 1
    s.release(a)
    sessionfile.save(s, path, saved=saved)
    assert list(ZarrStore(path + "/" + sessionfile.DATA)._root.array_keys()) == [b.id]

    t = _session(tmp_path, "c2.zarr")
    sessionfile.load(t, path)
    assert np.all(t.get(t.lookup(b.id)).z == 5.0)


def test_saved_values_spill_without_writes(tmp_path):
    s = _session(tmp_path)
    hs = [s.put(_geo(i), name=str(i)) for i in range(4)]
    path = str(tmp_path / "s.topos")
    sessionfile.save(s, path)
    one = _geo(0).z.nbytes
    t = _session(tmp_path, "c2.zarr", budget=int(1.5 * one))
    saved = sessionfile.SavedSession()
    sessionfile.load(t, path, saved=saved)
    for h in hs:
        assert np.all(t.get(t.lookup(h.id)).z == float(h.name))
    assert t.memory_used() <= 1.5 * one and t.disk_used() == 0
    # Saving back to the folder it came from writes nothing.
    assert sessionfile.save(t, path, saved=saved)["written"] == 0
    # Save as: everything copied, then read from the new folder.
    out = sessionfile.save(t, str(tmp_path / "b.topos"), saved=saved)
    assert out["written"] == 4 and saved.path.endswith("b.topos")


def test_refuses_a_foreign_folder_and_skips_unknown_types(tmp_path):
    s = _session(tmp_path)
    s.put(_geo(1), name="a")
    other = tmp_path / "docs"
    other.mkdir()
    (other / "thesis.tex").write_text("x")
    with pytest.raises(FileExistsError):
        sessionfile.save(s, str(other))
    path = str(tmp_path / "s.topos")
    sessionfile.save(s, path)
    import json

    doc = json.load(open(path + "/" + sessionfile.MANIFEST))
    doc["items"].insert(0, {"id": "g", "type_id": GROUP, "name": "g", "store": "group"})
    doc["items"].insert(1, {"id": "x", "type_id": "nolib.Thing", "name": "thing", "parent": "g",
                            "store": "recipe"})
    json.dump(doc, open(path + "/" + sessionfile.MANIFEST, "w"))
    t = _session(tmp_path, "c2.zarr")
    _, info = sessionfile.load(t, path)
    assert info["skipped"] == ["thing"] and info["items"] == 2
