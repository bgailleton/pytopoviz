"""Session catalog: groups, series, sweep, spilling to a zarr store.

Author: B.G.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("zarr")

import pytopoviz.adapters  # noqa: F401,E402  (registers library adapters)
from pytopoviz.core import (  # noqa: E402
    CONVERTERS, GROUP, TYPES, Output, Param, Port, Session, SessionError, frames,
    process, record, sweep,
)
from pytopoviz.georaster import GeoRaster  # noqa: E402
from pytopoviz.zarrstore import ZarrStore, export_tree  # noqa: E402


def _geo(v, n=64):
    return GeoRaster(z=np.full((n, n), float(v)), cell_size=10.0)


def test_tree_series_and_spill(tmp_path):
    store = ZarrStore(str(tmp_path / "cache.zarr"))
    one = _geo(0).z.nbytes
    s = Session(TYPES, CONVERTERS, store=store, memory_budget=int(2.5 * one))
    run = s.group("flood run")
    aux = s.put("live program", type_id="string", name="aux", parent=run, role="auxiliary")
    series = s.group("frames", parent=run, axis="time")
    for t in (20.0, 10.0, 30.0):
        record(s, series, {"h": _geo(t), "q": None}, t)
    assert [c for c, _ in frames(s, series)] == [10.0, 20.0, 30.0]
    hs = [h for _, h in frames(s, series, "h")]
    assert len(hs) == 3 and all(h.name == "h" for h in hs)

    # Over budget: the oldest frames went to disk, the last put stays.
    assert s.memory_used() <= 2.5 * one and s.disk_used() > 0
    first = [h for c, h in frames(s, series, "h") if c == 20.0][0]
    assert not s.item(first)["resident"]
    assert s.read_array(first, (0, slice(0, 3)))[0].tolist() == [20.0] * 3
    assert np.all(s.get(first).z == 20.0) and s.item(first)["resident"]

    rows = s.items()
    assert [r["name"] for r in rows][:3] == ["flood run", "aux", "frames"]
    assert rows[1]["role"] == "auxiliary"
    s.pin(series)
    before = [s.item(h)["resident"] for h in hs]
    s.spill(run)  # pinned frames and the codec-less aux stay as they are
    assert [s.item(h)["resident"] for h in hs] == before and s.item(aux)["resident"]
    with pytest.raises(SessionError):
        s.get(series)
    s.release(run)
    assert len(s) == 0 and s.disk_used() == 0 and list(store._root.array_keys()) == []


def test_lsdtt3_raster_spills_via_georaster(tmp_path):
    pytest.importorskip("lsdtt3")
    s = Session(TYPES, CONVERTERS, store=ZarrStore(str(tmp_path / "c.zarr")))
    r = CONVERTERS.convert(_geo(3.0), "lsdtt3.Raster", from_type="georaster")
    h = s.put(r, type_id="lsdtt3.Raster")
    assert s.item(h)["nbytes"] > 0 and s.spillable(h)
    s.spill(h)
    back = CONVERTERS.convert(s.get(h), "georaster", from_type="lsdtt3.Raster")
    assert np.all(back.z == 3.0)


def test_sweep_over_a_param():
    @process(id="test.catalog_scale", inputs=[Port("dem", "georaster")],
             params=[Param("k", "float", default=1.0)], outputs=[Output("out", "georaster")])
    def scale(dem, k):
        return GeoRaster(z=dem.z * k, cell_size=dem.cell_size)

    s = Session(TYPES, CONVERTERS)
    series = sweep(s, scale, "k", [1.0, 2.0, 4.0], inputs={"dem": _geo(1.0, 8)})
    assert s.item(series)["axis"] == "k" and series.type_id == GROUP
    got = [(c, float(s.get(h).z[0, 0])) for c, h in frames(s, series, "out")]
    assert got == [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]


def test_export_tree(tmp_path):
    import zarr

    s = Session(TYPES, CONVERTERS)
    series = s.group("run", axis="t")
    for t in (1.0, 2.0):
        record(s, series, {"h": _geo(t, 8), "note": "x"}, t,
               type_ids={"note": "string"})
    skipped = export_tree(s, series, str(tmp_path / "out.zarr"))
    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert skipped == ["note", "note"]
    assert root["run/t = 2"].attrs["item"]["coord"] == {"t": 2.0}
    assert float(root["run/t = 2/h"][0, 0]) == 2.0


def test_series_table_stats_and_points():
    from pytopoviz.series_table import series_table

    s = Session(TYPES, CONVERTERS)
    series = s.group("run", axis="time")
    for t in (1.0, 2.0):
        g = _geo(0.0, 8)
        g.z[:] = np.arange(64.0).reshape(8, 8) * t
        record(s, series, {"h": g}, t)
    # point (15, 75): column 1, row 0 (north-up, 8 rows of 10 m)
    table = series_table(s, series, "h", [(15.0, 75.0)])
    assert table.columns["time"].tolist() == [1.0, 2.0]
    assert table.columns["h max"].tolist() == [63.0, 126.0]
    assert table.columns["h at point 1"].tolist() == [1.0, 2.0]
