"""Runs a workflow document from the command line, with pytopoviz alone.

    python -m pytopoviz.workflow graph.json [--input name=value]...
                                            [--output name=path]... [--quiet]

``--input``: a workflow input. A number, bool or string input takes the text
(``true``/``false``, numbers parsed); a grid input takes a raster file (any
format GDAL opens, read as a georaster, then converted to the input's type);
a vector input takes a vector file (anything geopandas reads).

``--output``: where a declared output is written: a grid as a GeoTIFF, a
vector as a file geopandas writes (its driver from the extension), a table
as CSV. Number / bool / string outputs are printed (one JSON object on
stdout); a data output without ``--output`` is only named.

A workflow that ``requires`` a host (e.g. Topos) is refused here, naming it.
Node status goes to stderr (``--quiet``: none).

Author: B.G.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np

import pytopoviz.adapters  # noqa: F401  (registers the library adapters)
import pytopoviz.routines  # noqa: F401  (registers the composite routines)
from pytopoviz.core import CONVERTERS, TYPES, Session, Workflow, is_param_kind

_SCALAR_KINDS = ("int", "float", "bool", "string", "enum", "path", "color")


def _pairs(items: List[str], what: str) -> Dict[str, str]:
    out = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise SystemExit(f"{what} {item!r}: expected name=value")
        out[name] = value
    return out


def _scalar(kind: str, text: str):
    if kind == "bool":
        low = text.strip().lower()
        if low not in ("true", "false", "1", "0"):
            raise SystemExit(f"{text!r} is not a bool (true / false)")
        return low in ("true", "1")
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    return text


def read_input(type_id: str, text: str):
    """The value of a workflow input of type ``type_id`` from its command-line
    text."""
    kind = TYPES.get(type_id).kind
    if is_param_kind(kind):
        return _scalar(kind, text)
    if kind == "grid":
        from pytopoviz.georaster import GeoRaster

        return CONVERTERS.convert(GeoRaster.read_file(text), type_id, "georaster")
    if kind == "vector":
        import geopandas as gpd

        return CONVERTERS.convert(gpd.read_file(text), type_id, "geopandas.GeoDataFrame")
    raise SystemExit(f"an input of type {type_id!r} ({kind}) cannot be read from the command line")


def write_output(value, type_id: str, path: str) -> None:
    """Writes output ``value`` (of type ``type_id``) to ``path``."""
    kind = TYPES.get(type_id).kind
    if kind == "grid":
        CONVERTERS.convert(value, "georaster", type_id).write_geotiff(path)
    elif kind == "vector":
        CONVERTERS.convert(value, "geopandas.GeoDataFrame", type_id).to_file(path)
    elif kind == "table":
        table = CONVERTERS.convert(value, "datatable", type_id)
        np.savetxt(path, table.to_array(), delimiter=",", header=",".join(table.columns), comments="")
    else:
        raise SystemExit(f"an output of type {type_id!r} ({kind}) cannot be written to a file")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pytopoviz.workflow",
                                     description="Run a pytopoviz workflow document.")
    parser.add_argument("workflow", help="workflow JSON file")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--output", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--quiet", action="store_true", help="no node status on stderr")
    ns = parser.parse_args(argv)

    with open(ns.workflow) as f:
        wf = Workflow.from_json(json.load(f))

    given = _pairs(ns.input, "--input")
    unknown = set(given) - set(wf.inputs)
    if unknown:
        raise SystemExit(f"unknown input(s) {sorted(unknown)}; the workflow has {sorted(wf.inputs)}")
    args = {name: read_input(wf.inputs[name].type, text) for name, text in given.items()}
    targets = _pairs(ns.output, "--output")
    unknown = set(targets) - set(wf.outputs)
    if unknown:
        raise SystemExit(f"unknown output(s) {sorted(unknown)}; the workflow has {sorted(wf.outputs)}")

    def status(event):
        if ns.quiet:
            return
        state = event["state"]
        if state == "running":
            print(f"[{event['index'] + 1}/{event['count']}] {event['node']}", file=sys.stderr)
        elif state == "progress" and event["total"] > 0:
            print(f"    {event['phase']} {event['done']:g}/{event['total']:g}", file=sys.stderr)
        elif state in ("error", "cancelled"):
            print(f"    {state}: {event['message']}", file=sys.stderr)

    session = Session(TYPES, CONVERTERS)
    result = wf.execute(args, session=session, on_status=status)
    printed = {}
    for name, handle in result.outputs.items():
        type_id = handle.type_id
        value = session.get(handle)
        if name in targets:
            write_output(value, type_id, targets[name])
            print(f"{name}: written to {targets[name]}", file=sys.stderr)
        elif TYPES.get(type_id).kind in _SCALAR_KINDS:
            printed[name] = value
        else:
            print(f"{name}: {type_id} (not written: --output {name}=PATH)", file=sys.stderr)
    for name in sorted(set(wf.outputs) - set(result.outputs)):
        print(f"{name}: not produced", file=sys.stderr)
    if printed:
        print(json.dumps(printed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
