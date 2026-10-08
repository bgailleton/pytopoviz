"""pytopoviz-conceptual data types shared across adapters.

These are pytopoviz's own flat conceptual types (DESIGN.md §3), not tied to any
library. They live here rather than in ``core`` because their ``check`` predicates
use numpy, and ``core`` imports no science lib. ``adapters/__init__`` imports this
module before any library adapter so every adapter can reference these type_ids.

Author: B.G.
"""

from __future__ import annotations

import numpy as np

from ..core import Codec, register_converter, register_type
from ..datatable import DataTable
from ..georaster import GeoRaster
from ..geovector import GEOVECTOR_TYPES, GeoVector

# Codecs: each hub type as one array + JSON meta, through which a Session
# spills it to disk (core/session.py); library types spill via their hub.
register_type(
    "field2d", "field", lambda v: isinstance(v, np.ndarray) and v.ndim == 2,
    codec=Codec(lambda a: (a, {}), lambda a, _m: np.asarray(a)),
)

# Georeferenced grid hub (pytopoviz/georaster.py). Every grid-kind library type
# registers a converter to/from it so frontends only handle this one container.
register_type("georaster", "grid", GeoRaster,
              codec=Codec(lambda g: (g.z, g.meta()), GeoRaster.from_meta))

register_converter("georaster", "field2d", lambda g: np.array(g.z))

# Vector hub (pytopoviz/geovector.py): one type id per geometry, so a port asks
# for the geometry it handles and frontends list only matching objects.
_GEOMETRY_DOCS = {
    "points": "Points (each feature a point or a multipoint), with their CRS.",
    "lines": "Lines (each feature a polyline or a multi-polyline), with their CRS.",
    "polygons": "Polygons (each feature a polygon or a multipolygon, holes allowed), with their CRS.",
}
for _geometry, _type_id in GEOVECTOR_TYPES.items():
    register_type(
        _type_id, "vector",
        lambda v, g=_geometry: isinstance(v, GeoVector) and v.geometry == g,
        doc=_GEOMETRY_DOCS[_geometry],
        codec=Codec(lambda v: (v.to_array(), v.meta()), GeoVector.from_meta),
    )

# Table hub (pytopoviz/datatable.py): numeric columns with roles, e.g. a profile.
register_type("datatable", "table", DataTable,
              doc="Table of numeric columns (e.g. a profile), with units and plot roles.",
              codec=Codec(lambda t: (t.to_array(), t.meta()), DataTable.from_meta))
