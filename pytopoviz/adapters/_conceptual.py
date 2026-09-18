"""pytopoviz-conceptual data types shared across adapters.

These are pytopoviz's own flat conceptual types (DESIGN.md §3), not tied to any
library. They live here rather than in ``core`` because their ``check`` predicates
use numpy, and ``core`` imports no science lib. ``adapters/__init__`` imports this
module before any library adapter so every adapter can reference these type_ids.

Author: B.G.
"""

from __future__ import annotations

import numpy as np

from ..core import register_type

register_type(
    "field2d", "field", lambda v: isinstance(v, np.ndarray) and v.ndim == 2
)
