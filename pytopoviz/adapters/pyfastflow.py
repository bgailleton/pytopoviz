"""pyfastflow adapter (stub).

Registration unit for pyfastflow (DESIGN.md §8). Hard-imports pyfastflow so the
missing-library guard in ``adapters/__init__`` exercises the mechanism, but surfaces
no process yet: pyfastflow's flow routines run on a GPU/taichi backend that must be
initialised and fed device grids, so those processes are added deliberately later.
Until then a workflow referencing a pyfastflow process fails at validation, as
intended.

Author: B.G.
"""

from __future__ import annotations

import pyfastflow  # noqa: F401  (import proves availability; nothing surfaced yet)

# TODO: surface flow processes (e.g. receivers / accumulation) once the backend
# setup and a grid data type are settled.
