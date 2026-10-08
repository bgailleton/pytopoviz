"""pytopoviz composite routines.

A routine is a normal ``@process`` whose body calls other registered processes
(DESIGN.md §2, §4) — it carries the same contract as any leaf process, with
``impl="composite"``. Routines resolve the processes they call through the default
registry at call time, so a routine over a missing library surfaces the failure
where the workflow runs, not at import.

Importing this package registers every routine.
"""

from __future__ import annotations

from . import dem, flow, rivers, swath  # noqa: F401  (registers routines)
