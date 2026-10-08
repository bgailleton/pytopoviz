"""Progress reports from a running process, and cooperative cancel.

A process calls ``report(done, total, phase)`` between units of work (steps
chunks, launches, phases). Nobody listening (scripts, tests): a no-op. A host
(e.g. the Topos bridge) wraps a run in ``listening(callback, cancelled)``;
the callback gets ``(done, total, phase)``. When ``cancelled()`` returns
True, ``report`` raises ``Cancelled`` and the process unwinds (its ``with``
blocks free the GPU programs).

Reports are throttled to ``MIN_INTERVAL`` seconds, except the first, a phase
change and ``done >= total``; the cancel check runs on every call.

Author: B.G.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Callable, Optional

#: Shortest time (s) between two forwarded reports of the same phase.
MIN_INTERVAL = 0.1

_LISTENER: contextvars.ContextVar = contextvars.ContextVar("pytopoviz_progress", default=None)


class Cancelled(Exception):
    """The host asked the running process to stop."""


class _Listener:
    def __init__(self, callback: Callable[[float, float, str], None],
                 cancelled: Optional[Callable[[], bool]]) -> None:
        self.callback = callback
        self.cancelled = cancelled
        self.last = -1.0
        self.phase = None


@contextmanager
def listening(callback: Callable[[float, float, str], None],
              cancelled: Optional[Callable[[], bool]] = None):
    """Forwards the reports made inside the block to ``callback(done, total,
    phase)``; ``cancelled()`` (checked on every report) returning True makes
    ``report`` raise ``Cancelled``."""
    token = _LISTENER.set(_Listener(callback, cancelled))
    try:
        yield
    finally:
        _LISTENER.reset(token)


def report(done: float, total: float, phase: str = "") -> None:
    """``done`` of ``total`` units of ``phase`` are finished (total <= 0: no
    measure, only the phase). Raises ``Cancelled`` when the host asked to stop."""
    listener = _LISTENER.get()
    if listener is None:
        return
    if listener.cancelled is not None and listener.cancelled():
        raise Cancelled("cancelled")
    now = time.monotonic()
    final = total > 0 and done >= total
    if (phase != listener.phase or final or now - listener.last >= MIN_INTERVAL):
        listener.phase = phase
        listener.last = now
        listener.callback(float(done), float(total), str(phase))


def steps(run: Callable[[int], object], n: int, phase: str = "", parts: int = 40,
          offset: int = 0, total: Optional[int] = None,
          sync: Optional[Callable[[], object]] = None) -> None:
    """Calls ``run(k)`` in up to ``parts`` chunks summing to ``n`` steps,
    reporting after each (``offset``/``total`` place them in a longer count).
    ``sync`` (e.g. a GPU synchronize) runs before each report so it follows
    finished work, not queued work. With nobody listening: one ``run(n)``."""
    n = int(n)
    total = n + offset if total is None else int(total)
    if _LISTENER.get() is None or n <= 1:
        if n > 0:
            run(n)
        report(offset + n, total, phase)
        return
    chunk = max(1, -(-n // max(int(parts), 1)))
    done = 0
    report(offset, total, phase)
    while done < n:
        k = min(chunk, n - done)
        run(k)
        done += k
        if sync is not None:
            sync()
        report(offset + done, total, phase)
