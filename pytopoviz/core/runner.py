"""Live runners: a process whose library state stays open between calls.

A runner shares its process's interface (ports, params, outputs: the contract
entry and the UI forms are the process's). It is opened once with the input
values and the params, then ``advance(params)`` moves it on by one chunk (the
process's own count: steps, launches, model time) from where the previous
chunk stopped, and returns the process's outputs. ``close()`` frees it.

``coordinate()`` says where the runner stands after its advances, as
``{axis: value}`` (e.g. ``{"time": 3600.0}``, ``{"step": 300}``); a host
recording the advances as a series (``core/series.py``) uses it. ``axis``
(declared with the decorator) names that axis in the contract.

``status()`` says how the last advance went, for a host to show and to know
when to stop advancing: ``{"text": one line, "stopped": True when the run
reached an end of its own (e.g. converged)}``.

``config`` names the params fixed at opening (topology, solver, ...): a host
that changes one opens a new runner. The other params apply from the next
``advance``.

A process with a runner is usually written as "open, advance once, close"
(``run_once``), so one code path serves both.

Author: B.G.
"""

from __future__ import annotations

from typing import Dict, Iterable, Type

#: process id -> Runner subclass
RUNNERS: Dict[str, Type["Runner"]] = {}


class Runner:
    """Base of the live runners (see module doc). Subclasses take
    ``(inputs, params)`` (port -> value, full param dict) in ``__init__``."""

    process_id: str = ""
    config: tuple = ()
    axis: str = ""
    #: method name -> label: what can be asked of the runner between advances
    #: (the wire's ``runner_call``). A method taking ``x, y`` asks at a point
    #: (map coordinates of its grid). It returns a dict: ``text`` to show,
    #: optionally ``at`` ([x, y], a place to mark) and ``outputs`` (new
    #: values of its outputs, published as an advance's are).
    actions: dict = {}

    def advance(self, params: Dict) -> Dict:
        raise NotImplementedError

    def update(self, inputs: Dict) -> None:
        """Takes new values of some inputs (port -> value) between advances,
        keeping the rest of its state (a workflow Repeat zone feeding the
        runner what an iteration made). A runner that cannot leaves this
        out (``takes_updates`` is then False)."""
        raise NotImplementedError

    @classmethod
    def takes_updates(cls) -> bool:
        return cls.update is not Runner.update

    def coordinate(self) -> Dict[str, float]:
        """``{axis: value}`` reached by the advances so far ({}: none)."""
        return {}

    def status(self) -> Dict:
        """``{"text", "stopped"}`` after the last advance (see module doc)."""
        return {"text": "", "stopped": False}

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def runner(process_id: str, config: Iterable[str] = (), axis: str = "", actions: Dict[str, str] = None):
    """Class decorator registering a Runner for ``process_id``; ``axis``:
    the key of its ``coordinate()``; ``actions``: see Runner.actions."""

    def decorate(cls):
        cls.process_id = process_id
        cls.config = tuple(config)
        cls.axis = axis
        cls.actions = dict(actions or {})
        for name in cls.actions:
            if not callable(getattr(cls, name, None)):
                raise TypeError(f"runner {process_id!r}: action {name!r} is not a method")
        RUNNERS[process_id] = cls
        return cls

    return decorate


def run_once(cls, inputs: Dict, params: Dict) -> Dict:
    """Open ``cls``, advance once with ``params``, close: the process call."""
    with cls(inputs, params) as r:
        return r.advance(params)
