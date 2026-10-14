"""Run control: stopping a search on request or after a time limit.

运行控制：用户请求停止或超过时间限制时中止搜索。

Every long loop of the planner - the classification of the lattice, the A*
and Dijkstra heaps, the value-iteration sweeps of dynamic programming, the
generations of the genetic algorithm and the coastline-check rounds - calls
:meth:`RunControl.check` at regular intervals. ``check`` raises
:class:`SearchStopped` when

* the user pressed *Stop* (the web server calls :meth:`RunControl.cancel`
  from another thread when a ``cancel`` message arrives on the WebSocket), or
* the time limit set in the interface has passed.

The planner catches the exception, returns a plan with ``status`` equal to
``"cancelled"`` or ``"time_limit"`` and writes it to the result file like any
other outcome, so an interrupted search is never lost.

Why an exception
----------------
The checks sit deep inside nested loops of several modules. An exception
unwinds all of them at once without each function having to return and test
a flag, and the search code stays readable. The cost of ``check`` is one
``Event.is_set`` and one ``perf_counter`` call; the callers invoke it every
few thousand expansions, so the overhead is far below one per cent.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

#: Status values of a search outcome. 搜索结果状态。
STATUS_FOUND = "found"            # a route was found
STATUS_NO_PATH = "no_path"        # the search space was exhausted without reaching the goal
STATUS_STEP_LIMIT = "step_limit"  # the maximum number of steps was reached
STATUS_CANCELLED = "cancelled"    # the user pressed Stop
STATUS_TIME_LIMIT = "time_limit"  # the time limit set by the user passed
STATUSES = (STATUS_FOUND, STATUS_NO_PATH, STATUS_STEP_LIMIT, STATUS_CANCELLED, STATUS_TIME_LIMIT)

#: Human-readable text of every status (used in the UI and the result files).
STATUS_TEXT = {
    STATUS_FOUND: "Route found",
    STATUS_NO_PATH: "No navigable route between the points",
    STATUS_STEP_LIMIT: "Step limit reached before the goal",
    STATUS_CANCELLED: "Stopped by the user",
    STATUS_TIME_LIMIT: "Time limit reached",
}


class SearchStopped(Exception):
    """Raised by :meth:`RunControl.check` to unwind a running search.

    Attributes
    ----------
    reason : str
        ``"cancelled"`` or ``"time_limit"``.
    elapsed_s : float
        Seconds since the run started.
    """

    def __init__(self, reason: str, elapsed_s: float):
        super().__init__(f"{STATUS_TEXT.get(reason, reason)} after {elapsed_s:.1f} s")
        self.reason = reason
        self.elapsed_s = elapsed_s


class RunControl:
    """Thread-safe stop flag with an optional time limit. 线程安全的停止标志。

    Parameters
    ----------
    time_limit_s : float, optional
        Maximum wall-clock time of the run in seconds; ``None`` or ``0``
        means no limit.

    Examples
    --------
    >>> control = RunControl(time_limit_s=30)
    >>> control.check()          # does nothing while the run may continue
    >>> control.cancel()
    >>> control.check()          # raises SearchStopped("cancelled")
    Traceback (most recent call last):
    ...
    maritime_route.routing.control.SearchStopped: Stopped by the user after 0.0 s
    """

    def __init__(self, time_limit_s: Optional[float] = None,
                 parent: Optional["RunControl"] = None):
        self.time_limit_s = float(time_limit_s) if time_limit_s else None
        #: A control whose Stop also stops this one (the comparison gives every
        #: method its own time limit but shares one Stop button).
        self.parent = parent
        self._event = threading.Event()
        self._t0 = time.perf_counter()
        #: Number of :meth:`check` calls (for the tests and the overhead estimate).
        self.checks = 0

    # ------------------------------------------------------------------ state
    def cancel(self) -> None:
        """Request the run to stop at its next check (may be called from any thread)."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True after :meth:`cancel` has been called (here or on the parent)."""
        return self._event.is_set() or (self.parent is not None and self.parent.cancelled)

    @property
    def elapsed_s(self) -> float:
        """Seconds since the control object was created (the start of the run)."""
        return time.perf_counter() - self._t0

    def restart_clock(self) -> None:
        """Reset the start time (used when the same control serves a new run)."""
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ check
    def check(self) -> None:
        """Raise :class:`SearchStopped` if the run must stop, otherwise return.

        Raises
        ------
        SearchStopped
            With ``reason="cancelled"`` after :meth:`cancel`, or
            ``reason="time_limit"`` once the time limit has passed.
        """
        self.checks += 1
        if self.cancelled:
            raise SearchStopped("cancelled", self.elapsed_s)
        if self.time_limit_s is not None:
            elapsed = self.elapsed_s
            if elapsed > self.time_limit_s:
                raise SearchStopped("time_limit", elapsed)


def check(control: Optional[RunControl]) -> None:
    """Call ``control.check()`` when a control object is given (shorthand)."""
    if control is not None:
        control.check()
