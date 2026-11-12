"""One Euro filter: smoothing that is steady when you hold still and quick
when you move.

A plain moving average has to pick one: smooth (but laggy) or fast (but
jittery). The One Euro filter changes how much it smooths based on speed.
Slow hand -> heavy smoothing, so the arm doesn't shake. Fast hand -> light
smoothing, so the arm keeps up.
"""

import math


def _alpha(cutoff_hz: float, dt: float) -> float:
    """How much of the new sample to trust (0..1) for a given cutoff."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.5, beta: float = 4.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff  # lower = steadier when still
        self.beta = beta              # higher = less lag when moving fast
        self.d_cutoff = d_cutoff      # smoothing applied to the speed estimate
        self._x = None                # last filtered value
        self._dx = 0.0                # last filtered speed
        self._t = None                # time of the last sample

    def reset(self) -> None:
        self._x, self._dx, self._t = None, 0.0, None

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or t <= self._t:
            self._x, self._t = x, t   # first sample: nothing to smooth yet
            return x
        dt = t - self._t
        self._t = t

        # 1. estimate how fast the value is changing, and smooth that
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        # 2. faster movement -> higher cutoff -> less smoothing
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


class VectorFilter:
    """One independent One Euro filter per number in a list (e.g. x, y, z)."""

    def __init__(self, size: int, **kwargs):
        self._filters = [OneEuroFilter(**kwargs) for _ in range(size)]

    def reset(self) -> None:
        for f in self._filters:
            f.reset()

    def __call__(self, values, t: float) -> list:
        return [f(v, t) for f, v in zip(self._filters, values)]
