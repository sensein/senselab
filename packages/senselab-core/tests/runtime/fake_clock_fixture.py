"""A clock for tests that moves only when told."""


class FakeClock:
    """A `Clock` that moves only when told: `sleep` advances it and records each wait."""

    def __init__(self) -> None:
        """Start at an arbitrary positive time with no waits recorded."""
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        """Return the current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record the wait and advance by it."""
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        """Move the clock forward without a wait."""
        self.now += seconds
