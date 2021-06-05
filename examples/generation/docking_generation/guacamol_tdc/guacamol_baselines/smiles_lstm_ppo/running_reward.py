class RunningReward(object):
    def __init__(self, keep_factor: float, initial_value=0) -> None:
        """
        Args:
            keep_factor: How much of the last value to keep when a new one is added.
            initial_value: Initial reward
        """
        assert keep_factor >= 0.0
        assert keep_factor <= 1.0

        self._keep_factor = keep_factor
        self._reward = initial_value
        self.last_added = initial_value

    @property
    def value(self):
        """Get the current running reward."""
        return self._reward

    def update(self, reward):
        """Update the running reward with a new value."""
        self._reward *= self._keep_factor
        self._reward += reward * (1.0 - self._keep_factor)
        self.last_added = reward
