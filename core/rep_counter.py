class RepCounter:
    """
    State machine: UP → DOWN → UP = 1 rep.

    down_threshold: angle below which we consider the position "down"
    up_threshold:   angle above which we consider the position "up"
    """

    def __init__(self, down_threshold: float, up_threshold: float):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self._state = 'up'
        self.count = 0

    def update(self, angle: float) -> bool:
        """Feed current angle. Returns True when a rep is completed."""
        completed = False
        if self._state == 'up' and angle < self.down_threshold:
            self._state = 'down'
        elif self._state == 'down' and angle > self.up_threshold:
            self._state = 'up'
            self.count += 1
            completed = True
        return completed

    def reset(self):
        self._state = 'up'
        self.count = 0
