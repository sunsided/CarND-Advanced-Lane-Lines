from .LaneHistory import LaneHistory


class Lanes:
    def __init__(self, max_age: int, max_history: int):
        self._left = LaneHistory(max_age, max_history)
        self._right = LaneHistory(max_age, max_history)
        self._tracks_left = []
        self._tracks_right = []
        self._max_age = max_age
        self._max_history = max_history

    @property
    def left(self) -> LaneHistory:
        return self._left

    @property
    def right(self) -> LaneHistory:
        return self._right
