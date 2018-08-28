from typing import Optional, List
from .types import Track, HistoricalTrack


class LaneHistory:
    def __init__(self, max_age: int, max_history: int):
        self._history = []  # type: HistoricalTrack
        self._max_age = max_age
        self._max_history = max_history

    @property
    def latest(self) -> Optional[HistoricalTrack]:
        hist = self._history
        return hist[-1] if len(hist) > 0 else None

    @property
    def track(self) -> Optional[Track]:
        latest = self.latest
        return latest.track if latest is not None else None

    @property
    def age(self) -> Optional[int]:
        latest = self.latest
        return latest.age if latest is not None else None

    @property
    def tracks(self) -> List[Track]:
        return [h.track for h in self._history]

    @property
    def history(self) -> List[Track]:
        return self._history

    def append(self, track: Track):
        """
        Appends a track
        :param track: The track.
        """
        self._history.append(HistoricalTrack(track=track, age=0))
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def increment_age(self):
        """
        Age the tracks.
        """
        for i in range(len(self._history)):
            h = self._history[i]
            self._history[i] = HistoricalTrack(track=h.track, age=h.age + 1)
