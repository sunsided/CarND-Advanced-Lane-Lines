import numpy as np
from typing import Optional, List
from .types import Track, HistoricalTrack, Fit


class LaneHistory:
    def __init__(self, max_age: int, max_history: int, decay: float):
        # History is weighted by a Gaussian. However, we want to react to new information
        # quickly still; because of this, we add in a high coefficient for the first result.
        # An adaptive system like a Kalman filter would be a better choice here.
        self._initial = 0.8
        self._sigma = decay
        self._mu = 1.5
        self._history = []  # type: HistoricalTrack
        self._max_age = max_age
        self._max_history = max_history

    def _gauss(self, x):
        return 1 / (self._sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - self._mu) ** 2 / (2 * self._sigma ** 2))

    @property
    def latest(self) -> Optional[HistoricalTrack]:
        hist = self._history
        return hist[-1] if len(hist) > 0 else None

    @property
    def track(self) -> Optional[Track]:
        latest = self.latest
        return latest.track if latest is not None else None

    @property
    def valid(self) -> bool:
        latest = self.latest
        return latest.track.valid if latest is not None else False

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

    def smoothened_fit(self) -> Optional[Fit]:
        if len(self._history) == 0:
            return None
        fit = np.array((0, 0, 0), np.float32)
        norm = 0
        for i, h in enumerate(reversed(self._history)):
            track = h.track
            # Instead of the age, we use the index as the weight; otherwise, old tracks will
            # be forgotten relatively quickly, leading to loss of track.
            alpha = self._initial if i == 0 else self._gauss(i)  # h.age)
            if alpha < 1e-8:
                break
            weight = alpha * track.confidence
            norm += weight
            fit += track.fit * weight
        if norm < 0.01:
            return None
        return tuple(fit / norm)

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
