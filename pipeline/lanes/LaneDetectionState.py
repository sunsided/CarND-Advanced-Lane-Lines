from typing import Optional, NamedTuple, List
from pipeline.lanes.tracks import Track, InvalidLeftTrack, InvalidRightTrack

LaneMatch = NamedTuple('LaneMatch', [('track', Optional[Track]), ('age', int)])


class LaneDetectionState:
    def __init__(self, max_age: int=5, max_history: int=10):
        self._left = LaneMatch(track=None, age=0)
        self._right = LaneMatch(track=None, age=0)
        self._tracks_left = []
        self._tracks_right = []
        self._max_age = max_age
        self._max_history = max_history

    @property
    def tracks_left(self):
        return self._tracks_left

    @property
    def tracks_right(self):
        return self._tracks_right

    @property
    def left(self) -> Optional[Track]:
        """
        Obtains the left track if it exists.
        :return: The left track or None
        """
        if self._left.age >= self._max_age:
            self._left = LaneMatch(track=None, age=0)
            return None
        return self._left.track

    @property
    def right(self) -> Optional[Track]:
        """
        Obtains the right track if it exists.
        :return: The right track or None
        """
        if self._right.age >= self._max_age:
            self._right = LaneMatch(track=None, age=0)
            return None
        return self._right.track

    def set_left(self, track: Track):
        """
        Sets the left track.
        :param track: The track.
        """
        self._left = LaneMatch(track=track, age=0)

    def set_right(self, track: Track):
        """
        Sets the right track.
        :param track: The track.
        """
        self._right = LaneMatch(track=track, age=0)

    def age_left(self):
        """
        Ages the left track.
        """
        self._left = LaneMatch(track=self._left.track, age=self._left.age + 1)

    def age_right(self):
        """
        Age the right track.
        """
        self._right = LaneMatch(track=self._right.track, age=self._right.age + 1)

    def confirm_left(self, apply_aging: bool=False):
        """
        Confirms the latest left track from history.
        """
        if len(self._tracks_left) > 0:
            self._left = LaneMatch(track=self._tracks_left[-1], age=0 if not apply_aging else self._left.age + 1)

    def confirm_right(self, apply_aging: bool=False):
        """
        Confirms the latest right track from history.
        """
        if len(self._tracks_right) > 0:
            self._right = LaneMatch(track=self._tracks_right[-1], age=0 if not apply_aging else self._right.age + 1)

    def update_history(self, tracks: List[Track]):
        """
        Updates the history with new tracks.
        :param tracks: The tracks to add to history.

        """
        self._update_history_left([t for t in tracks if t.side < 0])
        self._update_history_right([t for t in tracks if t.side > 0])

    def _update_history_left(self, detection_left: List[Track]):
        if len(detection_left) > 0:
            latest_track = detection_left[0]
            # self.set_left(latest_track)
        else:
            latest_track = InvalidLeftTrack
        self._tracks_left.append(latest_track)
        if len(self._tracks_left) > self._max_history:
            self._tracks_left.pop(0)

    def _update_history_right(self, detection_right: List[Track]):
        if len(detection_right) > 0:
            latest_track = detection_right[0]
            # self.set_right(latest_track)
        else:
            latest_track = InvalidRightTrack
        self._tracks_right.append(latest_track)
        if len(self._tracks_right) > self._max_history:
            self._tracks_right.pop(0)
