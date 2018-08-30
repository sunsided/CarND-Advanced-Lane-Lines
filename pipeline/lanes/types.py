import numpy as np
from enum import Enum
from typing import Tuple, List, NamedTuple, Callable, Iterable, Any, Optional

Rect = Tuple[int, int, int, int]
Rects = List[Rect]

SeedHistogram = NamedTuple('SeedHistogram', [
    ('pos', List[int]),
    ('val', List[float])
])

SeedFilter = Callable[[Iterable[int]], List[int]]

Fit = Tuple[np.ndarray, Any, np.ndarray]


class TrackType(Enum):
    Left = -1
    Right = 1


Track = NamedTuple('Track', [('side', TrackType), ('valid', bool),
                             ('curvature_radius', float),
                             ('confidence', float),
                             ('fit', Fit),
                             ('rects', Rects),
                             ('rects_valid', List[bool])])

HistoricalTrack = NamedTuple('HistoricalTrack', [('track', Optional[Track]), ('age', int)])
