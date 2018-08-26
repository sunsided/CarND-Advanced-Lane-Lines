import cv2
import numpy as np
from typing import Optional, List, Tuple
from pipeline import LaneDetectionState, validate_fit, create_track, recenter_rects, regress_lanes, Track, Fit, \
    blend_tracks

VALID_COLOR = (1, 0.5, 0.1)
CACHED_COLOR = (0.75, 0.1, 1)
WARNING_COLOR = (0.05, 0.1, 1)


def render_lane(canvas: np.ndarray, fit: Fit, highest_rect: Optional[float] = None, color=VALID_COLOR) -> np.ndarray:
    h, w = canvas.shape[:2]
    highest_rect = min(h // 2, highest_rect) if highest_rect is not None else h // 2
    ys = np.linspace(h - 1, highest_rect, h - highest_rect)
    xs = np.polyval(fit, ys)
    pts = np.int32([(x, y) for (x, y) in zip(xs, ys)])
    cv2.polylines(canvas, [pts], False, color=color, thickness=4, lineType=cv2.LINE_AA)
    return canvas


def render_rects(canvas: np.ndarray, rects: List[Tuple[int, int, int, int]], alpha: float) -> np.ndarray:
    for rect in rects:
        cv2.rectangle(canvas, rect[:2], rect[2:], color=(0, 0, alpha), thickness=1)
    return canvas


def detect_and_render_lane(canvas: np.ndarray, edges: np.ndarray, tracks: List[Track],
                           offset_thresh: int, confidence_thresh: float,
                           render_lanes: bool = False, render_boxes: bool = True,
                           cached: Optional[Track] = None) -> Tuple[bool, Optional[Fit]]:
    assert tracks is not None
    h, w = edges.shape[:2]

    # We pick the latest track for rendering the rectangles.
    track = tracks[-1]
    assert track is not None

    # The rendering of the lane line estimate will extend up to either
    # the "highest" position of a matched box or half the image size.
    highest_rect = h // 2
    if track.valid:
        highest_rect = track.rects[-1][1]
        ys = np.linspace(h - 1, highest_rect, h - highest_rect)
        xs = np.polyval(track.fit, ys)
        if xs[0] < offset_thresh or xs[0] > (w - offset_thresh) or track.confidence < confidence_thresh:
            if render_boxes:
                render_rects(canvas, track.rects, 0.25)
            if cached is not None and render_lanes:
                render_lane(canvas, cached.fit, highest_rect, CACHED_COLOR)
            return False, cached.fit if cached is not None else None
        if render_boxes:
            render_rects(canvas, track.rects, 1)

    # Interpolate the tracks based on their deviation of curvature from the consensus.
    # Validate the interpolated fit against support in the image. If none is found, discard.
    fit = blend_tracks(tracks)
    valid = True
    if fit is None:
        valid = False
    if valid and validate_fit(edges, fit) is None:
        valid = False
    if valid:
        if render_lanes:
            render_lane(canvas, fit, highest_rect)
        return True, fit
    elif cached is not None:
        if render_lanes:
            render_lane(canvas, cached.fit, highest_rect, CACHED_COLOR)
        return False, cached.fit
    return False, None


def detect_and_render_lanes(img: np.ndarray, edges: np.ndarray, state: LaneDetectionState, mx: float, my: float,
                            left_thresh: int = 50, right_thresh: int = 50, confidence_thresh: float = .3,
                            box_width: int = 30, box_height: int = 10, degree: int = 2,
                            render_lanes: bool = False, render_boxes: bool = False) \
        -> Tuple[Tuple[bool, Optional[Fit]], Tuple[bool, Optional[Fit]]]:
    tracks = []

    left = state.left
    left_from_cache = False

    if left is not None and left.confidence > 0:
        rects = validate_fit(edges, left.fit, box_width=box_width, box_height=box_height)
        if rects is not None:
            left = create_track(-1, rects, xs=None, ys=None, mx=mx, degree=degree)
            if left.confidence > confidence_thresh:
                left = recenter_rects(left, rects)
                tracks.append(left)
                left_from_cache = True
                print('Selecting left line from cache.')
            else:
                print('Confidence too low on cached left line.')
        else:
            print('Support too low on left line.')

    right = state.right
    right_from_cache = False

    if right is not None and right.confidence > 0:
        rects = validate_fit(edges, right.fit, box_width=box_width, box_height=box_height)
        if rects is not None:
            right = create_track(1, rects, xs=None, ys=None, mx=mx, degree=degree)
            if right.confidence > confidence_thresh:
                right = recenter_rects(right, rects)
                tracks.append(right)
                right_from_cache = True
                print('Selecting right line from cache.')
            else:
                print('Confidence too low on cached right line.')
        else:
            print('Support too low on right line.')

    if not left_from_cache or not right_from_cache:
        new_tracks = regress_lanes(edges, k=2, degree=degree,
                                   search_height=10, max_height=0.55,
                                   max_strikes=15, box_width=box_width, box_height=box_height, threshold=5,
                                   fit_weight=.1, centroid_weight=1, n_smooth=10,
                                   mx=mx, my=my,
                                   detect_left=not left_from_cache,
                                   detect_right=not right_from_cache)
        tracks.extend(new_tracks)

    state.update_history(tracks)
    left_match, left_fit = detect_and_render_lane(img, edges, state.tracks_left, left_thresh,
                                                  confidence_thresh, cached=left,
                                                  render_lanes=render_lanes, render_boxes=render_boxes)
    right_match, right_fit = detect_and_render_lane(img, edges, state.tracks_right, right_thresh,
                                                    confidence_thresh, cached=right,
                                                    render_lanes=render_lanes, render_boxes=render_boxes)

    if (left_match or left_from_cache) and left_fit is not None:
        should_age = not left_match and left_from_cache
        state.confirm_left(should_age)

    if (right_match or right_from_cache) and right_fit is not None:
        should_age = not right_match and right_from_cache
        state.confirm_right(should_age)

    return (left_match, left_fit), (right_match, right_fit)
