import cv2
import numpy as np
from typing import Optional, List, Tuple, NamedTuple
from pipeline import Lanes, validate_fit, create_track, recenter_rects, regress_lanes, Track, Fit, \
    blend_tracks

VALID_COLOR = (1, 0.5, 0.1)
CACHED_COLOR = (0.75, 0.1, 1)
WARNING_COLOR = (0.05, 0.1, 1)


def interpolate_and_render_lane(canvas: np.ndarray, edges: np.ndarray, tracks: List[Track],
                                offset_thresh: int, confidence_thresh: float,
                                render_lanes: bool = False, render_boxes: bool = True,
                                cached: Optional[Track] = None,
                                boxes_thresh: int=8) -> Tuple[bool, Optional[Fit]]:
    assert tracks is not None
    h, w = edges.shape[:2]

    # We pick the latest track for rendering the rectangles.
    if len(tracks) == 0:
        if cached is None:
            return False, None
        else:
            tracks = [cached]
    track = tracks[-1]
    assert track is not None

    # The rendering of the lane line estimate will extend up to either
    # the "highest" position of a matched box or half the image size.
    highest_rect = h // 2
    if track.valid:
        highest_rect = track.rects[-1][1]
        ys = np.linspace(h - 1, highest_rect, h - highest_rect)
        xs = np.polyval(track.fit, ys)
        confidence_ok = track.confidence >= confidence_thresh
        boxes_ok = len(track.rects) >= boxes_thresh
        offset_ok = (xs[0] >= offset_thresh) and (xs[0] <= (w - offset_thresh))
        if not confidence_ok:
            print('    Confidence too low for track ({} < {}).'.format(track.confidence, confidence_thresh))
        if not boxes_ok:
            print('    Too few boxes for track ({}/{}).'.format(len(track.rects), boxes_thresh))
        if not offset_ok:
            print('    Track offset violated.')
        if not (offset_ok and confidence_ok and boxes_ok):
            if render_boxes:
                render_rects(canvas, track.rects, 0.25)
            if cached is not None and render_lanes:
                render_lane(canvas, cached.fit, highest_rect, CACHED_COLOR)
            return False, None
        if render_boxes:
            render_rects(canvas, track.rects, 1)

    # Interpolate the tracks based on their deviation of curvature from the consensus.
    # Validate the interpolated fit against support in the image. If none is found, discard.
    fit = blend_tracks(tracks)
    valid = True
    if fit is None:
        print('    Track interpolation yielded invalid result.')
        valid = False
    if valid and validate_fit(edges, fit) is None:
        print('    Track interpolation could not be verified.')
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



def detect_and_render_lanes(img: np.ndarray, edges: np.ndarray, state: Lanes, mx: float, my: float,
                            left_thresh: int = 50, right_thresh: int = 50,
                            confidence_thresh: float = .5,
                            confidence_thresh_cached: float = .5,
                            box_width: int = 30, box_height: int = 10, degree: int = 2,
                            boxes_thresh: int=7, boxes_thresh_cached: int=4,
                            render_lanes: bool = False, render_boxes: bool = False) \
        -> Tuple[Tuple[bool, Optional[Fit]], Tuple[bool, Optional[Fit]]]:
    tracks = []

    left = state.left.track




    if left is not None and left.confidence > 0:
        rects = validate_fit(edges, left.fit, box_width=box_width, box_height=box_height, min_support=.3)
        if rects is not None:
            left = create_track(-1, rects, xs=None, ys=None, mx=mx, degree=degree)
            if left.confidence > confidence_thresh:
                left = recenter_rects(left, rects)
                tracks.append(left)
                left_from_cache = True
            else:
                print('[X] Confidence too low on cached left line.')
        else:
            print('[X] Support too low on left line.')

    right = state.right
    right_from_cache = False

    if right is not None and right.confidence > 0:
        rects = validate_fit(edges, right.fit, box_width=box_width, box_height=box_height, min_support=.3)
        if rects is not None:
            right = create_track(1, rects, xs=None, ys=None, mx=mx, degree=degree)
            if right.confidence > confidence_thresh:
                right = recenter_rects(right, rects)
                tracks.append(right)
                right_from_cache = True
            else:
                print('[X] Confidence too low on cached right line.')
        else:
            print('[X] Support too low on right line.')

    # If we have a match from cache for the left lane line, validate it.
    left_match, left_fit = False, None
    if left_from_cache:
        print('[C] Trying left line from cache.')
        left_tracks = []
        left_tracks.extend(state.tracks_left)
        left_tracks.extend([t for t in tracks if t.side < 0])
        left_match, left_fit = interpolate_and_render_lane(img, edges, left_tracks, left_thresh,
                                                           confidence_thresh=confidence_thresh_cached,
                                                           boxes_thresh=boxes_thresh_cached,
                                                           cached=left,
                                                           render_lanes=render_lanes, render_boxes=render_boxes)
        left_from_cache = left_match

    # Likewise, valide the right cache entry, if it exists.
    right_match, right_fit = False, None
    if right_from_cache:
        print('[C] Trying right line from cache.')
        right_tracks = []
        right_tracks.extend(state.tracks_right)
        right_tracks.extend([t for t in tracks if t.side > 0])
        right_match, right_fit = interpolate_and_render_lane(img, edges, right_tracks, right_thresh,
                                                             confidence_thresh=confidence_thresh_cached,
                                                             boxes_thresh=boxes_thresh_cached,
                                                             cached=right,
                                                             render_lanes=render_lanes, render_boxes=render_boxes)
        right_from_cache = right_match

    if not left_from_cache:
        print('[?] Re-scanning for left line.')
    if not right_from_cache:
        print('[?] Re-scanning for right line.')

    # If a cache entry did not exist or didn't check out good, we re-start from scratch.
    if not left_from_cache or not right_from_cache:
        new_tracks = regress_lanes(edges, degree=degree,
                                   search_height=10, max_height=0.8,
                                   max_strikes=15, box_width=box_width, box_height=box_height, threshold=5,
                                   fit_weight=.1, centroid_weight=1, n_smooth=10,
                                   mx=mx, my=my,
                                   detect_left=not left_from_cache,
                                   detect_right=not right_from_cache)
        if new_tracks[0] is not None:
            tracks.append(new_tracks[0])
        if new_tracks[1] is not None:
            tracks.append(new_tracks[1])

        if not left_from_cache:
            print('[!] Verifying left line from search.')
            left_tracks = []
            left_tracks.extend(state.tracks_left)
            left_tracks.extend([t for t in tracks if t.side < 0])
            left_match, left_fit = interpolate_and_render_lane(img, edges, left_tracks, left_thresh,
                                                               confidence_thresh, boxes_thresh=boxes_thresh,
                                                               render_lanes=render_lanes, render_boxes=render_boxes)

        if not right_from_cache:
            print('[!] Verifying right line from search.')
            right_tracks = []
            right_tracks.extend(state.tracks_right)
            right_tracks.extend([t for t in tracks if t.side > 0])
            right_match, right_fit = interpolate_and_render_lane(img, edges, right_tracks, right_thresh,
                                                                 confidence_thresh, boxes_thresh=boxes_thresh,
                                                                 render_lanes=render_lanes, render_boxes=render_boxes)

    if not left_match:
        print('[X] Left line failed.')
        tracks = [t for t in tracks if t.side != -1]
    if not right_match:
        print('[X] Right line failed.')
        tracks = [t for t in tracks if t.side != 1]

    if len(tracks) > 0:
        state.update_history(tracks)

        if (left_match or left_from_cache) and left_fit is not None:
            should_age = not left_match and left_from_cache
            state.confirm_left(should_age)
        else:
            state.age_left()

        if (right_match or right_from_cache) and right_fit is not None:
            should_age = not right_match and right_from_cache
            state.confirm_right(should_age)
        else:
            state.age_right()
    else:
        state.age_left()
        state.age_right()

    return (left_match, left_fit), (right_match, right_fit)
