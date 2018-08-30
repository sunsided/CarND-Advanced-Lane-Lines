class LaneDetectionParams:
    def __init__(self, mx: float, my: float,
                 lane_max_age: int=35,
                 lane_max_history: int=35,
                 search_px_lo: int=0.05,
                 search_box_width: int = 30,
                 search_box_height: int = 15,
                 search_max_strikes: int=15,
                 search_height: float=0.7,
                 search_local_fit: float=0.3,
                 search_local_fit_hist: int=8,
                 fit_quality_allowed_deviation: float=5,
                 fit_quality_decay: float=0.09,
                 fit_history_decay: float = 4,
                 confidence_thresh: float=.5,
                 confidence_thresh_cached: float=.5,
                 boxes_thresh: int=15,
                 boxes_thresh_cached: int=4,
                 validation_min_box_support: float=0.6,
                 validation_px_lo: float=0.05,
                 validation_px_hi: float=0.8,
                 seed_height: float=0.6,
                 render_lanes: bool=False, render_boxes: bool=False):
        """
        Initializes the lane detection parameters.
        :param mx: Conversion factor from horizontal pixels to meters.
        :param my: Conversion factor from vertical pixels to meters.
        :param lane_max_age: The maximum age of a line match / track before it will be forgotten.
        :param lane_max_history: The maximum number of line matches / tracks to keep for smoothing.
        :param search_px_lo: The (normalized) minimum fill amount to consider a search box "on track".
        :param search_box_width: The box width for searching lane lines.
        :param search_box_height: The box height for searching lane lines.
        :param search_max_strikes: The maximum number of box matches without support before terminating the search.
        :param search_local_fit: The strength of a local fit weight during line searching (0..1).
        :param search_local_fit_hist: The minimum number of previous boxes required for a local fit.
        :param fit_quality_allowed_deviation: The number of pixels search rectangles are allowed
                                              to deviate from the fit before quality decays.
        :param fit_quality_decay: The factor by which fit quality decays exponentially.
        :param fit_history_decay: The factor by which historical fit importance decays exponentially.
        :param confidence_thresh:
        :param confidence_thresh_cached:
        :param boxes_thresh: The minimum number of pixels required to accept a box.
        :param boxes_thresh_cached: The minimum number of pixels required to accept a box if the box is from cache.
        :param render_lanes: Whether lanes should be rendered.
        :param render_boxes: Whether boxes should be rendered.
        :param validation_min_box_support: The fraction of boxes required to support an existing fit during validation.
        :param validation_px_lo: The minimum fill amount for a box to be considered "on track".
        :param validation_px_hi: The maximum fill amount for a box to be considered "not flooded".
        :param seed_height: The fraction of the warped image height so use for the seed histogram.
        :param search_height: The fraction of the warped image height so use for line searching.
        """
        self.mx = mx
        self.my = my
        self.lane_max_history = lane_max_history
        self.lane_max_age = lane_max_age
        self.search_px_lo = search_px_lo
        self.search_box_width = search_box_width
        self.search_box_height = search_box_height
        self.search_max_strikes = search_max_strikes
        self.search_height = search_height
        self.search_local_fit = search_local_fit
        self.search_local_fit_hist = search_local_fit_hist
        self.fit_quality_allowed_deviation = fit_quality_allowed_deviation
        self.fit_quality_decay = fit_quality_decay
        self.fit_history_decay = fit_history_decay
        self.confidence_thresh = confidence_thresh
        self.confidence_thresh_cached = confidence_thresh_cached
        self.boxes_thresh = boxes_thresh
        self.boxes_thresh_cached = boxes_thresh_cached
        self.render_lanes = render_lanes
        self.render_boxes = render_boxes
        self.validation_min_box_support = validation_min_box_support
        self.validation_px_lo = validation_px_lo
        self.validation_px_hi = validation_px_hi
        self.seed_height = seed_height

