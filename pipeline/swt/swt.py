"""
This is a highly experimental implementation of the Stroke Width Transform, described in
https://www.microsoft.com/en-us/research/publication/stroke-width-transform/.

The core idea of SWT, coming from an OCR domain, is that every stroke of a letter
has roughly similar thickness. This can be applied to lane detection as well, where
the stroke of the lane should roughly be constant and within expected bounds.

This code is copied and adjusted from my original project at https://github.com/sunsided/stroke-width-transform.
"""

from typing import TypeVar, NamedTuple, List, Optional
import cv2
import numpy as np


Image = np.ndarray
GradientImage = np.ndarray
Position = NamedTuple('Position', [('x', int), ('y', int)])
Stroke = NamedTuple('Stroke', [('x', int), ('y', int), ('width', float)])
Ray = List[Position]
Component = List[Position]
ImageOrValue = TypeVar('ImageOrValue', float, Image)
Gradients = NamedTuple('Gradients', [('x', GradientImage), ('y', GradientImage)])


def get_edges(im: Image, lo: float=175, hi: float=220, window: int=3) -> Image:
    """
    Detects edges in the image by applying a Canny edge detector.

    :param im: The image.
    :param lo: The lower threshold.
    :param hi: The higher threshold.
    :param window: The window (aperture) size.
    :return: The edges.
    """
    # OpenCV's Canny detector requires 8-bit inputs.
    im = (im * 255.).astype(np.uint8)
    edges = cv2.Canny(im, lo, hi, apertureSize=window)
    # Note that the output is either 255 for edges or 0 for other pixels.
    # Conversion to float wastes space, but makes the return value consistent
    # with the other methods.
    return edges.astype(np.float32) / 255.


def get_gradients(im: Image) -> Gradients:
    """
    Obtains the image gradients by means of a 3x3 Scharr filter.

    :param im: The image to process.
    :return: The image gradients.
    """
    # In 3x3, Scharr is a more correct choice than Sobel. For higher
    # dimensions, Sobel should be used.
    grad_x = cv2.Scharr(im, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(im, cv2.CV_64F, 0, 1)
    return Gradients(x=grad_x, y=grad_y)


def apply_swt(im: Image, edges: Image, gradients: Gradients, min_length: float=0, max_length: float=1000, edge_response: float=0) -> Image:
    """
    Applies the Stroke Width Transformation to the image.

    :param im: The grayscale image.
    :param edges: The edges of the image.
    :param gradients: The gradients of the image.
    :param min_length: The minimum length required for a stroke.
    :param max_length: The maximum length allowed for a stroke.
    :param edge_response: The minimum edge response required.
    :return: The transformed image.
    """
    # Prepare the output map.
    swt = np.squeeze(np.zeros_like(im))

    # For each pixel, let's obtain the normal direction of its gradient.
    # We add some epsilon to the norms to avoid numerical instabilities.
    norms = np.sqrt(gradients.x ** 2 + gradients.y ** 2) + .001
    inv_norms = 1. / norms
    directions = Gradients(x=gradients.x * inv_norms, y=gradients.y * inv_norms)

    # We keep track of all the rays found in the image.
    rays = []

    # Suppress edges with low activation
    norm_thresh_sq = edge_response ** 2

    def process_row(y: int, width: int, edges: np.ndarray, norms: np.ndarray):
        """
        Processes a single image row.
        :param y: The y coordinate.
        :param width: The width of the row.
        :param edges: The edge map.
        :param norms: The edge norms.
        :return: The list of detected rays.
        """
        rays = []
        for x in range(width):
            # Edges are either 0. or 1.
            if edges[y, x] < .5:
                continue
            # Suppress tiny edges
            if norms[y, x] < norm_thresh_sq:
                continue
            ray = swt_process_pixel(Position(x=x, y=y), edges, directions, out=swt,
                                    min_length=min_length, max_length=max_length)
            if ray:
                rays.append(ray)
        return rays

    # Find a pixel that lies on an edge.
    height, width = im.shape[0:2]
    for y in range(height):
        rays.extend(process_row(y, width, edges, norms))

    # Multiple rays may cross the same pixel in scenarios where edges are due to noise.
    # Each line crossing counts as a hit. By taking the inverse of each pixel, lines that
    # are robust are counted fully, whereas erratic lines are reduced in intensity.
    for ray in rays:
        for p in ray:
            swt[p.y, p.x] = 255 / swt[p.y, p.x]

    return swt


def swt_process_pixel(pos: Position, edges: Image, directions: Gradients, out: Optional[Image] = None,
                      min_length: float=0, max_length: float=float('inf')) -> Optional[Ray]:
    """
    Obtains the stroke width starting from the specified position.
    :param pos: The starting point
    :param edges: The edges.
    :param directions: The normalized gradients
    :param out: The output image.
    :param min_length: The minimum length required for a stroke.
    :param max_length: The maximum length allowed for a stroke.
    """
    if isinstance(pos, tuple):
        pos = Position(x=pos[0], y=pos[1])
    if isinstance(directions, tuple):
        directions = Gradients(x=directions[0], y=directions[1])

    # Keep track of the image dimensions for boundary tests.
    height, width = edges.shape[0:2]

    # For length testing, we avoid taking the square root.
    min_length_sq = min_length ** 2
    max_length_sq = max_length ** 2

    # Starting from the current pixel we will shoot a ray into the direction
    # of the pixel's gradient and keep track of all pixels in that direction
    # that still lie on an edge.
    ray = [pos]

    # Obtain the direction to step into
    input_dir_x = directions.x[pos.y, pos.x]
    input_dir_y = directions.y[pos.y, pos.x]
    if input_dir_x == 0 and input_dir_y == 0:
        return None

    # Since a line can be obtained bidirectional, we limit our search to positive X and Y
    # directions only. If we don't, we end up with twice the number of rays.
    if input_dir_x < 0:
        return None

    # Normalize the directions
    inv_norm = 1. / np.sqrt(input_dir_x ** 2 + input_dir_y ** 2)
    dir_x = input_dir_x * inv_norm
    dir_y = input_dir_y * inv_norm

    # Since some pixels have no gradient, normalization of the gradient
    # is a division by zero for them, resulting in NaN. These values
    # should not bother us since we explicitly tested for an edge before.
    assert not (np.isnan(dir_x) or np.isnan(dir_y)), 'dx: {}, dy: {}, 1/norm: {}'.format(input_dir_x, input_dir_y, inv_norm)

    # Traverse the pixels along the direction.
    prev_pos = Position(x=-1, y=-1)
    steps_taken = 0
    while True:
        # Advance to the next pixel on the line.
        steps_taken += 1
        cur_x = int(np.floor(pos.x + dir_x * steps_taken))
        cur_y = int(np.floor(pos.y + dir_y * steps_taken))
        cur_pos = Position(x=cur_x, y=cur_y)
        if cur_pos == prev_pos:
            continue
        prev_pos = Position(x=cur_x, y=cur_y)
        # If we reach the edge of the image without crossing a stroke edge,
        # we discard the result.
        if not ((0 <= cur_x < width) and (0 <= cur_y < height)):
            return None
        # We determine the "width" of the stroke.
        stroke_width_sq = (cur_x - pos.x) * (cur_x - pos.x) + (cur_y - pos.y) * (cur_y - pos.y)
        if stroke_width_sq > max_length_sq:
            return None
        # The point is either on the line or the end of it, so we register it.
        ray.append(cur_pos)
        # If that pixel is not an edge, we are still on the line and
        # need to continue scanning.
        if edges[cur_y, cur_x] < .5:  # TODO: Test for image boundaries here
            continue
        # If this edge is pointed in a direction approximately opposite of the
        # one we started in, it is approximately parallel. This means we
        # just found the other side of the stroke.
        # The original paper suggests the gradients need to be opposite +/- PI/6 (30 degree).
        # Since the dot product is the cosine of the enclosed angle and
        # cos(pi/6) = 0.8660254037844387, we can discard all values that exceed
        # this threshold.
        cur_dir_x = directions.x[cur_y, cur_x]
        cur_dir_y = directions.y[cur_y, cur_x]
        dot_product = dir_x * cur_dir_x + dir_y * cur_dir_y
        if dot_product >= -0.866:
            return None
        if stroke_width_sq < min_length_sq:
            return None
        # In regular SWT, we would set this pixel to the currently longest stroke.
        # Since we don't really care for stroke widths later, we just add up
        # the number of times a stroke crossed this pixel.
        if out is not None:
            for p in ray:
                out[p.y, p.x] += 1
        return ray

    # noinspection PyUnreachableCode
    assert False, 'This code cannot be reached.'
