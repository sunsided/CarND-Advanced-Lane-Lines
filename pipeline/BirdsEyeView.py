import cv2
import numpy as np
from typing import NamedTuple


Point = NamedTuple('Point', [('x', float), ('y', float)])
Size = NamedTuple('Size', [('width', int), ('height', int)])


class ImageSection:
    """
    Describes a section of an image.
    """

    def __init__(self, top_left: Point, top_right: Point, bottom_right: Point, bottom_left: Point):
        self._top_left = top_left
        self._top_right = top_right
        self._bottom_right = bottom_right
        self._bottom_left = bottom_left

    @property
    def height(self) -> float:
        return ((self._bottom_left.y - self._top_left.y) + (self._bottom_right.y - self._top_right.y)) * 0.5

    @property
    def top_width(self) -> float:
        return self._top_right.x - self._top_left.x

    @property
    def bottom_width(self) -> float:
        return self._bottom_right.x - self._bottom_left.x

    @property
    def polygon(self):
        return np.array([self._top_left,
                         self._top_right,
                         self._bottom_right,
                         self._bottom_left])


class BirdsEyeView:
    """
    Transformation from a perspective view to an orthogonal bird's eye view.
    """

    def __init__(self, section: ImageSection, section_width: float, section_height: float):
        self._base_width = 100
        self._section = section
        self._section_width = section_width
        self._section_height = section_height
        self._pixels_per_unit = 0
        self._units_per_pixel = 0
        self._projected_height = 0
        self._M = None  # type: np.ndarray
        self.update()

    def pixel_to_unit(self, pt: Point) -> Point:
        """
        Converts the specified point in the warped image to physical units.
        :param pt: The pixel coordinate
        :return: The coordinate in physical units
        """
        # TODO: Allow for different scales in X and Y
        return Point(
            x=pt.x * self._units_per_pixel,
            y=pt.y * self._units_per_pixel
        )

    def update(self) -> None:
        """
        Calculates a new projection matrix after parameter changes.
        """
        src = self._section.polygon
        section_width = self._section_width
        section_height = self._section_height
        self._pixels_per_unit = self._base_width / section_width
        self._units_per_pixel = 1.0 / self._pixels_per_unit
        dst = np.array([
            [section_width * 2, 0],
            [section_width * 3, 0],
            [section_width * 3, section_height],
            [section_width * 2, section_height]
        ]) * self._pixels_per_unit

        self._projected_height = np.ceil(dst.max()).astype(int)
        self._M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        self._iM = cv2.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32))

    def warp(self, img: np.ndarray) -> np.ndarray:
        """
        Warps an image into bird's eye view.
        :param img: The image to warp
        :return: The warped image.
        """
        warped = cv2.warpPerspective(img, self._M, (5 * self._base_width, self._projected_height))
        return warped

    def unwarp(self, img: np.ndarray, size: Size) -> np.ndarray:
        """
        Un-warps an image from bird's eye view.
        :param img: The image to unwarp
        :param size: The target image size
        :return: The unwarped image.
        """
        return cv2.warpPerspective(img, self._iM, size)


def __main():
    import os
    from CameraCalibration import CameraCalibration

    cc = CameraCalibration.from_pickle(os.path.join('..', 'calibration.pkl'))
    path = os.path.join('..', 'test_images', 'straight_lines1.jpg')
    img = cv2.imread(path)
    assert img is not None

    img, _ = cc.undistort(img, False)

    section = ImageSection(
        top_left=Point(x=580, y=461.75),
        top_right=Point(x=702, y=461.75),
        bottom_right=Point(x=1013, y=660),
        bottom_left=Point(x=290, y=660),
    )

    bev = BirdsEyeView(section,
                       section_width=3.6576,        # one lane width in meters
                       section_height=2 * 13.8826)  # two dash distances in meters

    shape = Size(width=img.shape[1], height=img.shape[0])

    warped = bev.warp(img)
    unwarped = bev.unwarp(warped, shape)

    cv2.namedWindow('warped', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('unwarped', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('warped', warped)
    cv2.imshow('unwarped', unwarped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    __main()
