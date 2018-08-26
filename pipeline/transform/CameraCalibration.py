import glob
import os
import pickle
import cv2
import numpy as np
from typing import Optional, Tuple, List, NamedTuple

ObjectPointList = List[np.ndarray]
ImagePointList = List[np.ndarray]
ImageSize = NamedTuple('ImageSize', [('width', int), ('height', int)])
Rectangle = Tuple[int, int, int, int]


class CameraCalibration:
    """
    Implements image distortion correction using a previously obtained camera model.
    """

    @staticmethod
    def from_pickle(path: str) -> 'CameraCalibration':
        with open(path, 'rb') as f:
            params = pickle.load(f)
            return CameraCalibration(params['coeffs'], params['mtx'], params['roi'],
                                     params['refined_mtx'], params['refined_roi'])

    @staticmethod
    def calibrate_from_files(image_path: str, inner_corner_dims: Tuple[int, int]) -> 'CameraCalibration':
        file_pattern = os.path.join(image_path, "calibration*.jpg")
        image_paths = glob.glob(file_pattern)
        obj_points, img_points, img_size = CameraCalibration.__find_chessboard_patterns_in_files(
            image_paths, inner_corner_dims)
        roi = (0, 0, img_size.width, img_size.height)
        _, mtx, coeffs, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
        refined_mtx, refined_roi = cv2.getOptimalNewCameraMatrix(mtx, coeffs, img_size, alpha=1, newImgSize=img_size)
        return CameraCalibration(coeffs, mtx, roi, refined_mtx, refined_roi)

    @staticmethod
    def __create_object_coordinates(inner_corner_dims: Tuple[int, int]) -> np.ndarray:
        """
        Generates expected (object space) coordinates for a checkerboard calibration target of
        the specified dimensions.
        :param inner_corner_dims: The number of inner corners in x and y direction.
        :return: The corner coordinates in object space.
        """
        num_coords = np.prod(inner_corner_dims)
        points = np.zeros((num_coords, 3), np.float32)
        points[:, :2] = np.mgrid[0:inner_corner_dims[0], 0:inner_corner_dims[1]].T.reshape(-1, 2)
        return points

    @staticmethod
    def __find_chessboard_patterns_in_files(paths: List[str], inner_corner_dims: Tuple[int, int]) \
            -> Tuple[ObjectPointList, ImagePointList, ImageSize]:
        """
        Finds all chessboard patterns in the specified image files.
        :param paths: The list of image file paths.
        :param inner_corner_dims: The number of inner corners in x and y direction.
        :return: A tuple of corner coordinates in object space (3D) and corner coordinates in image space (2D).
        """
        assert len(paths) > 0
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane
        image_sizes = None  # type: ImageSize
        objp = CameraCalibration.__create_object_coordinates(inner_corner_dims)
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            img_size = ImageSize(width=img.shape[1], height=img.shape[0])
            if image_sizes is not None:
                # Some of the test images have slightly deviating sizes, such as a width of 1281 instead of 1280.
                width_ok = np.abs(img_size.width - image_sizes.width) <= 1
                height_ok = np.abs(img_size.height - image_sizes.height) <= 1
                assert width_ok and height_ok
            image_sizes = img_size

            corners = CameraCalibration.__find_chessboard_patterns_in_image(img, inner_corner_dims)
            if corners is None:
                continue
            obj_points.append(objp)
            img_points.append(corners)
        return obj_points, img_points, image_sizes

    @staticmethod
    def __find_chessboard_patterns_in_image(img: np.ndarray, inner_corner_dims: Tuple[int, int]) \
            -> Optional[np.ndarray]:
        """
        Finds the chessboard corners in the specified image.
        :param img: The image.
        :param inner_corner_dims: The number of inner corners in x and y direction.
        :return: An array of the coordinates if the pattern was found, None otherwise.
        """
        if len(img.shape) > 2 and img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img, inner_corner_dims, None)
        return corners if ret else None

    def __init__(self, coeffs: np.ndarray, mtx: np.ndarray, roi: Rectangle,
                 refined_mtx: Optional[np.ndarray] = None, refined_roi: Optional[Rectangle] = None):
        """
        Initializes this instance.
        :param coeffs: The distortion coefficients.
        :param mtx: The camera matrix.
        :param roi: The image region of interest.
        :param refined_mtx: The refined camera matrix.
        :param refined_roi: The region of interest of the refined matrix.
        """
        self._mtx = mtx
        self._refined_mtx = refined_mtx if refined_mtx is not None else mtx
        self._coeffs = coeffs
        self._roi = roi
        self._refined_roi = refined_roi if refined_roi is not None else roi

    def undistort(self, img: np.ndarray, refined: bool=True) -> Tuple[np.ndarray, List[int]]:
        """
        Un-distorts the specified image.
        :param img: The distorted image.
        :param refined: True if all source pixels should be in the image, False if only valid pixels should be.
        :return: The undistorted image and the image's region of interest of valid pixels.
        """
        mtx = self._mtx
        refined_mtx = self._mtx
        roi = self._roi
        if refined and self._refined_mtx is not None:
            refined_mtx = self._refined_mtx
            roi = self._refined_roi
        # noinspection PyTypeChecker
        return cv2.undistort(img, mtx, self._coeffs, None, refined_mtx), roi

    def pickle(self, path: str) -> None:
        """
        Saves the current parameters in pickled form.
        :param path: The pickle file.
        :return:
        """
        params = {
            'coeffs': self._coeffs,
            'mtx': self._mtx,
            'roi': self._roi,
            'refined_mtx': self._refined_mtx,
            'refined_roi': self._refined_roi
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)


def __main():
    # Perform calibration once
    path = os.path.join('..', 'camera_cal')
    cc = CameraCalibration.calibrate_from_files(path, (9, 6))

    # Store the parameters in a pickle
    test_pickle = 'calibration-test.pkl'
    if os.path.exists(test_pickle):
        os.unlink(test_pickle)
    cc.pickle(test_pickle)

    # Load from pickle
    cc2 = CameraCalibration.from_pickle(test_pickle)
    os.unlink(test_pickle)

    # Ensure the results are identical
    # noinspection PyProtectedMember
    assert np.all(np.equal(cc2._mtx, cc._mtx))
    # noinspection PyProtectedMember
    assert np.all(np.equal(cc2._refined_mtx, cc._refined_mtx))
    # noinspection PyProtectedMember
    assert np.all(np.equal(cc2._coeffs, cc._coeffs))
    # noinspection PyProtectedMember
    assert np.all(np.equal(cc2._roi, cc._roi))

    # Let's run some tests.
    cv2.namedWindow('Images', cv2.WINDOW_NORMAL)

    sample_path = os.path.join('..', '..', 'test_images')
    files = glob.glob(os.path.join(sample_path, '*.jpg'))
    files.extend(glob.glob(os.path.join(path, 'calibration*.jpg')))

    for file in files:
        img = cv2.imread(file)

        # Note that the images need to be the exact same size that was seen during calibration.
        undistorted, roi = cc2.undistort(img)

        top_left = roi[:2]
        bottom_right = (roi[2] + roi[0], roi[3] + roi[1])
        cv2.rectangle(undistorted, top_left, bottom_right, color=(0, 0, 255), thickness=2)

        img = np.hstack([img, undistorted])
        cv2.imshow('Images', img)
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    __main()
