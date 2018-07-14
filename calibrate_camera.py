"""
Obtains camera calibration data from images.
"""

from argparse import ArgumentParser
from pipeline import CameraCalibration


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='camera_cal',
                        metavar='DIR', dest='path',
                        help='Path to the camera calibration images (calibrationN.jpg)')
    parser.add_argument('-c', '--calibration_file', type=str, default='calibration.pkl',
                        metavar='FILE', dest='calibration_file',
                        help='Path to the output calibration pickle file')

    group = parser.add_argument_group(title='Calibration target')
    group.add_argument('-x', '--corners_x', type=int, default=9,
                        metavar='NX', dest='target_width',
                        help='Number of inner corners in X direction.')
    group.add_argument('-y', '--corners_y', type=int, default=6,
                       metavar='NY', dest='target_height',
                       help='Number of inner corners in Y direction.')
    args = parser.parse_args()

    calibration_dims = args.target_width, args.target_height
    cc = CameraCalibration.calibrate_from_files(args.path, calibration_dims)
    cc.pickle(args.calibration_file)


if __name__ == '__main__':
    main()
