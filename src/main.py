import utils.camera as camera
import utils.detection as detection

import argparse
import json

def main(args, passthrough):
    calibration = args['calibration']
    undistortion = args['undistortion']
    board = args['board']
    cars = args['cars']
    
    K = None
    dist_coeff = None
    std_int = None
    R = None
    t = None

    if not calibration['skip'] or passthrough:
        K, dist_coeff, std_int = camera.calibrate(calibration['dataset'],
                                                  calibration['board_size'],
                                                  calibration['square_size'],
                                                  calibration['rerun_detection'],
                                                  calibration['save_params'],
                                                  calibration['show_results'])

    if not undistortion['skip'] or passthrough:
        camera.undistort(undistortion['img_set'],
                        undistortion['calibration_dataset'],
                        undistortion['crop'],
                        undistortion['save_imgs'],
                        undistortion['std_samples'],
                        passthrough,
                        K,
                        dist_coeff,
                        std_int)

    if not board['skip'] or passthrough:
        board_corners = [board['markers']['upleft'],
                         board['markers']['upright'],
                         board['markers']['downright'],
                         board['markers']['downleft']]

        R, t = detection.detect_board(board['dictionary'],
                                         board['img_set'],
                                         board['img_idx'],
                                         board_corners,
                                         board['ids'],
                                         board['refined'],
                                         board['calibration_dataset'],
                                         board['save_imgs'],
                                         board['save_params'],
                                         board['save_rejected'],
                                         passthrough,
                                         K,
                                         dist_coeff)
    
    if not cars['skip']:
        detection.detect_cars(cars['img_set'],
                              cars['img_idx'],
                              cars['calibration_dataset'],
                              cars['board_img_set'],
                              cars['num_cars'],
                              cars['save_imgs'],
                              cars['detector_type'],
                              passthrough,
                              K,
                              dist_coeff,
                              R,
                              t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running the project.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', type=str, default='default', help='Options: [default, custom]')
    parser.add_argument('--passthrough', type=bool, default=True, help='Whether to keep using values from previous functions.')
    config_name = parser.parse_args().config
    passthrough = parser.parse_args().passthrough

    json_path = f'configs/{config_name}.json'

    with open(json_path) as f:
        args = json.load(f)

    main(args, passthrough)
