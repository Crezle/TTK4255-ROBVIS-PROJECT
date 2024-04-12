import utils.camera as camera
import utils.detection as detection

import argparse
import json

def main(args):
    calibration = args['calibration']
    undistortion = args['undistortion']
    board = args['board']

    if not calibration['skip']:
        camera.calibrate(calibration['dataset'],
                        calibration['board_size'],
                        calibration['square_size'],
                        calibration['rerun_detection'],
                        calibration['save_params'],
                        calibration['show_results'])

    if not undistortion['skip']:
        camera.undistort(undistortion['img_set'],
                        undistortion['coeffs'],
                        undistortion['crop'],
                        undistortion['save_imgs'],
                        undistortion['std_samples'])

    if not board['skip']:
        board_corners = [board['markers']['upleft'],
                         board['markers']['upright'],
                         board['markers']['downright'],
                         board['markers']['downleft']]

        detection.detect_board(board['dictionary'],
                               board['img_set'],
                               board['img_idx'],
                               board_corners,
                               board['ids'],
                               board['refind'],
                               board['coeffs'],
                               board['save_imgs'],
                               board['show_rejected'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running the project.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', type=str, default='default', help='Options: [default, custom]')
    config_name = parser.parse_args().config
    json_path = f'configs/{config_name}.json'

    with open(json_path) as f:
        args = json.load(f)

    main(args)
