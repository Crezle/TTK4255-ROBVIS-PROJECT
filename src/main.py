import utils.camera as camera
import utils.detection as detection
import utils.transformation as transformation
from tools.termformatter import title

import argparse
import json

def main(args):
    run_all = args['run_all']
    calibration = args['calibration']
    undistortion = args['undistortion']
    board = args['board']
    homography = args['homography']
    cars = args['cars']

    K1 = None
    K2 = None
    dist_coeff = None
    std_int = None
    R = None
    t = None
    img_pts = None
    warped_img = None

    if not calibration['skip'] or run_all:
        K1, dist_coeff, std_int = camera.calibrate(calibration['dataset'],
                                                  calibration['board_size'],
                                                  calibration['square_size'],
                                                  calibration['rerun_detection'],
                                                  calibration['save_params'],
                                                  calibration['show_results'])

    if not undistortion['skip'] or run_all:
        camera.undistort(undistortion['img_set'],
                        undistortion['calibration_dataset'],
                        undistortion['crop'],
                        undistortion['save_imgs'],
                        undistortion['std_samples'],
                        run_all,
                        K1,
                        dist_coeff,
                        std_int)

    board_corners = [board['markers']['upleft'],
                         board['markers']['upright'],
                         board['markers']['downright'],
                         board['markers']['downleft']]
        
    board_corners = transformation.change_origin(board_corners,
                                                 board['origin'])

    if not board['skip'] or run_all:
        

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
                                         run_all,
                                         K1,
                                         dist_coeff)
        
    if not homography['skip'] or run_all:
        img_pts = transformation.world_to_img_corners(homography['img_set'],
                                                      homography['calibration_dataset'],
                                                      board_corners,
                                                      homography['border_size'],
                                                      homography['save_params'],
                                                      run_all,
                                                      R,
                                                      t,
                                                      K1,
                                                      dist_coeff)
        
        warped_img, K2 = transformation.warp_to_world(homography['img_set'],
                                                  homography['img_idx'],
                                                  homography['height'],
                                                  homography['board_size'],
                                                  homography['save_imgs'],
                                                  run_all,
                                                  img_pts)
    
    if not cars['skip']:
        detection.detect_cars(cars['warp_img_set'],
                              cars['img_idx'],
                              cars['calibration_dataset'],
                              cars['board_img_set'],
                              cars['num_cars'],
                              cars['save_imgs'],
                              cars['hsv_levels'],
                              cars['thresholds'],
                              cars['min_distance'],
                              run_all,
                              warped_img,
                              K1,
                              K2,
                              dist_coeff,
                              R,
                              t)
    
    title('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running the project.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--use_custom', type=bool, default=False, help='Choose if you want to use a custom configuration. Default is False.')
    use_custom = parser.parse_args().use_custom

    if use_custom:
        with open(f'configs/custom.json') as f:
            args = json.load(f)
    else:
        with open (f'configs/default.json') as f:
            args = json.load(f)

    main(args)
