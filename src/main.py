import utils.camera as camera
import utils.detection as detection
import utils.transformation as transformation

import argparse
import json

def main(args):
    run_all = args['run_all']
    calibration = args['calibration']
    undistortion = args['undistortion']
    board = args['board']
    cars = args['cars']
    homography = args['homography']

    K = None
    dist_coeff = None
    std_int = None
    R = None
    t = None

    if not calibration['skip'] or run_all:
        K, dist_coeff, std_int = camera.calibrate(calibration['dataset'],
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
                        K,
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
                                         K,
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
                                                      K,
                                                      dist_coeff)
        
        warped_img = transformation.warp_to_world(homography['img_set'],
                                                  homography['img_idx'],
                                                  homography['height'],
                                                  homography['save_imgs'],
                                                  run_all,
                                                  img_pts)
    
    if not cars['skip']:
        detection.detect_cars(cars['img_set'],
                              cars['img_idx'],
                              cars['calibration_dataset'],
                              cars['board_img_set'],
                              cars['num_cars'],
                              cars['save_imgs'],
                              cars['red_threshold'],
                              cars['detector_type'],
                              cars['min_distance'],
                              run_all,
                              warped_img,
                              K,
                              dist_coeff,
                              R,
                              t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running the project.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', type=str, default='default', help='Options: [default, custom]')
    config_name = parser.parse_args().config

    json_path = f'configs/{config_name}.json'

    with open(json_path) as f:
        args = json.load(f)

    main(args)
