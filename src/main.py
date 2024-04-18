import utils.camera as camera
import utils.detection as detection
import utils.transformation as transformation
from tools.termformatter import title

import argparse
import json

class Config:
    def __init__(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        self.run_all = config['run_all']
        self.calibration = config['calibration']
        self.undistortion = config['undistortion']
        self.board = config['board']
        self.homography = config['homography']
        self.cars = config['cars']

def main(config: Config):

    K1, dist_coeff, std_int = camera.calibrate(config.calibration)

    camera.undistort(config.undistortion,
                     config.run_all,
                     K1,
                     dist_coeff,
                     std_int)

    R, t = detection.detect_board(config.board,
                                  config.run_all,
                                  K1,
                                  dist_coeff)

    img_pts = transformation.world_to_img_corners(config.homography,
                                                  config.board,
                                                  config.run_all,
                                                  R,
                                                  t,
                                                  K1,
                                                  dist_coeff)
        
    warped_img, K2 = transformation.warp_to_world(config.homography,
                                                  config.run_all,
                                                  img_pts)
    
    detection.detect_cars(config.cars,
                          config.run_all,
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

    config_file = 'configs/custom.json' if use_custom else 'configs/default.json'
    config = Config(config_file)

    main(config)
