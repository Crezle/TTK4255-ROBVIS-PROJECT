import utils.camera as camera
import utils.detection as detection
import utils.transformation as transformation
from tools.termformatter import title

import argparse
import json
import os
import time
import shutil

class Config:
    def __init__(self, config_file):

        self.json_file = config_file

        with open(config_file) as f:
            config = json.load(f)
        self.calibration = config['calibration']
        self.undistortion = config['undistortion']
        self.board = config['board']
        self.homography = config['homography']
        self.cars = config['cars']

def main(config: Config):

    timestamp = time.strftime('%d.%m.%y-%H.%M.%S')
    output_dir = os.path.join('output', timestamp)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    shutil.copy(config.json_file, os.path.join(output_dir, 'config.json'))

    title(f'OUTPUT DIRECTORY: {output_dir}')
    

    camera.calibrate(config.calibration,
                     output_dir)

    camera.undistort(config.undistortion,
                     output_dir)

    detection.estimate_pose(config.board,
                            output_dir)

    transformation.project_world_to_img(config.homography,
                                        config.board,
                                        output_dir)
        
    transformation.perspective(config.homography,
                               output_dir)
    
    detection.detect_cars(config.cars,
                          output_dir)
    
    title('FINISHED, ALL RESULTS SAVED IN OUTPUT DIRECTORY.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running the project.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--use_custom', type=bool, default=False, help='Choose if you want to use a custom configuration. Default is False.')
    use_custom = parser.parse_args().use_custom

    config_file = 'configs/custom.json' if use_custom else 'configs/default.json'
    config = Config(config_file)

    main(config)
