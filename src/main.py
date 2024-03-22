import camera
import image_processing
import apriltag
import cv2

def main():
    calibration_folder = 'chessboard'
    img_path = 'data/apriltags/multiple_test18/0001.jpg'
    images_path = 'data/apriltags/multiple_test18'
    
    # camera.calibrate(rerun_detection=False, calib_img_folder=calibration_folder)
    params = apriltag.train_parameters_apriltag(images_path=images_path, num_corners=18, save_params=True, evals=150)
    img = image_processing.undistort(img_path=img_path, calib_results_folder=calibration_folder, save_img=False)
    apriltag.detect(img=img, params=params)

if __name__ == '__main__':
    main()