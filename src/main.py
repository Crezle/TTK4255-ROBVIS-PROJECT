import camera
import image_processing
import apriltag

def main():
    img_path = 'data/apriltags/multiple_test/0001.jpg'
    calibration_folder = 'chessboard'

    camera.calibrate(rerun_detection=False, calib_img_folder=calibration_folder)
    img = image_processing.undistort(img_path=img_path, calib_results_folder=calibration_folder, save_img=False)
    img = image_processing.enhance_image(img)
    apriltag.detect(num_corners=18, img=img)

if __name__ == '__main__':
    main()