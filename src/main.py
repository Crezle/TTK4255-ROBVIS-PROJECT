import camera
import image_processing
import apriltag
import cv2

def main():
    calibration_folder = 'checkerboard_6x9_2.0'
    img_path = 'data/apriltags/36h11/multiple_test18/0001.jpg'
    images_path = 'data/apriltags/36h11/multiple_test18'
    
    camera.calibrate(board_size=(6, 9),
                     square_size=0.02,
                     calib_img_foldername=calibration_folder)

    camera.show_calibration_results(results_foldername=calibration_folder)

    params = apriltag.train_parameters_apriltag(images_path=images_path,
                                                num_corners=18,
                                                save_params=True,
                                                evals=150)

    img = image_processing.undistort(img_path=img_path,
                                     calib_results_folder=calibration_folder,
                                     save_img=True)

    apriltag.detect(img=img,
                    params=params)


if __name__ == '__main__':
    main()