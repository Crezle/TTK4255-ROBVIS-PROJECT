import camera
import os
import image_processing
import apriltag
import cv2
import numpy as np

def main():
    calibration_folder = 'checkerboard_6x9_2.0'
    img_path = 'data/apriltags/36h11/homography_test4/0000.jpg'
    images_path = 'data/apriltags/36h11/homography_test4/'
    aug_params_path = 'data/apriltags/36h11/homography_test4/optimal_solution.txt'
    
    # Find distortion coefficients
    camera.calibrate(board_size=(6, 9),
                     square_size=0.02,
                     calib_img_foldername=calibration_folder)

    # Show calibration results
    camera.show_calibration_results(results_foldername=calibration_folder)

    # Train apriltag parameters using images in same folder as subject image
    params = apriltag.train_parameters_apriltag(images_path=images_path,
                                                num_corners=4,
                                                save_params=True,
                                                evals=500)

    # Undistort subject image
    img = image_processing.undistort(img_path=img_path,
                                     calib_results_folder=calibration_folder,
                                     save_img=True)

    # Detect apriltags in undistorted image
    corners, _, _ = apriltag.detect(img=img, aug_params=params)

    pts_src = np.array(corners, dtype=np.int32)[:, 0, 0]
    pts_dst = np.array([[0, 0], [1280, 0], [0, 720], [1280, 720]])
    
    h, status = cv2.findHomography(pts_src, pts_dst)
    img_warped = cv2.warpPerspective(img, h, (1280, 720))

    cv2.imshow('Source Image', img)
    cv2.imshow('Warped Source Image', img_warped)

    if os.getenv('GITHUB_ACTIONS') != 'true':
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not os.path.exists('data/homography/36h11/homography_test4'):
        os.makedirs('data/homography/36h11/homography_test4')
    cv2.imwrite('data/homography/36h11/homography_test4/warped.jpg', img_warped)

if __name__ == '__main__':
    main()