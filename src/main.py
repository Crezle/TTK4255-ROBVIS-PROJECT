import camera
import os
import image_processing
import apriltag
import cv2
import numpy as np

def main():
    #calibration_folder = 'checkerboard_6x9_2.0'
    world_img_path = 'data/apriltags/36h11/worldframe_to_cars/worldframe_4/0000.jpg'
    world_images_path = 'data/apriltags/36h11/worldframe_to_cars/worldframe_4/'
    
    cars_img_path = 'data/apriltags/36h11/worldframe_to_cars/cars_4/0000.jpg'
    cars_images_path = 'data/apriltags/36h11/worldframe_to_cars/cars_4/'
    #aug_params_path = 'data/apriltags/36h11/cars_fone4+6/optimal_solution.txt'
    
    # Find distortion coefficients
    #camera.calibrate(board_size=(6, 9),
    #                 square_size=0.02,
    #                 calib_img_foldername=calibration_folder)

    # Show calibration results
    # camera.show_calibration_results(results_foldername=calibration_folder)

    # Train apriltag parameters using images in same folder as subject image
    #params = apriltag.train_parameters_apriltag(images_path=world_images_path,
    #                                            num_corners=4,
    #                                            save_params=True,
    #                                            evals=200)

    # Undistort subject image
    # img = image_processing.undistort(img_path=img_path,
    #                                 calib_results_folder=calibration_folder,
    #                                 save_img=True)
    
    world_img = cv2.imread(world_img_path)

    # Detect apriltags in undistorted image
    corners, ids, _ = apriltag.detect(img=world_img)
    
    if len(ids) != 4:
        raise ValueError('Not all corners detected: ', len(ids))
    ids = ids.flatten().tolist()
    # Choose corners to use for homography
    # Order: Upper left, Upper right, Lower right, Lower left
    desired_id_order = [14, 19, 18, 8]
    
    id_to_corners = {id: corner for id, corner in zip(ids, corners)}
    
    ordered_corners = [id_to_corners[id] for id in desired_id_order]
    
    pts_src = np.array(ordered_corners, dtype='int32')[:, 0, 0]
    
    H, width, height = image_processing.findHomography(aspect_ratio=16/9, pts_src=pts_src)
    
    cars_img = cv2.imread(cars_img_path)
    
    warped_img = image_processing.homography_reprojection(cars_img, H, width, height, result_path='data/homography/36h11/worldframe_to_cars/', result_name='warped_image.jpg')

    params = apriltag.train_parameters_apriltag(images_path=cars_images_path,
                                                num_corners=4,
                                                save_params=True,
                                                evals=100)

    corners, ids, _ = apriltag.detect(img=warped_img, aug_params=params)
    

if __name__ == '__main__':
    main()