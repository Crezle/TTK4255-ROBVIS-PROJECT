import cv2
import numpy as np
from datetime import datetime
import os
from image_processing import enhance_image, undistort
from hyperopt import fmin, tpe, hp, STATUS_OK
import ast

def _draw_corners(img, corners, ids, show_img=False, foldername="unnamed_detections"):
    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    if show_img:
        cv2.imshow(f'image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if not os.path.exists(f"data/apriltags/detections/{foldername}"):
        os.makedirs(f"data/apriltags/detections/{foldername}")
    cv2.imwrite(f"data/apriltags/detections/{foldername}/cornerdetections.jpg", img)

def configure_aruco_params(tag="APRILTAG_36h11", accuracy=None):
    if tag == "APRILTAG_36h11":
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    else:
        raise NotImplementedError("Only APRILTAG_36h11 is supported at the moment.")
    aruco_params = cv2.aruco.DetectorParameters()
    if accuracy is not None:
        aruco_params.polygonalApproxAccuracyRate = accuracy
    return aruco_dict, aruco_params

def train_parameters_apriltag(images_path, num_corners, tag="APRILTAG_36h11", save_params=False, evals=100):
    '''
    Takes in undistorted images and trains the parameters for the apriltag detection
    Args:
        images: list of images
        num_corners: number of corners to detect
        save_params: save the optimal parameters to a file
    Returns:
        optimal parameters
    '''
    image_paths = [os.path.join(images_path, img) for img in os.listdir(images_path) if os.path.splitext(img)[1] == '.jpg']
    images = []
    optimal_solutions = []
    for image_path in image_paths:
        undistorted_image = undistort(image_path)
        images.append(undistorted_image)

    def objective(params):
        total_loss = 0
        for img in images:
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            threshold, kernel_dim1, kernel_dim2, sigma1, sigma2, accuracy = int(params['threshold']), int(params['kernel_dim1']), int(params['kernel_dim2']), params['sigma1'], params['sigma2'], params['accuracy']

            grey_img = enhance_image(grey_img, threshold, kernel_dim1, kernel_dim2, sigma1, sigma2)
            aruco_dict, aruco_params = configure_aruco_params(tag=tag, accuracy=accuracy)
            temp_img = cv2.threshold(grey_img, threshold, 255, cv2.THRESH_BINARY)[1]

            (corners, _, _) = cv2.aruco.detectMarkers(temp_img, aruco_dict, parameters=aruco_params)

            loss = int(abs(len(corners) - num_corners))
            total_loss += loss
        
        if total_loss == 0:
            optimal_solutions.append(params)

        return {'loss': total_loss, 'status': STATUS_OK}
    
    space = {
        'threshold': hp.quniform('threshold', int(255*0.4), int(255*0.6), 1),
        'kernel_dim1': hp.choice('kernel_dim1', [3, 5, 7, 9]),
        'kernel_dim2': hp.choice('kernel_dim2', [3, 5, 7, 9]),
        'sigma1': hp.uniform('sigma1', 0, 10),
        'sigma2': hp.uniform('sigma2', 0, 10),
        'accuracy': hp.uniform('accuracy', 0.01, 0.5)
    }

    params = fmin(objective, space, algo=tpe.suggest, max_evals=evals)
    if save_params:
        if len(optimal_solutions) > 1:
            print("Multiple perfect solutions found, saving all of them.")
            with open(os.path.join(images_path, 'perfect_solutions.txt'), 'w') as f:
                for solution in optimal_solutions:
                    f.write(str(solution) + '\n')
        else:
            if len(optimal_solutions) == 1:
                print("Perfect solution found, saving it.")
                with open(os.path.join(images_path, 'perfect_solution.txt'), 'w') as f:
                    f.write(str(params) + '\n')
            else:
                print("No optimal solutions found, saving the best solution")
                with open(os.path.join(images_path, 'optimal_solution.txt'), 'w') as f:
                    f.write(str(params) + '\n')

    return params

def detect(img, tag="APRILTAG_36h11", aug_params=None, aug_params_path=None):
    
    # if aug_params is None:
    #     with open('data/apriltags/multiple_test18/optimal_solutions.txt', 'r') as f:
    #         params_str = f.read()
    #         aug_params = ast.literal_eval(params_str)
    
    timestamp = datetime.now().strftime('%d.%m.%Y_%H.%M.%S')
    
    
    if aug_params is not None or aug_params_path is not None:
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if aug_params_path is not None:
            try:
                with open(aug_params_path, 'r') as f:
                    params_str = f.read()
                    aug_params = ast.literal_eval(params_str)
            except FileNotFoundError:
                raise FileNotFoundError("No optimal solutions found, please run train_parameters_apriltag first.")
        
        # For some reason, params returns indices and not the value itself
        kernel_dims = [3, 5, 7, 9]  # The list of possible kernel dimensions
        aug_params['kernel_dim1'] = kernel_dims[int(aug_params['kernel_dim1'])]
        aug_params['kernel_dim2'] = kernel_dims[int(aug_params['kernel_dim2'])]
        threshold, kernel_dim1, kernel_dim2, sigma1, sigma2, accuracy = int(aug_params['threshold']), int(aug_params['kernel_dim1']), int(aug_params['kernel_dim2']), aug_params['sigma1'], aug_params['sigma2'], aug_params['accuracy']

        grey_img = enhance_image(grey_img, threshold, kernel_dim1, kernel_dim2, sigma1, sigma2)
        aruco_dict, aruco_params = configure_aruco_params(tag=tag, accuracy=accuracy)
        img = cv2.threshold(grey_img, threshold, 255, cv2.THRESH_BINARY)[1]
    else:
        aruco_dict, aruco_params = configure_aruco_params(tag=tag)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _draw_corners(img, corners, ids, foldername=f"corner_detections/{timestamp}")

    return corners, ids, rejected
