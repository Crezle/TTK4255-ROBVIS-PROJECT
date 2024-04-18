import os
import cv2
import glob
from cv2 import aruco
import numpy as np
from tools.termformatter import title
import json

def _detect_markers(detector: aruco.ArucoDetector, 
                    img: np.ndarray, 
                    board: aruco.Board, 
                    refined: bool, 
                    K: np.ndarray, 
                    dist_coeff: np.ndarray):

    corners, ids, rejected = detector.detectMarkers(img)
    
    if refined:
        print('Refining detected markers...', end='')
        if K is None or dist_coeff is None:
            corners, ids, rejected, recoveredIdx = detector.refineDetectedMarkers(img, board, corners, ids, rejected)
        else:
            corners, ids, rejected, recoveredIdx = detector.refineDetectedMarkers(img, board, corners, ids, rejected, K, dist_coeff)
    
        if recoveredIdx is not None:
            print(f'Recovered {len(recoveredIdx)} markers.')
        else:
            print('No markers recovered.')
    
    return corners, ids, rejected

def detect_board(dictionary: str,
                 img_set: str,
                 img_idx: int,
                 board_corners: list[list[float]],
                 ids: list[int],
                 refined: bool,
                 calibration_dataset: str,
                 save_imgs: bool,
                 save_rejected: bool,
                 save_params: bool,
                 run_all: bool,
                 K: np.ndarray = None,
                 dist_coeff: np.ndarray = None):
    """
    Args:
    dictionary: The name of the ArUco dictionary to use.
    img_set: The name of the image set to use. Must be a folder in 'data/detection/images'.
    img_idx: The index of the image in the image set to use.
    board_corners: The corners of the board in the image. Should be a list of 4 points, each with 2 or 3 coordinates.
    ids: The IDs of the markers on the board.
    refined: Whether to refine the detected markers.
    calibration_dataset: The name of the calibration baseline to use. Must be a folder in 'data/calibration/results'.
    save_imgs: Whether to save the resulting images.
    save_rejected: Whether to save the rejected markers.
    
    Returns:
    R: The 3x3 rotation matrix of the board.
    t: The translation vector of the board.
    """
    
    title("BOARD DETECTION")

    img_path = f'data/detection/images/{img_set}/*.jpg'
    out_path = f'data/detection/results/{img_set}'
    calibration_dataset_path = f'data/calibration/results/{calibration_dataset}'

    if not run_all:
        try:
            K           = np.loadtxt(os.path.join(calibration_dataset_path, 'K.txt'))
            dist_coeff  = np.loadtxt(os.path.join(calibration_dataset_path, 'dist_coeff.txt'))
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find calibration dataset at {calibration_dataset_path}.')

    dist_coeff = dist_coeff.flatten()
    assert K.shape == (3, 3), 'K must be a 3x3 matrix.'
    assert dist_coeff.shape == (5,), 'dist_coeff must be a 5-element vector.'

    #TODO: Make this configurable
    aruco_params = aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    aruco_params.adaptiveThreshConstant = 10
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dictionary))
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    assert aruco_params is not None, 'Could not create ArUco parameters.'
    assert aruco_dict is not None, 'Could not create ArUco dictionary.'
    assert detector is not None, 'Could not create ArUco detector.'

    print('Intializing board detection')
    images = sorted(glob.glob(img_path))
    assert len(images) > 0, 'No images found in the specified path.'
    assert img_idx < len(images), 'Image index out of bounds.'
    
    img = cv2.imread(images[img_idx])

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    board_ids = np.array(ids, dtype='int32')
    board_corners = np.array(board_corners, dtype='float32')

    if board_corners.shape[2] == 2:
        print('Corners given in 2D-coordinates, adding z=0 to board corners')
        board_corners = np.concatenate((board_corners, np.zeros_like(board_corners[:, :, 0:1])), axis=2)
    elif board_corners.shape[2] != 3:
        raise ValueError('Board corners must have shape (N, 4, 2) or (N, 4, 3)')

    board = aruco.Board(board_corners, aruco_dict, board_ids)

    corners, ids, rejected = _detect_markers(detector, img, board, refined, K, dist_coeff)
    
    if ids is not None:
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        print('Estimating board pose...', end='')
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeff)
        except cv2.error as e:
            Warning('Could not estimate board pose.')
        else:
            print('Board pose estimated successfully.')
            num_det_markers = len(obj_pts) / 4

    else:
        print('No markers detected, cannot estimate board pose.')
        num_det_markers = 0
        rvec = None
        tvec = None
        Warning('Board pose not estimated, returning None for R and tvec.')

    if save_imgs:
        out_img = None

        if ids is not None:
            out_img = aruco.drawDetectedMarkers(img, corners, ids)
            
        if save_rejected and rejected is not None:
            rej_img = aruco.drawDetectedMarkers(img, rejected, borderColor=(100, 0, 255))
            cv2.imwrite(os.path.join(out_path, 'board_rejected.png'), rej_img)  
        
        if num_det_markers > 0:
            out_img = cv2.drawFrameAxes(out_img, K, dist_coeff, rvec, tvec, 3, 3)
        else:
            print('No markers detected, cannot draw axes.')
        
        if out_img is not None:
            cv2.imwrite(os.path.join(out_path, 'board_result.png'), out_img)
        else:
            print('No image to save.')

    # Convert rotation vector to rotation matrix
    R = cv2.Rodrigues(rvec)[0] if rvec is not None else None
    
    if save_params and R is not None and tvec is not None:
        np.savetxt(os.path.join(out_path, 'R.txt'), R)
        np.savetxt(os.path.join(out_path, 't.txt'), tvec)
    
    print('Returning extrinsics0')
    
    return R, tvec

def detect_cars(img_set: str,
                img_idx: int,
                calibration_dataset: str,
                board_img_set: str,
                num_cars: int,
                save_imgs: bool,
                red_thrsh: int,
                detector_type: str,
                pix_thrsh: int,
                run_all: bool,
                warped_img: np.ndarray = None,
                K: np.ndarray = None,
                dist_coeff: np.ndarray = None,
                R: np.ndarray = None,
                t: np.ndarray = None):
    """_summary_
    This script assumes the cars are orange.
    
    
    Args:
        img_set (_type_): _description_
        img_idx (_type_): _description_
        calibration_dataset (_type_): _description_
        R (_type_): _description_
        t (_type_): _description_
    """

    title('CAR DETECTION')

    # Load the image
    img_path = f'data/transformation/results/{img_set}/*.png'
    intr_path = f'data/calibration/results/{calibration_dataset}'
    extr_path = f'data/detection/results/{board_img_set}'
    out_path = f'data/detection/results/{img_set}'
    
    if not run_all:
        try:
            K           = np.loadtxt(os.path.join(intr_path, 'K.txt'))
            dist_coeff  = np.loadtxt(os.path.join(intr_path, 'dist_coeff.txt'))
            R           = np.loadtxt(os.path.join(extr_path, 'R.txt'))
            t           = np.loadtxt(os.path.join(extr_path, 't.txt'))
            images = sorted(glob.glob(img_path))
            warped_img = cv2.imread(images[img_idx])
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f'File could not be found: {e}')

    dist_coeff = dist_coeff.flatten()

    assert K.shape == (3, 3), 'K must be a 3x3 matrix.'
    assert dist_coeff.shape == (5,), 'dist_coeff must be a 5-element vector.'
    assert R.shape == (3, 3), 'R must be a 3x3 matrix.'
    assert t.shape == (3, 1), 't must be a 3-element vector.'

    print('Detecting car features...', end='')

    # Normalize image such that brightness is uniform and avoid zero division
    epsilon = 1e-8
    norm_img = warped_img / (np.sum(warped_img, axis=2, keepdims=True) + epsilon)
    norm_img = (norm_img * 255).astype(np.uint8)
    
    hsv = cv2.cvtColor(norm_img, cv2.COLOR_BGR2HSV)
    
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    res = cv2.bitwise_and(norm_img, norm_img, mask=mask)
    
    red = res[:, :, 2]
    red = cv2.threshold(red, red_thrsh, 255, cv2.THRESH_BINARY)[1]
    
    cv2.imwrite(os.path.join(out_path, 'red_binary_map.png'), red)

    detector_type = detector_type.upper()
    match detector_type:
        case 'SIFT':
            detector = cv2.SIFT.create()
        case 'FAST':
            raise NotImplementedError('FAST detector not implemented.')
            detector = cv2.FastFeatureDetector.create()
        case 'BRIEF':
            raise NotImplementedError('BRIEF detector not implemented.')
            detector = cv2.xfeatures2d.BriefDescriptorExtractor.create()
        case 'ORB':
            raise NotImplementedError('ORB detector not implemented.')
            detector = cv2.ORB.create()
        case _:
            raise ValueError('Invalid detector specified.')
        
    keypoints = detector.detect(red, None)
    keypoints = np.array(keypoints)
    print(f'Found {len(keypoints)} keypoints.')

    kp_img = warped_img.copy()
    kp_img = cv2.drawKeypoints(kp_img, keypoints, kp_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    sizes = np.array([kp.size for kp in keypoints])
    
    largest_kps = keypoints[np.argsort(sizes)]
    
    car_img_pos = []
    for kp in reversed(largest_kps):
        if any(np.linalg.norm(np.array(kp.pt) - np.array(pos)) < pix_thrsh for pos in car_img_pos):
            continue

        car_img_pos.append(kp.pt)

        if len(car_img_pos) == num_cars:
            break
    
    for pos in car_img_pos:
        out_img = cv2.circle(warped_img, tuple(map(int, pos)), 10, (0, 255, 0), -1)
    
    
    out_img = cv2.resize(out_img, (0, 0), fx=0.3, fy=0.3)
    
    if save_imgs:
        cv2.imwrite(os.path.join(out_path, 'kp_detection.png'), kp_img)
        cv2.imwrite(os.path.join(out_path, 'car_detection.png'), out_img)
    
    for idx, pos in enumerate(car_img_pos):
        print(f'Car {idx} detected at image position {pos}.')
        
    print('Changing the coordinates to be given with respect to center of image...')

    car_direction = {"direction1": 0,
                     "direction2": 0,
                     "direction3": 0,
                     "direction4": 0}

    for i in range(len(car_img_pos)):
        car_img_pos[i] = np.array(car_img_pos[i]) - np.array([warped_img.shape[1] / 2, warped_img.shape[0] / 2])
        print(f'Car {i} detected at image position {car_img_pos[i]}.')
        if car_img_pos[i][0] > 0 and car_img_pos[i][1] < 0:
            car_direction["direction1"] += 1
        elif car_img_pos[i][0] < 0 and car_img_pos[i][1] < 0:
            car_direction["direction2"] += 1
        elif car_img_pos[i][0] < 0 and car_img_pos[i][1] > 0:
            car_direction["direction3"] += 1
        elif car_img_pos[i][0] > 0 and car_img_pos[i][1] > 0:
            car_direction["direction4"] += 1
        else:
            print('Undetermined position')
            
    with open(os.path.join(out_path, 'data.json'), 'w') as f:
        json.dump(car_direction, f, indent=4)
    