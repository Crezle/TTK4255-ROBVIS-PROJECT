import os
import cv2
import glob
from cv2 import aruco
import numpy as np
from tools.termformatter import title
from tools.dataloader import load_parameters
from utils.transformation import change_origin
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

def estimate_pose(config: dict,
                 output_dir: str):
    """Detects the board in the image and estimates the pose.
    
    Args:
        config (dict): Dictionary containing the configuration parameters.
        output_dir (str): Output directory for saving the results.
    """
    try:
        dictionary = config['dictionary']
        img_set = config['img_set']
        img_idx = config['img_idx']
        board_corners = change_origin(config)
        ids = config['ids']
        refined = config['refined']
        calib_dataset = config['calibration_dataset']
        save_rejected = config['save_rejected']
    except KeyError as e:
        raise KeyError(f'Could not find key {e} in config.')
    
    if config['skip']:
        title('BOARD DETECTION SKIPPED')
        return
        
    title("BOARD DETECTION")

    data_path = os.path.join('data', 'detection', img_set, '*.jpg')
    out_path = os.path.join(output_dir, 'detection', 'board', img_set)
    calib_path = os.path.join(output_dir, 'calibration', calib_dataset)

    print('Loading intrinsics parameters...')
    try:
        intr_params = load_parameters(calib_path,
                                    'calibration',
                                    calib_dataset,
                                    ['K', 'dist_coeff'])
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Could not load calibration parameters. {e}')
    print('...success!\n')
    
    K = intr_params['K']
    dist_coeff = intr_params['dist_coeff']

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
    images = sorted(glob.glob(data_path))
    assert len(images) > 0, 'No images found in the specified path.'
    assert img_idx < len(images), 'Image index out of bounds.'
    
    img = cv2.imread(images[img_idx])

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

    out_img = None

    if ids is not None:
        out_img = aruco.drawDetectedMarkers(img, corners, ids)
        
    if save_rejected and rejected is not None:
        out_img = aruco.drawDetectedMarkers(img, rejected, borderColor=(100, 0, 255))
    
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
    
    tvec = tvec.flatten()
    try:
        np.savetxt(os.path.join(out_path, 'R.txt'), R)
        np.savetxt(os.path.join(out_path, 't.txt'), tvec)
    except FileNotFoundError as e:
        Warning(f'Could not save results. {e}')

def detect_cars(config: dict,
                output_dir: str):
    """Detects cars in the image and exports directions to JSON-file for further processing.

    Args:
        config (dict): Configuration dictionary.
        output_dir (str): Output directory.
    """
    try:
        img_set = config['warp_img_set']
        img_idx = config['img_idx']
        calibration_dataset = config['calibration_dataset']
        board_img_set = config['board_img_set']
        num_cars = config['num_cars']
        hsv_levels = config['hsv_levels']
        thresholds = config['thresholds']
        pix_thrsh = config['min_distance']
    except KeyError as e:
        raise KeyError(f'Missing key in config: {e}')

    if config['skip']:
        title('CAR DETECTION SKIPPED')
        return

    title('CAR DETECTION')

    warped_img_path = os.path.join(output_dir, 'transformation', img_set, '*.png')
    transf_path = os.path.join(output_dir, 'transformation', img_set)
    intr_path = os.path.join(output_dir, 'calibration', calibration_dataset)
    extr_path = os.path.join(output_dir, 'detection', 'board', board_img_set)
    out_path = os.path.join(output_dir, 'detection', 'cars', img_set)




    print('Loading previously saved parameters...')
    try:
        extr_params = load_parameters(extr_path,
                                    'detection',
                                    board_img_set,
                                    ['R', 't'])
        
        intr_params = load_parameters(intr_path,
                                    'calibration',
                                    calibration_dataset,
                                    ['K', 'dist_coeff'])
        
        K2 = load_parameters(transf_path,
                            'transformation',
                            img_set,
                            ['K2'])['K2']
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Could not load parameters. {e}')
    print('...success!\n')
    
    R = extr_params['R']
    t = extr_params['t']
    K1 = intr_params['K']
    dist_coeff = intr_params['dist_coeff']
    images = sorted(glob.glob(warped_img_path))
    warped_img = cv2.imread(images[img_idx])

    assert K1.shape == (3, 3), 'K must be a 3x3 matrix.'
    assert dist_coeff.flatten().shape == (5,), 'dist_coeff must be a 5-element vector.'
    assert R.shape == (3, 3), 'R must be a 3x3 matrix.'
    assert t.flatten().shape == (3,), 't must be a 3-element vector.'

    print('Detecting car features...', end='')

    # Normalize image such that brightness is uniform and avoid zero division
    hsv_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

    lower = np.array(hsv_levels[0])
    upper = np.array(hsv_levels[1])
    lower = lower * (255/100)
    upper = upper * (255/100)

    mask = cv2.inRange(hsv_img, lower, upper)
    
    masked_hsv = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    
    bgr_img = cv2.cvtColor(masked_hsv, cv2.COLOR_HSV2BGR)
    
    red = bgr_img[:, :, 2]
    green = bgr_img[:, :, 1]
    blue = bgr_img[:, :, 0]

    red = cv2.threshold(red, thresholds[0], 255, cv2.THRESH_BINARY)[1]
    green = cv2.threshold(green, thresholds[1], 255, cv2.THRESH_BINARY)[1]
    blue = cv2.threshold(blue, thresholds[2], 255, cv2.THRESH_BINARY)[1]
    
    binary_map = cv2.bitwise_or(cv2.bitwise_or(red, green), blue)

    detector = cv2.SIFT.create()

    keypoints = detector.detect(bgr_img, binary_map)
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
    
    out_img = warped_img
    for pos in car_img_pos:
        out_img = cv2.circle(out_img, tuple(map(int, pos)), 10, (0, 255, 0), -1)
    
    os.makedirs(out_path)
    
    cv2.imwrite(os.path.join(out_path, 'masked_hsv.png'), bgr_img)
    cv2.imwrite(os.path.join(out_path, 'binary_map.png'), binary_map)
    cv2.imwrite(os.path.join(out_path, 'binary_map_red.png'), red)
    cv2.imwrite(os.path.join(out_path, 'binary_map_blue.png'), green)
    cv2.imwrite(os.path.join(out_path, 'binary_map_green.png'), blue)
    cv2.imwrite(os.path.join(out_path, 'kp_detection.png'), kp_img)
    cv2.imwrite(os.path.join(out_path, 'car_detection.png'), out_img)

    for idx, pos in enumerate(car_img_pos):
        print(f'Car {idx} detected at image position {pos}.')
        
    print('\nChanging the coordinates to be given with respect to center of image and in world units (centimeters)...')

    car_direction = {"direction1": 0,
                     "direction2": 0,
                     "direction3": 0,
                     "direction4": 0}

    for i in range(len(car_img_pos)):
        car_img_pos[i] = np.array(car_img_pos[i]) - np.array([warped_img.shape[1] / 2, warped_img.shape[0] / 2])
        car_img_pos[i] = (np.linalg.inv(K2) @ np.concatenate((car_img_pos[i].T, np.ones(1)), axis=0))[:2]
        print(f'Car {i} detected at image position {car_img_pos[i]}.')
        if car_img_pos[i][0] >= 5.5 and np.abs(car_img_pos[i][1]) <= 3.5:
            car_direction["direction1"] += 1
        elif np.abs(car_img_pos[i][0]) <= 3.5 and car_img_pos[i][1] >= 5.5:
            car_direction["direction2"] += 1
        elif car_img_pos[i][0] <= -5.5 and car_img_pos[i][1] <= 3.5:
            car_direction["direction3"] += 1
        elif np.abs(car_img_pos[i][0]) <= 3.5 and car_img_pos[i][1] <= -5.5:
            car_direction["direction4"] += 1
        else:
            print(f'Car {i} at undetermined position, cannot determine direction.')

    print("Exporting car directions to JSON-file...", end='')
    with open(os.path.join(out_path, 'data.json'), 'w') as f:
        json.dump(car_direction, f, indent=4)
    print("Success!")
