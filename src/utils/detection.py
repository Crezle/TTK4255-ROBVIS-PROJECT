import os
import cv2
import glob
from cv2 import aruco
import numpy as np
from utils import camera

def detect_markers(detector, img, board, refined, K, dist_coeff):
    corners, ids, rejected = detector.detectMarkers(img)
    
    if refined:
        print('Refining detected markers...', end='')
        corners, ids, rejected, recoveredIdx = detector.refineDetectedMarkers(img, board, corners, ids, rejected, K, dist_coeff)
    
        if recoveredIdx is not None:
            print(f'Recovered {len(recoveredIdx)} markers.')
        else:
            print('No markers recovered.')
    
    return corners, ids, rejected

def detect_board(dictionary,
                 img_set,
                 img_idx,
                 board_corners,
                 ids,
                 refined,
                 calib_baseline,
                 save_imgs,
                 save_rejected):

    img_path = f'data/detection/images/{img_set}/*.jpg'
    out_path = f'data/detection/results/{img_set}'
    calib_baseline_path = f'data/calibration/results/{calib_baseline}'

    K           = np.loadtxt(os.path.join(calib_baseline_path, 'K.txt'))
    dist_coeff  = np.loadtxt(os.path.join(calib_baseline_path, 'dist_coeff.txt'))

    aruco_params = aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    aruco_params.adaptiveThreshConstant = 10
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dictionary))
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    print('Intializing board detection')
    images = camera.undistort(img_set, calib_baseline, True, False, 0)
    img = images[img_idx]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    board_ids = np.array(ids, dtype='int32')
    board_corners = np.array(board_corners, dtype='float32')

    if board_corners.shape[2] == 2:
        board_corners = np.concatenate((board_corners, np.zeros_like(board_corners[:, :, 0:1])), axis=2)
    elif board_corners.shape[2] != 3:
        raise ValueError('Board corners must have shape (N, 4, 2) or (N, 4, 3)')

    board = aruco.Board(board_corners, aruco_dict, board_ids)

    corners, ids, rejected = detect_markers(detector, img, board, refined, K, dist_coeff)
 
    if ids is not None:
        best_num_ids = len(ids)
    else:
        best_num_ids = 0
    temp_img = img
    for _ in range(10):
        # Since image is jpg format it needs to be smoothed and sharpened
        smooth_kernel = np.ones((5, 5), np.float32) / 25
        temp_img = cv2.filter2D(temp_img, -1, smooth_kernel)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        temp_img = cv2.filter2D(temp_img, -1, sharpen_kernel)
        new_corners, new_ids, new_rejected = detect_markers(detector, img, board, refined, K, dist_coeff)
        if ids is not None:
            if len(ids) > best_num_ids:
                corners = new_corners
                ids = new_ids
                rejected = new_rejected
                img = temp_img
                best_num_ids = len(ids)
            elif len(ids) < best_num_ids:
                break
    
    if ids is not None:
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        print('Estimating board pose...', end='')
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeff)
        R = cv2.Rodrigues(rvec)[0]
        num_det_markers = len(obj_pts) / 4
    
        if not ok:
            raise ValueError('Could not estimate board pose.')
        else:
            print('Board pose estimated successfully.')
    else:
        print('No markers detected, cannot estimate board pose.')
        num_det_markers = 0
        R = None
        tvec = None

    if save_imgs:
        out_img = None

        if ids is not None:
            out_img = aruco.drawDetectedMarkers(img, corners, ids)
            
        if save_rejected and rejected is not None:
            rej_img = aruco.drawDetectedMarkers(img, rejected, borderColor=(100, 0, 255))
            cv2.imwrite(os.path.join(out_path, 'rejected.png'), rej_img)  
        
        if num_det_markers > 0:
            out_img = cv2.drawFrameAxes(out_img, K, dist_coeff, rvec, tvec, 5)
        else:
            print('No markers detected, cannot draw axes.')
        
        if out_img is not None:
            cv2.imwrite(os.path.join(out_path, 'result.png'), out_img)
        else:
            print('No image to save.')

    if R is None or tvec is None:
        Warning('Board pose not estimated, returning None for R and tvec.')

    return R, tvec

def detect_cars(img_set, img_idx, calib_baseline, R, t):
    # Load the image
    img_path = f'data/detection/images/{img_set}/*.jpg'
    images = sorted(glob.glob(img_path))
    img = cv2.imread(images[img_idx])

    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the orange color range
    lower_orange = np.array([0, 50, 50])
    upper_orange = np.array([30, 255, 255])

    # Threshold the image to get the orange regions
    mask = cv2.inRange(hsv_img, lower_orange, upper_orange)

    # Find contours of the orange regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and shape
    min_area = 1000
    max_area = 10000
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2.0
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if area > min_area and area < max_area and aspect_ratio > min_aspect_ratio and aspect_ratio < max_aspect_ratio:
            filtered_contours.append(contour)

    # Draw bounding boxes around the filtered contours
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow("Detected Cars", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()