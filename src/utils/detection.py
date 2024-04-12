import os
import cv2
import glob
from cv2 import aruco
import numpy as np

def detect_board(dictionary,
                 img_set,
                 img_idx,
                 board_corners,
                 ids,
                 refind,
                 coeffs,
                 save_imgs,
                 show_rejected):

    img_path = f'data/detection/images/{img_set}/*.jpg'
    out_path = f'data/detection/results/{img_set}'
    coeffs_path = f'data/calibration/results/{coeffs}'
    
    K           = np.loadtxt(os.path.join(coeffs_path, 'K.txt'))
    dist_coeff  = np.loadtxt(os.path.join(coeffs_path, 'dist_coeff.txt'))
    
    aruco_params = aruco.DetectorParameters()
    
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    aruco_params.adaptiveThreshConstant = 10
    
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dictionary))

    images = sorted(glob.glob(img_path))
    img = cv2.imread(images[img_idx])
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    board_ids = np.array(ids, dtype='int32')
    
    board_corners = np.array(board_corners, dtype='float32')
    if board_corners.shape[2] == 2:
        board_corners = np.concatenate((board_corners, np.zeros_like(board_corners[:, :, 0:1])), axis=2)
    elif board_corners.shape[2] != 3:
        raise ValueError('Board corners must have shape (N, 4, 2) or (N, 4, 3)')

    board = aruco.Board(board_corners, aruco_dict, board_ids)
    
    corners, ids, rejected = aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    
    if refind:
        corners, ids, rejected, recoveredIdx = aruco.refineDetectedMarkers(img, board, corners, ids, rejected, K, dist_coeff, parameters=aruco_params)
    
    # Estimate board pose
    if ids is not None:
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
    
    ok, R, t = cv2.solvePnP(obj_pts, img_pts, K, dist_coeff)
    
    if not ok:
        raise ValueError('Could not estimate board pose.')
    else:
        print('Board pose estimated successfully.')

    num_det_markers = len(obj_pts)

    if save_imgs:
        if ids is not None:
            out_img = aruco.drawDetectedMarkers(img, corners, ids)
            
        if show_rejected:
            out_img = aruco.drawDetectedMarkers(out_img, rejected, borderColor=(100, 0, 240))    
        
        if num_det_markers > 0:
            out_img = cv2.drawFrameAxes(out_img, K, dist_coeff, R, t, 1)
            cv2.imwrite(os.path.join(out_path, 'result.png'), out_img)
        else:
            print('No markers detected, cannot draw axes.')
            
    return R, t, num_det_markers
