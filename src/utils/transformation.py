import cv2
import numpy as np
import glob
import os

def change_origin(board_corners,
                  origin):
    
    board_corners = np.array(board_corners)
    origin = np.array(origin)
    
    board_corners -= origin
    
    return board_corners

def world_to_img_corners(img_set,
                         calib_set,
                         world_corners,
                         brdr_sz,
                         save_params,
                         run_all,
                         R: np.ndarray = None,
                         t: np.ndarray = None,
                         K: np.ndarray = None,
                         dist_coeff: np.ndarray = None):
    
    extrinsics_path = f'data/detection/results/{img_set}'
    intrinsics_path = f'data/calibration/results/{calib_set}'
    world_corners = np.array(world_corners, np.float32)
    out_path = f'data/transformation/results/{img_set}'
    
    if not run_all:
        try:
            R = np.loadtxt(os.path.join(extrinsics_path, 'R.txt'))
            t = np.loadtxt(os.path.join(extrinsics_path, 't.txt'))
            K = np.loadtxt(os.path.join(intrinsics_path, 'K.txt'))
            dist_coeff = np.loadtxt(os.path.join(intrinsics_path, 'dist_coeff.txt'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Could not load extrinsics/intrinsics parameters. {e}')
    
    assert R.shape == (3, 3), 'R must be a 3x3 matrix.'
    assert t.flatten().shape == (3,), 't must be a 3x1 matrix.'
    assert K.shape == (3, 3), 'K must be a 3x3 matrix.'
    assert dist_coeff.flatten().shape == (5,), 'dist_coeff must be a 5x1 matrix.'
    
    # Only choose corners of world
    if world_corners.shape == (4, 4, 2):
        world_corners = np.array([world_corners[0, 0] + np.array([-brdr_sz, brdr_sz]),
                              world_corners[1, 1] + np.array([brdr_sz, brdr_sz]),
                              world_corners[2, 2] + np.array([brdr_sz, -brdr_sz]),
                              world_corners[3, 3] + np.array([-brdr_sz, -brdr_sz])])

    if world_corners.shape[1] == 2:
        # Add z-coordinate
        world_corners = np.concatenate((world_corners, np.zeros((world_corners.shape[0], 1))), axis=1)
    elif world_corners.shape[1] != 3:
        raise ValueError('world_pts must be a 4x2 or 4x3 matrix.')

    R = np.array(R, np.float32)
    t = np.array(t, np.float32)

    rvec = cv2.Rodrigues(R)[0]
    # Call cv2.projectPoints on each 2D array in world_pts

    img_corners = np.array([cv2.projectPoints(corner, rvec, t, K, dist_coeff)[0] for corner in world_corners]).reshape(4, 2)
    
    if save_params:
        np.savetxt(os.path.join(out_path, 'img_pts.txt'), img_corners)
    

    return img_corners

def warp_to_world(img_set,
                  img_idx,
                  height,
                  save_img,
                  run_all,
                  src_pts: np.ndarray = None):
    
    img_path = f'data/detection/images/{img_set}/*.jpg'
    out_path = f'data/transformation/results/{img_set}'
    
    if not run_all:
        try:
            src_pts = np.loadtxt(os.path.join(out_path, 'img_pts.txt'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Could not load img_pts. {e}')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img_path = glob.glob(img_path)[img_idx]
    
    img = cv2.imread(img_path)
    
    aspect_ratio = img.shape[1] / img.shape[0]

    width = int(aspect_ratio * height)
    
    src_pts = np.array(src_pts)
    
    assert src_pts.shape == (4, 2), 'src_pts must be a 4x2 matrix.'
    
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    out_img = cv2.warpPerspective(img, H, (width, height))
    
    if save_img:
        cv2.imwrite(os.path.join(out_path, 'warped.png'), out_img)

    return out_img
    