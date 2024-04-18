import cv2
import numpy as np
import glob
import os
from tools.termformatter import title

def change_origin(config: dict):
    """Change the origin of the board corners to the specified origin.
    
    Args:
        config (dict): Dictionary containing the origin and markers.
        
    Returns:
        np.ndarray: New board corners.
    """
    title('CHANGING ORIGIN OF BOARD CORNERS')

    board_corners = np.array([config['markers']['upleft'],
                     config['markers']['upright'],
                     config['markers']['downright'],
                     config['markers']['downleft']])
    origin = np.array(config['origin'])

    print('Current board corners: \n', board_corners)
    
    board_corners -= origin

    print('New board corners: \n', board_corners)
    
    return board_corners

def world_to_img_corners(config: dict,
                         board_config: dict,
                         run_all: bool,
                         R: np.ndarray,
                         t: np.ndarray,
                         K: np.ndarray,
                         dist_coeff: np.ndarray):
    """Convert world coordinates to image coordinates.

    Args:
        config (dict): Configuration dictionary.
        board_config (dict): Board configuration dictionary.
        run_all (bool): Run all steps.
        R (np.ndarray): Rotation matrix.
        t (np.ndarray): Translation vector.
        K (np.ndarray): Intrinsic camera matrix.
        dist_coeff (np.ndarray): Distortion coefficients.

    Returns:
        img_corners (np.ndarray): Image coordinates of world corners.
    """
    try:
        img_set = config['img_set']
        calib_set = config['calibration_dataset']
        world_corners = change_origin(board_config)
        brdr_sz = config['border_size']
        save_params = config['save_params']
    except KeyError as e:
        raise KeyError(f'Missing key in config: {e}')

    title('IMAGE COORDINATE OF WORLD CORNERS')
    
    extrinsics_path = f'data/detection/results/{img_set}'
    intrinsics_path = f'data/calibration/results/{calib_set}'
    world_corners = np.array(world_corners, np.float32)
    out_path = f'data/transformation/results/{img_set}'
    
    if config['skip']:
        print('Skipping world to image coordinate conversion, returning previous results...', end='')
        try:
            img_corners = np.loadtxt(os.path.join(out_path, 'src_corners.txt'))
        except FileNotFoundError as e:
            Warning(f'Could not load src_corners. {e}. Returning None.')
            return None
        print('Successful!')
        return img_corners

    elif not run_all:
        print('Loading previously calculated extrinsics/intrinsics parameters...', end='')
        try:
            R = np.loadtxt(os.path.join(extrinsics_path, 'R.txt'))
            t = np.loadtxt(os.path.join(extrinsics_path, 't.txt')).flatten()
            K = np.loadtxt(os.path.join(intrinsics_path, 'K.txt'))
            dist_coeff = np.loadtxt(os.path.join(intrinsics_path, 'dist_coeff.txt')).flatten()
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Could not load extrinsics/intrinsics parameters. {e}')
        print('Success!')
    else:
        print('Using provided extrinsics/intrinsics parameters.')
    
    assert R.shape == (3, 3), 'R must be a 3x3 matrix.'
    assert t.shape == (3,), 't must be a 3x1 matrix.'
    assert K.shape == (3, 3), 'K must be a 3x3 matrix.'
    assert dist_coeff.shape == (5,), 'dist_coeff must be a 5x1 matrix.'
    
    # Only choose outer corners of each marker and offset them by brdr_sz
    if world_corners.shape == (4, 4, 2):
        print('Calculating corners of crop of world image...')
        world_corners = np.array([world_corners[0, 0] + np.array([-brdr_sz, brdr_sz]),
                              world_corners[1, 1] + np.array([brdr_sz, brdr_sz]),
                              world_corners[2, 2] + np.array([brdr_sz, -brdr_sz]),
                              world_corners[3, 3] + np.array([-brdr_sz, -brdr_sz])])
        print('Image coordinate of world corners: \n', world_corners)

    if world_corners.shape[1] == 2:
        # Add z-coordinate
        world_corners = np.concatenate((world_corners, np.zeros((world_corners.shape[0], 1))), axis=1)
    elif world_corners.shape[1] != 3:
        raise ValueError('world_pts must be a 4x2 or 4x3 matrix.')

    R = np.array(R, np.float32)
    t = np.array(t, np.float32)

    rvec = cv2.Rodrigues(R)[0]

    img_corners = np.array([cv2.projectPoints(corner, rvec, t, K, dist_coeff)[0] for corner in world_corners]).reshape(4, 2)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    if save_params:
        np.savetxt(os.path.join(out_path, 'src_corners.txt'), img_corners)

    return img_corners

def warp_to_world(config: dict,
                  run_all: bool,
                  src_pts: np.ndarray):
    """Warp the image to world coordinates.

    Args:
        config (dict): Configuration dictionary.
        run_all (bool): Run all steps.
        src_pts (np.ndarray): Image coordinates of world corners.

    Returns:
        out_img (np.ndarray | None): Warped image.
        K2 (np.ndarray | None): World to image units conversion matrix.
    """
    try:
        img_set = config['img_set']
        img_idx = config['img_idx']
        height = config['height']
        board_size = config['board_size']
        save_img = config['save_imgs']
    except KeyError as e:
        raise KeyError(f'Missing key in config: {e}')

    title('PERSPECTIVE TRANSFORM WITH HOMOGRAPHY')

    img_path = f'data/detection/images/{img_set}/*.jpg'
    out_path = f'data/transformation/results/{img_set}'

    if config['skip']:
        print('Skipping perspective transform, returning previous results...', end='')
        try:
            out_img = cv2.imread(os.path.join(out_path, 'warped.png'))
            K2 = np.loadtxt(os.path.join(out_path, 'K2.txt'))
        except FileNotFoundError as e:
            Warning(f'Could not load warped image or K2. {e}. Returning None.')
            return None, None
        print('Successful!')

    elif not run_all:
        print('Loading previously calculated src_pts...', end='')
        try:
            src_pts = np.loadtxt(os.path.join(out_path, 'src_corners.txt'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Could not load src_corners. {e}')
        print('Success!')
    else:
        print('Using provided src_corners.')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img_path = glob.glob(img_path)[img_idx]
    
    img = cv2.imread(img_path)
    
    aspect_ratio = img.shape[1] / img.shape[0]

    width = int(aspect_ratio * height)
    
    src_pts = np.array(src_pts)
    
    assert src_pts.shape == (4, 2), 'src_pts must be a 4x2 matrix.'
    
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    print("Finding homography matrix...")
    H, _ = cv2.findHomography(src_pts, dst_pts)
    print("Found homography matrix: \n", H)
    
    out_img = cv2.warpPerspective(img, H, (width, height))
    
    if save_img:
        cv2.imwrite(os.path.join(out_path, 'warped.png'), out_img)

    print("Calculating world to image units conversion matrix...")
    fx = width / board_size[0]
    fy = -(height / board_size[1])
    # K2 translates pixels to world units (centimeters in this case) with y-axis pointing upwards
    K2 = np.array([[fx, 0, 0],
                     [0, fy, 0],
                     [0, 0, 1]])
    print("World to image units conversion matrix: \n", K2)
    np.savetxt(os.path.join(out_path, 'K2.txt'), K2)

    return out_img, K2
    