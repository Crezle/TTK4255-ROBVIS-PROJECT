import cv2
import numpy as np
import glob
import os
from tools.termformatter import title
from tools.dataloader import load_parameters

def change_origin(config: dict):
    """Change the origin of the board corners to the specified origin.
    
    Args:
        config (dict): Dictionary containing the origin and markers.
        
    Returns:
        np.ndarray: New board corners.
    """

    board_corners = np.array([config['markers']['upleft'],
                     config['markers']['upright'],
                     config['markers']['downright'],
                     config['markers']['downleft']])
    origin = np.array(config['origin'])
    
    board_corners -= origin
    
    return board_corners

def world_to_img_corners(config: dict,
                         board_config: dict,
                         output_dir: str):
    """Convert world coordinates to image coordinates.

    Args:
        config (dict): Configuration dictionary.
        board_config (dict): Board configuration dictionary.
        output_dir (str): Output directory.
    """
    try:
        img_set = config['img_set']
        calib_dataset = config['calibration_dataset']
        world_corners = change_origin(board_config)
        brdr_sz = config['border_size']
    except KeyError as e:
        raise KeyError(f'Missing key in config: {e}')

    if config['skip']:
        title('WORLD TO IMAGE COORDINATE CONVERSION SKIPPED')
        return

    title('WORLD TO IMAGE COORDINATE CONVERSION')

    extr_path = os.path.join(output_dir, 'detection', 'board', img_set)
    intr_path = os.path.join(output_dir, 'calibration', calib_dataset)
    out_path = os.path.join(output_dir, 'transformation', img_set)

    world_corners = np.array(world_corners, np.float32)


    print('Loading extrinsics and intrinsics parameters...')
    try:
        extr_params = load_parameters(extr_path,
                                    'detection/board',
                                    img_set,
                                    ['R', 't'])
        
        intr_params = load_parameters(intr_path,
                                    'calibration',
                                    calib_dataset,
                                    ['K', 'dist_coeff'])
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Could not load extrinsics/intrinsics parameters. {e}')
    print('...success!\n')
    
    R = extr_params['R']
    t = extr_params['t']
    K = intr_params['K']
    dist_coeff = intr_params['dist_coeff']
    
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
    
    os.makedirs(out_path)

    np.savetxt(os.path.join(out_path, 'img_corners.txt'), img_corners)

def warp_to_world(config: dict,
                  output_dir: str):
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
    except KeyError as e:
        raise KeyError(f'Missing key in config: {e}')

    if config['skip']:
        title('PERSPECTIVE TRANSFORM SKIPPED')
        return

    title('PERSPECTIVE TRANSFORM')

    img_path = os.path.join('data', 'detection', img_set, '*.jpg')
    out_path = os.path.join(output_dir, 'transformation', img_set)

    print('Loading image coordinates of world corners...')
    try:
        src_pts = load_parameters(out_path,
                                'transformation',
                                img_set,
                                ['img_corners'])['img_corners']
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Could not load img_corners. {e}')
    print('...success!\n')

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
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
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
    