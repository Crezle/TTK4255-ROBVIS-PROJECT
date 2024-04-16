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

def world_to_img_corners(world_pts,
                         brdr_sz,
                         run_all,
                 R,
                 t,
                 K,
                 dist_coeff):
    
    world_pts = np.array(world_pts, np.float32)
    
    # Only choose corners of world
    if world_pts.shape == (4, 4, 2):
        world_pts = np.array([world_pts[0, 0] + np.array([-brdr_sz, brdr_sz]),
                              world_pts[1, 1] + np.array([brdr_sz, brdr_sz]),
                              world_pts[2, 2] + np.array([brdr_sz, -brdr_sz]),
                              world_pts[3, 3] + np.array([-brdr_sz, -brdr_sz])])

    if world_pts.shape[1] == 2:
        # Add z-coordinate
        world_pts = np.concatenate((world_pts, np.zeros((world_pts.shape[0], 1))), axis=1)
    elif world_pts.shape[1] != 3:
        raise ValueError('world_pts must be a 4x2 or 4x3 matrix.')

    R = np.array(R, np.float32)
    t = np.array(t, np.float32)

    rvec = cv2.Rodrigues(R)[0]
    # Call cv2.projectPoints on each 2D array in world_pts
    img_pts = np.array([cv2.projectPoints(corner, rvec, t, K, dist_coeff)[0] for corner in world_pts]).reshape(4, 2)

    return img_pts

def warp_to_world(img_set,
                  img_idx,
                  src_pts,
                  height,
                  run_all):
    
    img_path = f'data/detection/images/{img_set}/*.jpg'
    out_path = f'data/transformation/results/{img_set}'
    
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
    cv2.imwrite(os.path.join(out_path, 'warped.png'), out_img)
    