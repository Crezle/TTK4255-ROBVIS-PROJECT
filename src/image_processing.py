import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import cv2
import os
np.random.seed(0)

def is_grayscale(img):
    if len(img.shape) < 3: return True
    if img.shape[2] == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

def _sharpen_image(img, kernel_dim, sigma):
    
    sharp_kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
    
    img = cv2.GaussianBlur(img, (kernel_dim, kernel_dim), sigma)

    img = cv2.filter2D(img, -1, sharp_kernel)

    return img

def _threshold_img(img, threshold, max_pixel_value=255):
    '''
    Thresholds the image
    '''
    img = cv2.threshold(img, threshold, max_pixel_value, cv2.THRESH_BINARY)[1]
    
    return img

def enhance_image(img, threshold, kernel_dim=5, sigma=1):
    '''
    Enhances images for better detection
    '''
    img = _sharpen_image(img, kernel_dim, sigma)
    img = _threshold_img(img, threshold)
    
    return img

def undistort(img_path='data/sample_image.jpg', calib_results_folder=None, save_img=False, save_path="data/undistortion/", debug=False):
    if debug and calib_results_folder is not None:
        print("Undistorting image...")
    elif calib_results_folder is None:
        print("No calibration results folder provided. Returning image without undistortion.")
        return cv2.imread(img_path)
    
    file_name = img_path.split('/')[-1]  # Extract the file name of imgpath
    base_name = file_name.split('.')[0] 
    
    folder = f'data/calibration/results/{calib_results_folder}'

    K           = np.loadtxt(join(folder, 'K.txt')) # intrinsic matrix
    dc          = np.loadtxt(join(folder, 'dc.txt')) # distortion coefficients
    std_int     = np.loadtxt(join(folder, 'std_int.txt')) # standard deviations of intrinsics (entries in K and distortion coefficients)
    u_all       = np.load(join(folder, 'u_all.npy')) # detected checkerboard corner locations
    image_size  = np.loadtxt(join(folder, 'image_size.txt')).astype(np.int32) # height,width
    mean_errors = np.loadtxt(join(folder, 'mean_errors.txt')) # mean error per image

    fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy = std_int

    dc_std = np.array([k1, k2, p1, p2, k3])

    img = cv2.imread(img_path)

    h, w = img.shape[:2]

    # Sample distortion coefficients from normal distribution
    # Number of samples
    n = 5

    if not os.path.exists(save_path) and save_img:
        os.makedirs(save_path)

    for i in range(n):
        dc_sampled = np.random.normal(dc, dc_std)
        newcameramtx_sampled, roi_sampled = cv2.getOptimalNewCameraMatrix(K, dc_sampled, (w, h), 1, (w, h))
        undist_img_samled = cv2.undistort(img, K, dc_sampled, None, newcameramtx_sampled)
        if save_img:
            cv2.imwrite(join(save_path, f"undistorted_{base_name}_sampled_{i}.jpg"), undist_img_samled)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dc, (w, h), 1, (w, h))
    undist_img = cv2.undistort(img, K, dc, None, newcameramtx)

    if save_img:
        cv2.imwrite(join(save_path, f"undistorted_{base_name}_MLE.jpg"), undist_img)
        cv2.imwrite(join(save_path, file_name), img)

    if debug:
        print("Undistortion complete!")
    return undist_img

def findHomography(aspect_ratio, pts_src):
    height = 720
    width = int(height * aspect_ratio)
    
    pts_dst = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    
    H, _ = cv2.findHomography(pts_src, pts_dst)
    
    if H.shape != (3, 3):
        raise ValueError("Could not find homography matrix.")
    
    return H, width, height

def homography_reprojection(img, H, width, height, result_path=None, result_name='warped_image.jpg'):
    
    img_warped = cv2.warpPerspective(img, H, (width, height))

    cv2.imshow('Source Image', img)
    cv2.imshow('Warped Source Image', img_warped)

    if os.getenv('GITHUB_ACTIONS') != 'true':
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not result_path is None and not os.path.exists(result_path):
        os.makedirs(result_path)
    cv2.imwrite(os.path.join(result_path, result_name), img_warped)
    
    return img_warped