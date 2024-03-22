import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import cv2
import os
np.random.seed(0)

def enhance_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)

    sharp_kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])

    img = cv2.filter2D(img, -1, sharp_kernel)
    img = cv2.filter2D(img, -1, sharp_kernel)
    img = cv2.GaussianBlur(img, (3, 3), 4)

    return img

def learn_enhance_params(imgs_path='data/apriltags/multiple_test/', save_params=False):
    '''
    Learns the best parameters for april tag detection
    Args:
        imgs_path: path to the folder containing images
        save_params: if True, saves the parameters to a file
    Returns:
        None
    '''
    for filename in os.listdir(imgs_path):
        if filename.endswith('.jpg'):
            img = cv2.imread(join(imgs_path, filename))
            img = enhance_image(img)

def undistort(img_path='data/sample_image.jpg', calib_results_folder='chessboard' ,save_img=False):
    print("Undistorting image...")
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

    path = "data/undistortion/"

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Sample distortion coefficients from normal distribution
    # Number of samples
    n = 5

    for i in range(n):
        dc_sampled = np.random.normal(dc, dc_std)
        newcameramtx_sampled, roi_sampled = cv2.getOptimalNewCameraMatrix(K, dc, (w, h), 1, (w, h))
        undist_img_samled = cv2.undistort(img, K, dc_sampled, None, newcameramtx_sampled)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dc, (w, h), 1, (w, h))
    undist_img = cv2.undistort(img, K, dc, None, newcameramtx)

    if save_img:
        cv2.imwrite(join(path, f"undistorted_{base_name}_sampled_{i}.jpg"), undist_img_samled)
        cv2.imwrite(join(path, f"undistorted_{base_name}_MLE.jpg"), undist_img)
        cv2.imwrite(join(path, file_name), img)

    print("Undistortion complete!")
    return undist_img
