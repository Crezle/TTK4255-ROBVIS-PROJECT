import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import cv2 as cv
import os
np.random.seed(0)

def undistort(img_path='data/undistortion/sample_image.jpg'):
    
    file_name = img_path.split('/')[-1]  # Extract the file name of imgpath
    base_name = file_name.split('.')[0] 
    
    folder = 'data/calibration/results'

    K           = np.loadtxt(join(folder, 'K.txt')) # intrinsic matrix
    dc          = np.loadtxt(join(folder, 'dc.txt')) # distortion coefficients
    std_int     = np.loadtxt(join(folder, 'std_int.txt')) # standard deviations of intrinsics (entries in K and distortion coefficients)
    u_all       = np.load(join(folder, 'u_all.npy')) # detected checkerboard corner locations
    image_size  = np.loadtxt(join(folder, 'image_size.txt')).astype(np.int32) # height,width
    mean_errors = np.loadtxt(join(folder, 'mean_errors.txt')) # mean error per image

    fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy = std_int

    dc_std = np.array([k1, k2, p1, p2, k3])

    path = "data/undistortion/"

    sample_img = cv.imread(img_path)
    h, w = sample_img.shape[:2]

    # Sample distortion coefficients from normal distribution
    # Number of samples
    n = 5

    for i in range(n):
        dc_sampled = np.random.normal(dc, dc_std)
        newcameramtx_sampled, roi_sampled = cv.getOptimalNewCameraMatrix(K, dc, (w, h), 1, (w, h))
        undist_img_samled = cv.undistort(sample_img, K, dc_sampled, None, newcameramtx_sampled)
        cv.imwrite(join(path, f"undistorted_{base_name}_sampled_{i}.jpg"), undist_img_samled)


    newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, dc, (w, h), 1, (w, h))
    undist_sample_img = cv.undistort(sample_img, K, dc, None, newcameramtx)

    cv.imwrite(join(path, f"undistorted_{base_name}_MLE.jpg"), undist_sample_img)
    cv.imwrite(join(path, file_name), sample_img)
