import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from os.path import join, basename, realpath, dirname, exists, splitext
from tools.termformatter import title

def calibrate(dataset: str,
              board_size,
              square_size: float,
              rerun_detection: bool,
              save_params: bool,
              show_results: bool):

    title('CALIBRATION PROCESS')

    img_path    = f'data/calibration/images/{dataset}/*.jpg'
    out_path    = f'data/calibration/results/{dataset}'
    err_path    = f'data/calibration/failed/{dataset}'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    if not os.path.exists(err_path):
        os.makedirs(err_path)
        
    width = board_size[0]
    height = board_size[1]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    detect_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK


    if exists(join(out_path, 'u_all.npy')) and not rerun_detection:
        u_all = np.load(join(out_path, 'u_all.npy'))
        X_all = np.load(join(out_path, 'X_all.npy'))
        image_size = np.loadtxt(join(out_path, 'image_size.txt')).astype(np.int32)
        print('Using existing results.')
    else:
        X_board = np.zeros((width*height, 3), np.float32)
        X_board[:,:2] = square_size*np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        X_all = []
        u_all = []
        image_size = None
        image_paths = glob.glob(img_path)
        for image_path in tqdm(sorted(image_paths), desc=f'Finding checkerboard corners in {dataset}'):
            print('%s...' % basename(image_path), end='')

            I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if not image_size:
                image_size = I.shape
            elif I.shape != image_size:
                print('Image size mismatch, aborting.')
                quit()

            ok, u = cv2.findChessboardCorners(I, (width,height), flags=detect_flags)
            if ok:
                print(f'detected all {len(u)} checkerboard corners.')
                X_all.append(X_board)
                u = cv2.cornerSubPix(I, u, (11,11), (-1,-1), criteria)
                u_all.append(u)
            else:
                print(f'failed to detect checkerboard corners for image, skipping.')
                cv2.imwrite(err_path + "/" + basename(image_path), I)

        np.savetxt(join(out_path, 'image_size.txt'), image_size)
        np.save(join(out_path, 'u_all.npy'), u_all)
        np.save(join(out_path, 'X_all.npy'), X_all)

    print('Calibrating. This may take a minute or two...', end='')
    results = cv2.calibrateCameraExtended(X_all, u_all, image_size, None, None)
    print('Done!0')

    ok, K, dist_coeff, R, t, std_int, _, _ = results

    mean_errors = []
    for i in range(len(X_all)):
        u_hat, _ = cv2.projectPoints(X_all[i], R[i], t[i], K, dist_coeff)
        vector_errors = (u_hat - u_all[i])[:,0,:]
        scalar_errors = np.linalg.norm(vector_errors, axis=1)
        mean_errors.append(np.mean(scalar_errors))
        
    if save_params:
        np.savetxt(join(out_path, 'K.txt'), K)
        np.savetxt(join(out_path, 'dist_coeff.txt'), dist_coeff)
        np.savetxt(join(out_path, 'mean_errors.txt'), mean_errors)
        np.savetxt(join(out_path, 'std_int.txt'), std_int)
        
    if show_results:
        show_calibration_results(dataset)
        show_pose(img_path, width, height, square_size, u_all[0], X_all[0], K, dist_coeff)




    # Probably not necessary
    return K, dist_coeff, std_int

def undistort(img_set: str,
              calibration_dataset: str,
              crop: bool,
              save_imgs: bool,
              std_samples: int,
              passthrough: bool,
              K: np.ndarray = None,
              dist_coeff: np.ndarray = None,
              std_int: np.ndarray = None):

    title('UNDISTORTION PROCESS')

    img_path = f'data/undistortion/distorted/{img_set}/*.jpg'
    out_path = f'data/undistortion/results/{img_set}'
    calibration_dataset_path = f'data/calibration/results/{calibration_dataset}'
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if not passthrough:
        try:
            K           = np.loadtxt(join(calibration_dataset_path, 'K.txt'))
            dist_coeff  = np.loadtxt(join(calibration_dataset_path, 'dist_coeff.txt'))
            std_int     = np.loadtxt(join(calibration_dataset_path, 'std_int.txt'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Loading calibration results failed with error: {e}')

    dist_coeff = dist_coeff.flatten()
    std_int = std_int.flatten()

    assert K.shape == (3, 3), 'K must be a 3x3 matrix.'
    assert dist_coeff.shape == (5,), 'dist_coeff must be a 5-element vector.'
    assert std_int.shape == (18,), 'std_int must be a 17-element vector.'
    
    dc_std = np.array(std_int[4:9])

    images = glob.glob(img_path)
    undist_imgs = []
    
    for img_path in tqdm(sorted(images), desc=f'Undistorting images in {img_set}'):
        
        img = cv2.imread(img_path)

        height, width = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, (width, height), 1, (width, height))
        undist_img = cv2.undistort(img, K, dist_coeff, None, newcameramtx)

        if crop:
            x, y, w, h = roi
            undist_img = undist_img[y:y+h, x:x+w]
        
        if save_imgs:
            cv2.imwrite(join(out_path, basename(img_path)), undist_img)
        undist_imgs.append(undist_img)
        
        if std_samples > 0 and save_imgs:
            for i in tqdm(range(std_samples), desc='Sampling distortion coefficients'):
                dc_sampled = np.random.normal(dist_coeff, dc_std)
                newcameramtx_sampled, roi_sampled = cv2.getOptimalNewCameraMatrix(K, dc_sampled, (width, height), 1, (width, height))
                undist_img_sampled = cv2.undistort(img, K, dc_sampled, None, newcameramtx_sampled)
                
                if crop:
                    x, y, w, h = roi_sampled
                    undist_img_sampled = undist_img_sampled[y:y+h, x:x+w]
                
                cv2.imwrite(join(out_path, basename(img_path) + f'_sampled_{i}.jpg'), undist_img_sampled)

    print('Undistortion complete!0')

def show_calibration_results(dataset):
    folder = f'data/calibration/results/{dataset}'

    K           = np.loadtxt(join(folder, 'K.txt'))
    dc          = np.loadtxt(join(folder, 'dist_coeff.txt'))
    std_int     = np.loadtxt(join(folder, 'std_int.txt'))
    u_all       = np.load(join(folder, 'u_all.npy'))
    image_size  = np.loadtxt(join(folder, 'image_size.txt')).astype(np.int32) # height,width
    mean_errors = np.loadtxt(join(folder, 'mean_errors.txt'))

    fx,fy,cx,cy,k1,k2,p1,p2,k3,_,_,_,_,_,_,_,_,_ = std_int

    print()
    print('Calibration results')
    print('================================')
    print('Focal length and principal point')
    print('--------------------------------')
    print('fx:%13.5f +/- %.5f' % (K[0,0], fx))
    print('fy:%13.5f +/- %.5f' % (K[1,1], fy))
    print('cx:%13.5f +/- %.5f' % (K[0,2], cx))
    print('cy:%13.5f +/- %.5f' % (K[1,2], cy))
    print()
    print('Distortion coefficients')
    print('--------------------------------')
    print('k1:%13.5f +/- %.5f' % (dc[0], k1))
    print('k2:%13.5f +/- %.5f' % (dc[1], k2))
    print('k3:%13.5f +/- %.5f' % (dc[4], k3))
    print('p1:%13.5f +/- %.5f' % (dc[2], p1))
    print('p2:%13.5f +/- %.5f' % (dc[3], p2))
    print('--------------------------------')
    print()
    print('The number after "+/-" is the standard deviation.')
    print()

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.bar(range(len(mean_errors)), mean_errors)
    plt.title('Mean error per image')
    plt.xlabel('Image index')
    plt.ylabel('Mean error (pixels)')
    plt.tight_layout()

    plt.subplot(122)
    for i in range(u_all.shape[0]):
        plt.scatter(u_all[i, :, 0, 0], u_all[i, :, 0, 1], marker='.')
    plt.axis('image')
    plt.xlim([0, image_size[1]])
    plt.ylim([image_size[0], 0])
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')
    plt.title('All corner detections')
    plt.tight_layout()
    plt.savefig(folder + '/calibration_results.png')
    if os.getenv("GITHUB_ACTIONS") != 'true':
        plt.show()
    else:
        plt.clf()
        
def show_pose(img_path, width, height, square_size, u, X, K, dist_coeff):

    img_path = glob.glob(img_path)[0]
    img = cv2.imread(img_path)
    cv2.drawChessboardCorners(img, (width, height), u, True)
    
    ret, rvec, tvec = cv2.solvePnP(X, u, K, dist_coeff)

    img = cv2.drawFrameAxes(img, K, dist_coeff, rvec, tvec, square_size, 3)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Pose', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
