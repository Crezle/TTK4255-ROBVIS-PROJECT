import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from tools.termformatter import title
from tools.dataloader import load_parameters

def _show_pose(img_path, width, height, square_size, u, X, K, dist_coeff):

    img_path = glob.glob(img_path)[0]
    img = cv2.imread(img_path)
    cv2.drawChessboardCorners(img, (width, height), u, True)
    
    _, rvec, tvec = cv2.solvePnP(X, u, K, dist_coeff)

    img = cv2.drawFrameAxes(img, K, dist_coeff, rvec, tvec, square_size, 3)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Pose', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calibrate(config: dict,
              output_dir: str):
    """Calibrate the camera using a checkerboard pattern.

    Args:
        config (dict): Configuration dictionary.
        output_dir (str): Output directory.
    """
    try:
        dataset = config['dataset']
        board_size = config['board_size']
        square_size = config['square_size']
        rerun_detection = config['rerun_detection']
        show_results = config['show_results']
    except KeyError as e:
        raise KeyError(f'Missing key in config: {e}')

    title('CALIBRATION PROCESS')

    data_path   = os.path.join('data', 'calibration', dataset, '*.jpg')
    out_path    = os.path.join(output_dir, 'calibration', dataset)
    fail_path    = os.path.join(output_dir, 'calibration', 'failed', dataset)
    
    if config['skip']:
        print('Skipping calibration, returning previous results.')
        try:
            K           = np.loadtxt(os.join(out_path, 'K.txt'))
            dist_coeff  = np.loadtxt(os.join(out_path, 'dist_coeff.txt'))
            std_int     = np.loadtxt(os.join(out_path, 'std_int.txt'))
        except FileNotFoundError as e:
            Warning(f'Could not load calibration results, {e}. Returning None.')
            return None, None, None
        print('Loaded previous results.')
        return K, dist_coeff, std_int
        
    width = board_size[0]
    height = board_size[1]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    detect_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

    os.makedirs(out_path)
    os.makedirs(fail_path)

    if os.path.exists(os.path.join(out_path, 'u_all.npy')) and not rerun_detection:
        u_all = np.load(os.path.join(out_path, 'u_all.npy'))
        X_all = np.load(os.path.join(out_path, 'X_all.npy'))
        image_size = np.loadtxt(os.path.join(out_path, 'image_size.txt')).astype(np.int32)
        print('Using existing results.')
    else:
        X_board = np.zeros((width*height, 3), np.float32)
        X_board[:,:2] = square_size*np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        X_all = []
        u_all = []
        image_size = None
        image_paths = glob.glob(data_path)
        for image_path in tqdm(sorted(image_paths), desc=f'Finding checkerboard corners in {dataset}'):
            print('%s...' % os.path.basename(image_path), end='')

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
                cv2.imwrite(fail_path + "/" + os.path.basename(image_path), I)

        np.savetxt(os.path.join(out_path, 'image_size.txt'), image_size)
        np.save(os.path.join(out_path, 'u_all.npy'), u_all)
        np.save(os.path.join(out_path, 'X_all.npy'), X_all)

    print('Calibrating. This may take a minute or two...', end='')
    results = cv2.calibrateCameraExtended(X_all, u_all, image_size, None, None)
    print('Done!')

    ok, K, dist_coeff, _, _, std_int, _, _ = results
    dist_coeff = dist_coeff.flatten()
    std_int = std_int.flatten()

    np.savetxt(os.path.join(out_path, 'K.txt'), K)
    np.savetxt(os.path.join(out_path, 'dist_coeff.txt'), dist_coeff)
    np.savetxt(os.path.join(out_path, 'std_int.txt'), std_int)
        
    if show_results:
        _show_pose(data_path, width, height, square_size, u_all[0], X_all[0], K, dist_coeff)

def undistort(config: dict,
              output_dir: str):
    """Undistort images using the camera calibration results.

    Args:
        config (dict): Configuration dictionary.
        output_dir (str): Output directory.
    """
    try:
        img_set = config['img_set']
        calib_dataset = config['calibration_dataset']
        crop = config['crop']
        std_samples = config['std_samples']
    except KeyError as e:
        raise KeyError(f'Missing key in config: {e}')

    if config['skip']:
        title('UNDISTORTION PROCESS SKIPPED')
        return

    title('UNDISTORTION PROCESS')

    data_path = os.path.join('data', 'distorted', img_set, '*.jpg')
    out_path = os.path.join(output_dir, 'undistorted', img_set)
    calib_path = os.path.join(output_dir, 'calibration', calib_dataset)

    print('Loading calibration parameters...')
    try:
        params = load_parameters(calib_path,
                                'calibration',
                                calib_dataset,
                                ['K', 'dist_coeff', 'std_int'])
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Could not load calibration parameters. {e}')
    print('...success!\n')

    K = params['K']
    dist_coeff = params['dist_coeff']
    std_int = params['std_int']

    assert K.shape == (3, 3), 'K must be a 3x3 matrix.'
    assert dist_coeff.shape == (5,), 'dist_coeff must be a 5-element vector.'
    assert std_int.shape == (18,), 'std_int must be a 17-element vector.'
    
    dc_std = np.array(std_int[4:9])

    images = glob.glob(data_path)
    undist_imgs = []
    
    os.makedirs(os.path.join(out_path, 'estimate'))
    os.makedirs(os.path.join(out_path, 'sampled'))
    
    for data_path in tqdm(sorted(images), desc=f'Undistorting images in {img_set}'):
        
        img = cv2.imread(data_path)

        height, width = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, (width, height), 1, (width, height))
        undist_img = cv2.undistort(img, K, dist_coeff, None, newcameramtx)

        if crop:
            x, y, w, h = roi
            undist_img = undist_img[y:y+h, x:x+w]

        cv2.imwrite(os.path.join(out_path, 'estimate', os.path.basename(data_path)), undist_img)
        undist_imgs.append(undist_img)
        
        if std_samples > 0:
            for i in tqdm(range(std_samples), desc='Sampling distortion coefficients'):
                dc_sampled = np.random.normal(dist_coeff, dc_std)
                newcameramtx_sampled, roi_sampled = cv2.getOptimalNewCameraMatrix(K, dc_sampled, (width, height), 1, (width, height))
                undist_img_sampled = cv2.undistort(img, K, dc_sampled, None, newcameramtx_sampled)
                
                if crop:
                    x, y, w, h = roi_sampled
                    undist_img_sampled = undist_img_sampled[y:y+h, x:x+w]
                
                cv2.imwrite(os.path.join(out_path, 'sampled', os.path.basename(data_path) + f'_{i}.jpg'), undist_img_sampled)

    print('Undistortion complete!')
