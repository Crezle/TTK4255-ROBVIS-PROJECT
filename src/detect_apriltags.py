import cv2
import numpy as np
from datetime import datetime
import os

def draw_corners(img, corners, ids, threshold, show_img=False, foldername="unnamed_detections"):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedMarkers(color_img, corners, ids)
    if show_img:
        cv2.imshow(f'threshold: {threshold}', color_img)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(f"data/apriltags/detections/{foldername}"):
        os.makedirs(f"data/apriltags/detections/{foldername}")
    cv2.imwrite(f"data/apriltags/detections/{foldername}/threshold{threshold}.jpg", color_img)

def configure_aruco_params(tag="april_tag_36h11", accuracy=0.10):
    if tag == "april_tag_36h11":
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    else:
        raise NotImplementedError("Only april_tag_36h11 is supported at the moment.")
    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.polygonalApproxAccuracyRate = accuracy
    return aruco_dict, arucoParams

def sharpen_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)

    sharp_kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])

    img = cv2.filter2D(img, -1, sharp_kernel)
    img = cv2.filter2D(img, -1, sharp_kernel)
    img = cv2.GaussianBlur(img, (3, 3), 4)

    return img

def main():
    aruco_dict, arucoParams = configure_aruco_params()
    # Import image
    img_path = "data/apriltags/multiple_test/0007.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gt_num_corners = 24
    img = sharpen_image(img)
    
    min_threshold = int(255*0.1)
    max_threshold = int(255*0.9)
    best_num_corners = 0
    best_threshold = None
    best_img = None
    print(f"min_threshold: {min_threshold}, max_threshold: {max_threshold}")
    for threshold in range(min_threshold, max_threshold):
        temp_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
        (corners, ids, rejected) = cv2.aruco.detectMarkers(temp_img, aruco_dict, parameters=arucoParams)
        
        if len(corners) == gt_num_corners:
            best_num_corners = len(corners)
            draw_corners(img, corners, ids, threshold, foldername="eq_to_gt_detections")
        elif len(corners) > best_num_corners and len(corners) < gt_num_corners:
            best_threshold = threshold
            best_num_corners = len(corners)
            best_img = temp_img
    
    if best_num_corners != gt_num_corners:
        print(f"Best number of corners: {best_num_corners}, threshold: {best_threshold}")
        print(f"corners: {len(corners)}, rejected: {len(rejected)}")
        draw_corners(best_img, corners, ids, best_threshold, foldername="most_detections")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()