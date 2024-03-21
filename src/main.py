from calibrate_camera import calibrate_camera
from undistortion import undistort
from detect_apriltags import detect_apriltags

def main():
    img_path = "data/apriltags/multiple_test/0001.jpg"
    calibrate_camera(rerun_detection=False)
    undistort(img_path=img_path, save_img=False)
    detect_apriltags(num_corners=18, img_path=img_path)

if __name__ == '__main__':
    main()