# TTK4255-ROBVIS-PROJECT

Code for Robotic Vision project in TTK4255 by Christian Le and Klevis Resuli.

## Project Description

Smart traffic control system
The idea is creating a system based on ESP32-Cam that is going to control the traffic light based on the number of cars on each side of an intersection making the traffic easier and also giving priority to emergencies(case of an ambulance being in the intersection).

The following main code is a proof of concept that does calibration, undistortion, pose estimation and homography to estimate positions and directions of cars.
For deployment in a stationary environment (camera is stationary in world frame), only the position and direction estimation would need to be done during inference.

## Prerequisites

Below are the prerequisites for this project, divided into categories.

### Python Envrionment

The following environment also needs to be installed to include the needed packages, this can easily be done by executing,

```bash
conda env create -f environment.yml
```

If this has already been done before, but an update is desired, run,

```bash
conda env update -f environment.yml --prune
```

The environment can be used in terminal by running,
```bash
conda activate robvis_project
```

or if using Visual Studio Code, look for **robvis_project** environment in (CTRL+P) "Python: Select Interpreter".


### Calibration

Below are the specifications for each calibration set used in this project.

* Chessboard (for calibration)
  - Square Size          :  2.9 cm
  - Corners (w x h)      :  7x7 = 49
  - Notes                :  Has a crack from C6 to H6, in addition to it being foldable, having a crack in the middle.
  - Location             : [data/calibration/esp_7x7_2.9](data/calibration/esp_7x7_2.9/)

* Checkerboard (on monitor in dark room)
  - Square Size          :  2.0 cm
  - Corners (w x h)      :  9x6 = 54
  - Notes                :  Camera got melted after this session due to high heat from the ESP32, thus both this set and chessboard set are outdated due to the potential of changed distortion parameters.
  - Location             : [data/calibration/esp_9x6_2.0](data/calibration/esp_9x6_2.0/)

* Checkerboard (newer, in normal office environment)
  - Square Size          :  1.8 cm
  - Corners (w x h)      :  8x6 = 48
  - Notes                :  This set was used for the final calibration.
  - Location             : [data/calibration/esp_8x6_1.8](data/calibration/esp_8x6_1.8/)

* Asus Zenfone 8:
  - Square Size          :  2.3 cm
  - Corners (w x h)      :  8x6 = 48
  - Notes                :  Secondary camera when ESP32 camera was unavailable.
  - Location             : [data/calibration/zenfone8_8x6_2.3](data/calibration/zenfone8_8x6_2.3/)

* ESP32 Details:
  - Name                 :  ESP32-CAM
  - Camera               :  OV2640
  - Notes                :  Code for operating the camera is in [src/tools/ESP32](src/tools/ESP32/)

The digital checkerboard were generated from this [site](https://markhedleyjones.com/projects/calibration-checkerboard-collection), but for recreating the project, any arbitrary checkerboard can be used as long as number of corners and square size is known.

Note that square sizes are given in centimeters, thus the unit for all distances in the code is in centimeters.

### "World Frame" Setup

* ArUco Tags
  - Sqr. Size            :  3.0 cm
  - Type                 :  APRILTAG 36h11
  - Source               :  [Clark's Docs](https://docs.cbteeple.com/robot/april-tags)

* Intersection Image
  - Paper Size           :  A3
  - Intersection Size    :  39cm x 20.4cm
  - Source               :  [Adobe Stock](https://stock.adobe.com/no/search/images?k=intersection&asset_id=251213671)

* 3D Printed Cars
  - Color                :  Orange
  - Notes                :  3D printed at [MakeNTNU](https://www.makentnu.no/)

* 3D Printed Traffic Light
  - Color                :  Black (Red, Yellow, Green LEDs)
  - Notes                :  3D printed at [MakeNTNU](https://www.makentnu.no/)
  - Electronics          :  LEDs and wires by Klevis

## Tasks

The tasks of the [main code](src/main.py) is sectioned by,
1. Calibrating the ESP32-CAM by using OpenCV, yielding the camera intrinsics matrix $\mathbf{K}$ and distortion parameters $k_1, k_2, p_1, p_2, k_3$, including the standard deviation of each estimate.
2. Printing undistorted images using results of calibration (Only for demonstration purposes, not needed for practical use), both using the estimate and a random sample from the distribution.
3. Estimating pose of world frame by using point correspondences of image and world coordinates of ArUco tags (Image coords found through OpenCV and world coords predetermined in centimeters), yielding the rotation $\mathbf{R}$ and translation $\mathbf{t}$ of the camera.
4. Projecting the corners of the intersection image onto the image plane.
5. Estimating the homography between the intersection and "above perspective" image to then project the intersection image onto the image plane.
6. Detecting and estimating the position and direction of cars in the image plane and then exporting the directions data for potential use in the ESP32.

### Practicalities

No explicit task of detecting the ambulances was done, but the last task is set up as general object detection, thus it should be possible to detect ambulances by specifying the colors of the ambulance by re-using "detect_cars" in the main code and extending the [configurations](configs/custom.json) to include the colors of the ambulance.

The part where the ESP32 uses the direction data to control the traffic light is not mentioned either as it is not relevant for the course, but can be found in the [ESP32 code](src/tools/ESP32/).

## How-to-use

The script can simply be run in two different configurations, specified in [default.json](configs/default.json) and [custom.json](configs/custom.json).

Default can be run without arguments,
```bash
python src/main.py
```

while to run custom, the argument is needed,
```bash
python src/main.py --config True
```

For evulation of the code, the main functionality is in the [utils](src/utils/) folder, where all the code is modularized for easy testing of each part. The [tools](src/tools/) folder contains peripheral code for ease of use and operation of the ESP32.

### Configuration

The configuration file is used to specify how the code should run and with which hyperparameters. Default and custom are currently identical, but the idea is to **only** change the custom file for tests and the default file contains the final project configuration. Check bottom of this README for a [rundown of the configuration.](#rundown-of-configuration)

## Outputs

For each run of the code, the following outputs are generated in the [output](output/) folder, where subdirectory names are the date and time of the run.

Each run contains the following outputs,

### [Calibration](output/example/calibration/)

* [data_name](output/example/calibration/esp_8x6_1.8/) : Shares the same name as the calibration set used, containing the estimated parameters.
  * dist_coeff.txt              : Contains the mean of estimated distortion parameters $k_1, k_2, p_1, p_2, k_3$.
  * image_size.txt      : Contains image dimensions, used to check if calibration images are of same dimension.
  * K.txt               : Contains the mean estimated camera intrinsics matrix $\mathbf{K}$.
  * std_int             : Contains the standard deviation of each estimate.
  * u_all.npy           : Detected checkerboard locations in image coordinates.
  * X_all.npy           : Detected checkerboard locations in 3D space coordinates.
* [failed](output/example/calibration/failed/esp_8x6_1.8/)      : Contains images that the code failed to detect the specified number of corners.

### [Undistorted](output/example/undistorted/esp32_cars4/)
* [estimate](output/example/undistorted/esp32_cars4/estimate/) : Contains undistorted images using the estimate of the calibration parameters.
* [sampled](output/example/undistorted/esp32_cars4/sampled/)   : Contains undistorted images using a random sample from the distribution of the calibration parameters, number of samples specified in the configuration file.

### [Detection](output/example/detection/)
* [board](output/example/detection/board/esp32_cars4/) : Contains modified version of input image along with pose estimation of the world frame.
  * board_result.png : Image with frame axes and detected corners.
  * R.txt           : Estimated rotation matrix $\mathbf{R}$.
  * t.txt           : Estimated translation vector $\mathbf{t}$.
* [cars](output/example/detection/cars/esp32_cars4/)   : Contains modified version of input image along with detected cars, i=0 is of the warped image while i=1 is non-warped but marked estimated board corners.
  * masked_hsv_{i}.png : Image with HSV mask applied, the limit values are specified in the configuration file.
  * binary_map_{i}.png : Contains thresholded version of HSV masked image, the threshold values are specified in the configuration file.
  * kp_detection_{i}.png : Modified version of input image with detected keypoints along with their sizes.
  * car_detection_{i}.png : Modified version of input image illustrating the assumed position of cars
  * data.json : Contains the assumed directions of cars, formatted to be used in the ESP32.
  In this context, the input image is the warped image from "Transformations".

### [Transformation](output/example/transformation/esp32_cars4/)
* img_corners.txt : Contains the image coordinates of the corners of the intersection image, the location of this corners have an offset 'border_size' specified in the configuration file.
* K2.txt : Contains the transformation matrix that converts world units to pixels that also retains the center of image as the origin.
* warped.png : Contains the intersection image warped onto the image plane.
* marked.png : Contains the intersection image with estimated corners marked.

### [Configuration](output/example/config.json)

The configuration for the run is also included in the output folder, for the sake of reproducibility.

## Rundown of Configuration file contents

- **calibration**: This section contains parameters related to the calibration process.
  - **dataset**: The name of the dataset used for calibration, located in the [data/calibration](data/calibration/) folder.
  - **board_size**: The size of the calibration board by number of corners, specified as [width, height].
  - **square_size**: The size of each square on the calibration board given in centimeters.
  - **rerun_detection**: A boolean indicating whether to rerun the detection process regardless of having existing results. [*BUG: Currently does this regardless of the boolean value.*]
  - **show_results**: A boolean indicating whether to display the calibration results.
  - **skip**: A boolean indicating whether to skip the calibration process.

- **undistortion**: This section contains parameters related to the undistortion process.
  - **img_set**: The set of images to be undistorted, located in the [data/distorted](data/distorted/) folder.
  - **calibration_dataset**: The dataset used for calibration, should be for the same camera as the images, located in [output/example/calibration/](output/example/calibration/).
  - **crop**: A boolean indicating whether to crop the undistorted images, if false, the undistorted images will have black borders.
  - **std_samples**: The number of random samples used in the undistortion process.
  - **skip**: A boolean indicating whether to skip the undistortion process.

- **board**: This section contains parameters related to the board.
  - **dictionary**: The dictionary used for marker detection. [Dataset only contains APRILTAG 36h11]
  - **img_set**: The set of images containing the board, located in the [data/detection](data/detection/) folder.
  - **img_idx**: The index of the image within the set to be processed.
  - **markers**: The world coordinates of the markers on the board relative to bottom left corner of the board.
  - **origin**: Location of desired new origin in world coordinates.
  - **ids**: The IDs of the markers, clockwise starting from upper left corner.
  - **refined**: A boolean indicating whether the board should be refined.
  - **calibration_dataset**: The dataset used for calibration, located in [output/example/calibration/](output/example/calibration/).
  - **save_rejected**: A boolean indicating whether to include rejected corners in the output.
  - **skip**: A boolean indicating whether to skip the board processing.

- **homography**: This section contains parameters related to the homography process.
  - **img_set**: The set of images to be processed, located in the [data/detection](data/detection/) folder. Should match the set used for the board.
  - **calibration_dataset**: The dataset used for calibration, located in [output/example/calibration/](output/example/calibration/).
  - **img_idx**: The index of the image within the set to be processed.
  - **height**: The height of the output image, recommended to be lower than height of screen resolution.
  - **board_size**: The size of the board in the image, specified as [width, height] in centimeters.
  - **border_size**: The length of the border around the intersection image in centimeters.
  - **skip**: A boolean indicating whether to skip the homography process.

- **cars**: This section contains parameters related to the car detection process.
  - **warp_img_set**: The set of warped images to be processed, located in [output/example/transformation/](output/example/transformation/).
  - **calibration_dataset**: The dataset used for calibration, located in [output/example/calibration/](output/example/calibration/).
  - **board_img_set**: The set of images containing the board, located in the [data/detection](data/detection/) folder.
  - **num_cars**: The total number of cars in the image to be detected.
  - **hsv_levels**: The HSV levels used for car detection, ranges from [0, 0, 0] to [360, 100, 100] (The code handles conversions internally.)
  - **thresholds**: The thresholds for R,G,B of output of HSV masked image. [0, 0, 0] turns mask to 0 and content to 1.
  - **min_distance**: The minimum distance between detected cars needed to be considered separate.
  - **detector_type**: The type of detector used for car detection, currently supports "ORB" and "SIFT".
  - **road_width**: The width of the road in the image.
  - **road_dist_from_orig**: The distance from the origin to the end of crossroad in the image.
  - **skip**: A boolean indicating whether to skip the car detection process.