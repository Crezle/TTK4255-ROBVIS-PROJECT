# TTK4255-ROBVIS-PROJECT

Code for Robotic Vision project in TTK4255 by Christian Le and Klevis Resuli.

## The Project

Smart traffic control system
The idea is creating a system based on ESP32-Cam that is going to control the traffic light based on the number of cars on each side of an intersection making the traffic easier and also giving priority to emergencies(case of an ambulance being in the intersection).
Our goals consists on:
1. Calibrating the ESP32-CAM either manually or by using OpenCV (Finding K matrix and distortion parameters)
2. Capture sequence of images of car trajectories from one angle, where each car has a marker (something like AprilTags from Assignment 7)
3. Sending those images to the python script and identify the number of cars. This will be done by mapping the position of the cars into 3d space and and perhaps repoject the image to be seen from another perspective, since it might be easy to identify the cars.
4. After identifying the number of cars in each side of the intersection we send the data back to ESP32-CAM and decide which traffic light to turn on.

(Only step 1 has currently been implemented)

## Prerequisites

For this project, the following equipments has been used:

* Chessboard (for calibration)
  - Size                 :  2.9 cm
  - Num. corners (w x h) :  7x7 = 49
  - Notes                :  Has a crack from C6 to H6, in addition to it being foldable, having a crack in the middle.

* Checkerboard (on monitor)
  - Size                 :  2.0 cm
  - Num. corners (w x h) :  9x6 = 54

* Camera Device (1):
  - Name                 :  ESP32-CAM
  - Camera               :  OV2640

  - Name                 :  Zenfone8

The following environment also needs to be installed to include the needed packages, this can easily be done by executing,

```
conda env create -f environment.yml
```

If this has already been done before, but a potential update is desired, run,

```
conda env update -f environment.yml --prune
```

## How-to-run

Currently, our code only performs calibration and undistortion which can be run without any inputs:

```
python src/main.py
```

## Outputs

### Calibration
* dc.txt              : Contains the mean of estimated distortion parameters $k_1, k_2, p_1, p_2, k_3$.
* image_size.txt      : Contains image dimensions, used to check if calibration images are of same dimension.
* K.txt               : Contains the mean estimated camera intrinsics matrix.
* mean_errors         : Contains the mean errors of each image.
* std_int             : Contains the standard deviation of each estimate.
* u_all.npy           : Detected checkerboard locations in image coordinates.
* X_all.npy           : Detected checkerboard locations in 3D space coordinates.

### Undistortion
* Source image        : Copied to the 'undistortion' folder for comparison.
* Undistorted MLE     : Undistorted using the MLE of each parameter.
* Undistorted Sample  : Undistorted using a random sample from estimate distribution.
