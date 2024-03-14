# TTK4255-ROBVIS-PROJECT

Code for Robotic Vision project in TTK4255 by Christian Le and Klevis Resuli.

## Prerequisites

For this project, the following equipments were used:

* Chessboard (for calibration)
  - Size (Height/Width)  :  2.9 cm
  - Num. squares         :  8x8 = 64
  - Num. corners         :  7x7 = 49
  - Notes                :  Has a crack from C6 to H6, in addition to it being foldable, having a crack in the middle.

* Camera Device:
  - Name                 :  ESP32-CAM
  - Camera               :  OV2640

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
