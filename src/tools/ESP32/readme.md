# ESP32-CAM

This folder contains the necessary tools and code for the ESP32-CAM microcontroller to work

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation](#implementation)


## Introduction

This part of the project is structured in two ways firstly the firmware for the esp32-cam which is inside the src folder and the dependent libraries in the lib/esp32cam folder. The second part is the python script(ESP32.py) which makes the communication with the Google Drive server and the esp32-cam server to get the capture. 


## Installation

To use the code in this folder, you will need to have the following software installed:

- [VSCode] (https://code.visualstudio.com/Download)
- [PlatormIO](https://platformio.org/install/ide?install=vscode)

## Usage
- Open the ESP32 folder in platformio
- Flash the esp32-cam, you need to have it connected to your computer through USB port
- to flash run in the terminal platformio run --target upload or directly use the upload button in the bottom left of vscode window
- To get the captures and upload them to google drive you can run python ESP32.py from the python folder and press k in the dialog window 

## Implementation
ESP32-CAM firmware consists of two threads running in parallel. The first one runs the handleClients where ESP32 works as a server and we can get a capture from the Python script using the dialog window showing the camera live-stream and pressing the k button on the keyboard. 
The second thread handles the traffic light control. First, it reads the data.json file from the git repo and deserializes it into JSON format.
After that, we read the specific direction and adjusted the green light timing accordingly.
For more details check the code. 
