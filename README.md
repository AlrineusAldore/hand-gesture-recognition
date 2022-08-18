# Hand Gesture Recognition
A Python 3 image recognition project that recognizes hand gestures and performs different actions accordingly.


## General Info
This is a training project that I took upon myself with a partner in order to learn about image processing and the math and algorithms behind it.
This program recognizes hand gestures shown to the camera and outputs a command to the computer based on the gesture made. I worked on this project because I loved the idea of controlling a computer from afar without touching anything, only using your hand gestures. The project also involves math in it which I quite like and want to better my programming implementation of it.


## Implementation
The program is divided it into 3 main stages: Segmentation, Hand Analysis, and Data Analysis.

### Segmentation
First, the program finds the hand and segments it from the background. To do this we must first let the program automatically configures itself based on the skin color and lighting of the area. This happens through multiple stages.

#### Stage 1:
  1. The user stands still with their arm not in the frame and lets the program save all the frames and calculate a weighted average grayscaled image of all the frames.
  2. Now with the user's arm in the frame, subtract the average image from current frame to get a rough estimate of the arm & noise.
  3. Biggest contour above a threshold is the arm.

#### Stage 2:
  1. The user puts their hand in a box so that only the skin color is inside.
  2. Pixels inside the box go to histograms of hue, saturation and value.
  3. Use algorithms to determine range of hsv of the skin color based on analyzing the given hsv functions in the histograms.

#### Stage 3:
  1. Get hand from hsv of skin color and overlap of arm from stage 1 to get accurate segmented hand that isn't affected by background as much.
  2. Get rid of all remaining noise through repetition of noise reduction algorithms in certain order.

### Hand Analysis
This part is about finding as much information about the hand as possible.
Currently the information is:
* Width and height of hand and fingers
* Number of fingers raised (calculated with finger width in case of no space between fingers)
* Angles of fingers and between fingers
* Hand keypoints

### Data Analysis
Using the data from the hand analysis and comparing it with a dataset of hand gestures. The most probable result above a threshold will be the resulted hand gesture and would activate the linked command accordingly.
Planning to implement this part with ML.


## Technologies and libraries
* python
* opencv
* numpy
* imutils
* matplotlib
* skimage
* scipy
* cython


## Status
Currently the project is still in progress and not finished
