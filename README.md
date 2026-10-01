# Fish Detector

A Python program that detects and tracks multiple fish in a video based on their color.

The program uses **OpenCV** to process video frames, identify fish based on HSV color thresholds, detect their positions, and draw their movement trajectories.

The detection parameters, including HSV thresholds and tracking settings, are stored in a separate JSON configuration file so they can be modified without changing the Python code.

## Features

* Detects multiple fish in a video.
* Supports three fish colors:

  * Orange (`o`)
  * Red (`r`)
  * Yellow (`y`)
* Uses HSV color thresholds to isolate fish from the background.
* Detects fish using contours.
* Tracks fish positions across video frames.
* Draws a trajectory for each detected fish.
* Displays the number of fish detected in each frame.
* Calculates the percentage of frames in which the expected number of fish was detected.
* Stores detection parameters in a JSON configuration file.

## Project Structure

```text
fish-detector/
│
├── detections_multiples.py
├── config.json
│
└── test_videos/
    └── video files
```

### `detections_multiples.py`

Contains the main program and the `detect_fishes()` function responsible for:

1. Reading the video.
2. Converting frames to HSV.
3. Creating a color mask.
4. Detecting contours.
5. Identifying fish centers.
6. Tracking fish between frames.
7. Drawing tracking paths.
8. Displaying detection results.

### `fish_colors.json`

Contains the HSV thresholds and other detection parameters.
The HSV thresholds can be adjusted if the program has difficulty detecting fish under different lighting or video conditions.

## Requirements

Python 3 is required.

The following Python packages are used:

* OpenCV
* NumPy
* imutils
* seaborn

Install the dependencies with:

```bash
pip install opencv-python numpy imutils seaborn
```

## Usage

Place the videos you want to analyze in the `test_videos/` directory.

Run the program with:

```bash
python detections_multiples.py
```

The program will ask for three inputs:

```text
-----------------
| FISH DETECTOR |
-----------------

Enter video filename:
Enter color ('r', 'o' or 'y'):
Enter number of fishes you wish to detect:
```

For example:

```text
Enter video filename: fish_video.mp4
Enter color ('r', 'o' or 'y'): o
Enter number of fishes you wish to detect: 3
```

The program will then display the video with detected fish and their movement trajectories.

At the end of the video, the program prints detection statistics such as:

```text
Frames detecting 3 fishes: 850 out of 1000 (85.0 %)
Frames detecting 2 fishes: 120 out of 1000 (12.0 %)
```

These values indicate how often the expected number of fish, or one fewer fish, was detected.

For a demonstration of the program, see this video: https://www.youtube.com/watch?v=W4X8g2lsMcA

## Limitations

The detection method relies primarily on color segmentation. Its performance can therefore be affected by:

* Changes in lighting
* Reflections
* Background objects with similar colors
* Fish overlapping each other
* Fish moving too quickly between frames
* Changes in camera position
* Poorly selected HSV thresholds

The current tracking method also uses distance between detected centers to associate fish between frames. When fish are very close to one another, their identities may become difficult to distinguish.

## Reference

The tracking approach is based on the following OpenCV object-tracking example:

[PyImageSearch — OpenCV Track Object Movement](https://pyimagesearch.com/2015/09/21/opencv-track-object-movement/)

