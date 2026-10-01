"""
detections_multiples.py: detects fishes in videos

Based on example from https://pyimagesearch.com/2015/09/21/opencv-track-object-movement/

Authors: Corinne Dumais, Benjamin Méthot, Marc-Olivier Lachance, Laurence Fontaine
Date: December 2022
"""

import json
from collections import deque
from math import dist

import cv2
import imutils
import numpy as np
import seaborn


# Load configuration
with open("config.json", "r") as file:
    config = json.load(file)

fish_colors = config["fish_colors"]
detection = config["detection"]


def detect_fishes(video_filename: str, fish_color: str, fish_number: int):
    """
    Detect fishes in a video with given parameters.

    :param video_filename: Name of the file containing the video.
    :param fish_color: Color of fishes to detect.
                       Must be 'r', 'o' or 'y'.
    :param fish_number: Number of fishes to detect.
    """

    assert fish_color in fish_colors, (
        "Fish color must be either red ('r'), orange ('o') or yellow ('y')"
    )
    assert fish_number > 0, (
        "Number of fishes to detect must be greater than 0"
    )

    # Get detection parameters
    buffer = detection["tracking_buffer"]
    minimum_radius = detection["minimum_radius"]
    maximum_tracking_distance = detection["maximum_tracking_distance"]
    display_width = detection["display_width"]
    display_height = detection["display_height"]
    wait_time = detection["wait_time"]

    # Get HSV thresholds for selected fish color
    hsv_lower = tuple(fish_colors[fish_color]["lower"])
    hsv_upper = tuple(fish_colors[fish_color]["upper"])

    # Store tracking points
    pts = [deque(maxlen=buffer) for _ in range(fish_number)]

    # Get video
    folder = "test_videos/"
    vs = cv2.VideoCapture(f"{folder}{video_filename}")

    frames_good = 0
    frames_less_good = 0
    frames_total = 0

    # Iterate over all frames
    while True:
        ret, frame = vs.read()

        if not ret:
            break

        # Get mask based on HSV
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)

        # Apply morphological transformation to clean up mask
        kernel = np.ones((3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Get all contours and sort in descending order
        cnts = cv2.findContours(
            mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        origins = []
        radii = []
        centers = []

        # Process the largest contours
        for i in range(min(fish_number, len(cnts))):
            contour = cnts[i]

            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            moments = cv2.moments(contour)

            # Avoid division by zero
            if moments["m00"] == 0:
                continue

            center = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"])
            )

            if radius > minimum_radius:
                origins.append((x, y))
                radii.append(radius)
                centers.append(center)

        # Update fish tracking
        if all(radius > 5 for radius in radii):
            if all(len(point) > 0 for point in pts):
                distances = np.zeros((len(centers), len(centers)))

                for i in range(len(centers)):
                    for j in range(len(centers)):
                        distances[i][j] = dist(
                            centers[i],
                            pts[j][0]
                        )

                for i in range(len(centers)):
                    min_index = np.argmin(distances[i])

                    cv2.circle(
                        frame,
                        (int(origins[i][0]), int(origins[i][1])),
                        int(radii[i]),
                        (0, 255, 255),
                        2
                    )

                    cv2.circle(
                        frame,
                        centers[i],
                        5,
                        (0, 0, 255),
                        -1
                    )

                    pts[min_index].appendleft(centers[i])

            else:
                for i in range(len(centers)):
                    cv2.circle(
                        frame,
                        (int(origins[i][0]), int(origins[i][1])),
                        int(radii[i]),
                        (0, 255, 255),
                        2
                    )

                    cv2.circle(
                        frame,
                        centers[i],
                        5,
                        (0, 0, 255),
                        -1
                    )

                    pts[i].appendleft(centers[i])

        # Create colors for tracking paths
        list_colors = [
            tuple(
                int(255 * value)
                for value in seaborn.color_palette(
                    "rocket",
                    n_colors=fish_number
                )[j]
            )
            for j in range(fish_number)
        ]

        # Draw tracking paths
        for color, point_history in enumerate(pts):
            for i in np.arange(1, len(point_history)):

                # Ignore missing tracking points
                if (
                    point_history[i - 1] is None
                    or point_history[i] is None
                ):
                    continue

                # Stop drawing if the fish moved too far
                if (
                    dist(
                        point_history[i - 1],
                        point_history[i]
                    )
                    > maximum_tracking_distance
                ):
                    break

                thickness = int(
                    np.sqrt(buffer / float(i + 1)) * 2.5
                )

                cv2.line(
                    frame,
                    point_history[i - 1],
                    point_history[i],
                    list_colors[color],
                    thickness
                )

        # Count detection accuracy
        if len(centers) == fish_number:
            frames_good += 1

        if len(centers) == fish_number - 1:
            frames_less_good += 1

        frames_total += 1

        # Display number of detected fish
        cv2.putText(
            frame,
            f"Fishes detected: {len(centers)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

        # Display frame
        cv2.imshow(
            "Frame",
            cv2.resize(
                frame,
                (display_width, display_height)
            )
        )

        key = cv2.waitKey(wait_time) & 0xFF

        # Stop if the 'q' key is pressed
        if key == ord("q"):
            break

    # Print detection statistics
    if frames_total > 0:
        print(
            f"Frames detecting {fish_number} fishes: "
            f"{frames_good} out of {frames_total} "
            f"({(frames_good / frames_total) * 100:.1f} %)"
        )

        print(
            f"Frames detecting {fish_number - 1} fishes: "
            f"{frames_less_good} out of {frames_total} "
            f"({(frames_less_good / frames_total) * 100:.1f} %)"
        )

    # Release video and close windows
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("-----------------")
    print("| FISH DETECTOR |")
    print("-----------------")
    print()

    filename = input("Enter video filename: ")
    color = input("Enter color ('r', 'o' or 'y'): ")
    number = int(
        input("Enter number of fishes you wish to detect: ")
    )

    print()

    detect_fishes(filename, color, number)