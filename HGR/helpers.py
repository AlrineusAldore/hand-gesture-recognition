import cv2
import numpy as np
import time


def empty(a):
    pass


# Initialize the manual hsv values windows
def InitializeWindows():
    # Trackbar
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 380)
    cv2.createTrackbar("Hue1 Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Hue1 Max", "Trackbars", 14, 179, empty)
    cv2.createTrackbar("Hue2 Min", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("Hue2 Max", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", 80, 255, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)


# Get binary img and return it cropped without black sides, and return the radius of the white area
def autoCropBinImg(bin):
    white_pt_coords = np.argwhere(bin)

    crop = bin  # in case bin is completely black
    r = 0

    if cv2.countNonZero(bin) != 0:
        min_y = min(white_pt_coords[:, 0])
        min_x = min(white_pt_coords[:, 1])
        max_y = max(white_pt_coords[:, 0])
        max_x = max(white_pt_coords[:, 1])

        r = int(sum([max_y - min_y, max_x - min_x]) // 4)

        crop = bin[min_y:max_y, min_x:max_x]

    return crop, r


# Find contours from img, and draw them on imgContour and imgCanvas
def draw_contours(img, imgContour, imgCanvas):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    significant_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # to discard small random lines
            # print("cuntour: ", cnt)
            # print('area: ', area)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 2)
            cv2.drawContours(imgCanvas, cnt, -1, (255, 0, 0), 2)

            peri = cv2.arcLength(cnt, True)  # perimeter
            approx = cv2.approxPolyDP(cnt, 0.2 * peri, True)  # Points
            # print("approx: ", approx)
            cv2.drawContours(imgContour, approx, -1, (0, 255, 0), 3)
            cv2.drawContours(imgCanvas, approx, -1, (0, 255, 0), 3)

            significant_contours.append(cnt)

    return significant_contours


def timer(seconds, stage, not_started_clock):
    time.sleep(seconds)

    stage[0] += 1
    not_started_clock[0] = True
