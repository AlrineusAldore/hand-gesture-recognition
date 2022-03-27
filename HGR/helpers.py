import cv2
import numpy as np
import time
from inspect import currentframe


def empty(a):
    pass

def get_line_num():
    return "line num: " + str(currentframe().f_back.f_lineno)

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
def draw_contours(img, img_contour=None, img_canvas=None, retrieval_method=cv2.RETR_EXTERNAL, draw_pts=True, min_area=20, contours=None):
    if contours is None:
        contours, hierarchy = cv2.findContours(img, retrieval_method, cv2.CHAIN_APPROX_NONE)
    if img_contour is None:
        img_contour = img.copy()
    if img_canvas is None:
        img_canvas = get_blank_img(img)

    significant_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:  # to discard small random lines
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 2)
            cv2.drawContours(img_canvas, cnt, -1, (255, 0, 0), 2)

            if draw_pts:
                peri = cv2.arcLength(cnt, True)  # perimeter
                approx = cv2.approxPolyDP(cnt, 0.2 * peri, True)  # Points
                cv2.drawContours(img_contour, approx, -1, (0, 255, 0), 3)
                cv2.drawContours(img_canvas, approx, -1, (0, 255, 0), 3)

            significant_contours.append(cnt)

    return significant_contours


def normalize_zero1_to_zero255(img):
    return (img*255).astype(np.uint8)

def get_gray_blurred_img(img, blur=(1,1)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, blur, 0)

def get_blank_img(img):
    blank_img = np.zeros(img.shape, img.dtype)
    return blank_img

def get_biggest_object(img, thresh=1):
    # if img is rgb then turn it to gray
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    bin = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    biggest_object_bin = get_blank_img(bin)

    cnts, _ = cv2.findContours(bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if there are contours, get biggest object
    if cnts is not None and len(cnts) > 0:
        # based on contour area, get the maximum contour which is the biggest object
        segmented = max(cnts, key=cv2.contourArea)
        cv2.fillPoly(biggest_object_bin, pts=[segmented], color=255)

    return biggest_object_bin



def timer(seconds, stage, not_started_clock):
    time.sleep(seconds)

    stage[0] += 1
    not_started_clock[0] = True
