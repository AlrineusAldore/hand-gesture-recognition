from segmentation.helpers import open_close
import info_handlers.stack as stk
from helpers import draw_contours
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def region_based_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return segmentate_from_channel(gray)


def segmentate_from_channel(channel_img):
    # channel_img is gray_img at first
    markers = np.zeros_like(channel_img)
    markers[channel_img < 60] = 2
    markers[channel_img > 150] = 1

    elevation_map = sobel(channel_img)  # Get edges

    # Fill edges
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    labeled_img, obj_count = ndi.label(segmentation)

    return elevation_map, labeled_img


def threshold_low_sat(img, thresh=10):
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    no_low_sat = cv2.inRange(sat, thresh, 255)
    sat_bin, sat_img = open_close(img, no_low_sat)

    return sat_bin, sat_img


def threshold_white_rgb(img, thresh=125):
    lower = np.array([thresh, thresh, thresh])
    upper = np.array([255, 255, 255])

    white_mask = cv2.inRange(img, lower, upper)
    non_white_mask = cv2.bitwise_not(white_mask)
    img_mask_res = cv2.bitwise_and(img, img, mask=non_white_mask)

    mask_bin = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1] #original bin included black as hand

    return mask_bin, img_mask_res


def threshold_dark_spots(img, thresh=40):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    no_dark_bin = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    no_dark_img = cv2.bitwise_and(img, img, mask=no_dark_bin)

    return no_dark_bin, no_dark_img




def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 50)

    img_contour = img.copy()
    img_canvas = img.copy()
    img_canvas[:] = 0,0,0

    sig_contours = draw_contours(edge, img_contour, img_canvas, draw_pts=False)

    stack = stk.Stack([img, gray, edge, img_contour, img_canvas])

    cv2.imshow("contours", stack.to_viewable_stack(2))





