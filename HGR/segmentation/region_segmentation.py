from segmentation.helpers import open_close
import stack.stack as stk
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


def threshold_white(img):
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    no_low_sat = cv2.inRange(sat, 10, 255)
    bin_sat_done, sat_done = open_close(img, no_low_sat)

    return bin_sat_done, sat_done



def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours_list, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_extr, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_comp, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours_tree, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours_types = [contours_list, contours_extr, contours_comp, contours_tree]
    #stack = stk.Stack([contours_list, contours_extr, contours_comp, contours_tree,
    #                   img, gray], size=(2, 4))
    for cnts in contours_types:
        sig_contours = draw_contours(img)

    cv2.imshow("contours", gray)





