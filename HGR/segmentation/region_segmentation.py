from segmentation.helpers import open_close
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
