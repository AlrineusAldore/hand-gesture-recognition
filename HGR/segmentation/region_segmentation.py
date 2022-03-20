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
    markers[channel_img < 30] = 2
    markers[channel_img > 150] = 1

    elevation_map = sobel(channel_img)  # Get edges

    # Fill edges
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    labeled_img, obj_count = ndi.label(segmentation)

    return elevation_map, labeled_img



def plot_hist(hist, name):
    plt.cla()
    plt.plot(hist, color='gray', label=name)
    plt.title(f"{name}Min:{hist.argmin()}, {name}Max:{hist.argmax()}")

    plt.legend()
    plt.pause(0.001)
