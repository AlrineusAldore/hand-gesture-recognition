import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def region_based_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, val = hsv_img[:,:,0], hsv_img[:,:,2]

    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    val_hist = cv2.calcHist([val], [0], None, [256], [0, 256])
    #plot_hist(gray_hist, "gray")
    #plot_hist(val_hist, "val")

    variables = segmentate_from_channel(gray)
    variables2 = segmentate_from_channel(val)

    return variables, variables2


def segmentate_from_channel(channel_img):
    # channel_img is gray_img at first
    markers = np.zeros_like(channel_img)
    markers[channel_img < 30] = 2
    markers[channel_img > 150] = 1

    elevation_map = sobel(channel_img)
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    labeled_img, obj_count = ndi.label(segmentation)

    return elevation_map, labeled_img, obj_count



def plot_hist(hist, name):
    plt.cla()
    plt.plot(hist, color='gray', label=name)
    plt.title(f"{name}Min:{hist.argmin()}, {name}Max:{hist.argmax()}")

    plt.legend()
    plt.pause(0.001)
