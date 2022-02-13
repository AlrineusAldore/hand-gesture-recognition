import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def region_based_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    #plot_gray(hist)

    markers = np.zeros_like(gray)
    markers[gray < 30] = 1
    markers[gray > 150] = 2

    elevation_map = sobel(gray)
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    labeled_img, _ = ndi.label(segmentation)

    return elevation_map, segmentation, labeled_img


def plot_gray(hist):
    plt.cla()
    plt.plot(hist, color='gray', label="gray")
    plt.title(f"grayMin:{hist.argmin()}, grayMax:{hist.argmax()}")

    plt.legend()
    plt.pause(0.001)
