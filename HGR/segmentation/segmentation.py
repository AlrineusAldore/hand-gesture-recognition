from segmentation.constants import *
from segmentation.helpers import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.signal as signal
from itertools import chain
import time


def histogram(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]

    # Makes histograms of h, s, v accordingly
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    start = time.time()
    #start_segmentation_plot(hist_h, hist_s, hist_v)
    end = time.time()
    #print("time for start of plot:", end-start)

    # Turns s (2d array) to a 1d array, chaining the end of each line with the start of the next one
    flattened_s_lst = list(chain.from_iterable(s.tolist()))
    y = []
    x = []
    # Get the values of s in the y & x axes
    for i in range(256):
        count_index = flattened_s_lst.count(i)
        y.append(count_index)
        x.append(i)

    # Smoothed out s graph
    yhat = savgol_filter(y, 21, 2)  # type: np.ndarray
    yhat2 = savgol_filter(yhat, 21, 2)  # type: np.ndarray

    # All local min & max points
    min_pts = np.array(signal.argrelmin(yhat2), dtype="uint8")
    max_pts = np.array(signal.argrelmax(yhat2), dtype="uint8")

    mn, mx = remove_useless_extreme_points(yhat2, min_pts, max_pts)
    s_start, s_end = find_min_between_max(yhat2, mn, mx)

    #start = time.time()
    #start_segmentation_plot(hist_h, hist_s, hist_v)
    #plt.plot(x, yhat, color='black', label="smooth s")
    #plt.plot(x, yhat2, color='orange', label="smoother s")
    #end_segmentation_plot((hist_h.argmax(), hist_h.argmin()), (s_end, s_start), (hist_v.argmax(), hist_v.argmin()))
    #end = time.time()
    #print("time for plot: ", end-start)

    return hist_h.argmax(), hist_h.argmin(), s_end, s_start, hist_v.argmax(), hist_v.argmin()


# f - function f(x)
def find_min_between_max(f, min_pts, max_pts):
    pts = sorted(min_pts + max_pts)
    # First and second highest max points
    first = f.argmax()
    second = f.argmin()  # Start with the lowest value
    lowest = first  # Start with the highest value

    # If there is only 1 max point then every value can be the hand
    if len(max_pts) < 2:
        return 0, 255

    # Check how many max points have values above NOT_MANY_PIXELS
    high_max_count = 0
    for x in max_pts:
        if f[x] > NOT_MANY_PIXELS:
            if x != first: # Get the last highest max that is not absolute max
                second = x
            high_max_count += 1

    # If all the other maxes are small (below NOT_MANY_PIXELS)
    if second == f.argmin():
        # Find second highest max point that is not adjacent to first highest
        prev = 0
        for x in max_pts:
            if f[x] > f[second] and f[x] < f[first] and prev != first:
                second = x

            # Only account for prev if there are multiple high points that are not first
            if high_max_count > 2:
                prev = x

    if first < second:
        left = first
        right = second
    else:
        left = second
        right = first

    # Get minimums to the left and right of the 2 max points
    left_i = pts.index(left)
    right_i = pts.index(right)

    # Get min point / zero point to the left of left max
    if left_i == 0:
        start = check_for_value(f, 0, end=left)
    else:
        start = pts[left_i - 1]
    # Get min point / zero point to the right of right max
    if right_i == len(pts) - 1 or True:
        end = check_for_value(f, 0, start=right)
    else:
        end = pts[right_i + 1]

    # Get lowest min between maxes
    for x in min_pts:
        if f[x] < f[lowest] and x > start and x < end:
            lowest = x

    # Return the range of the second max point as it's most likely the hand
    if first < second:
        return lowest, end
    else:
        return start, lowest


def check_for_value(f, value, start=0, end=256):
    """
    Function checks the first occurrence of value in function f(x) within the given range and returns it
    :param f: math function f(x) with x between 0 and 255
    :param value: wanted y value in function
    :param start: from when should we start
    :param end: when should we stop
    :return: the first x of the wanted y value
    """
    res = 0
    for x in range(start, end):
        if int(f[x]) == value:
            res = x
            break

    return res



# Remove any unnecessary extreme points with similar values
def remove_useless_extreme_points(f, min_pts, max_pts):
    new_mins = min_pts.tolist()[0]
    new_maxes = max_pts.tolist()[0]
    pts = sorted(new_mins+new_maxes)

    # Go through all pts except first and last
    for i in range(1, len(pts)-1):
        # If extreme points are really close to each other
        if pts[i] - pts[i-1] < 5 and pts[i+1] - pts[i] < 5:
            # And if the slope between them is small enough
            if slope(pts[i-1], f[i-1], pts[i], f[i]) < 0.25 and slope(pts[i], f[i], pts[i+1], f[i+1]) < 0.25:
                # Get rid of appropriate min/max depending on current point
                if pts[i] in new_mins:
                    a = new_mins
                    b = new_maxes
                elif pts[i] in new_maxes:
                    a = new_maxes
                    b = new_mins
                else:
                    continue

                a.remove(pts[i])  # Remove useless A point
                # Replace 2 B points with 1 B point in the middle
                mid = ((pts[i-1]+pts[i+1]))//2
                b.remove(pts[i+1])
                b[:] = [mid if x==pts[i-1] else x for x in b]  # Change pts[i-1]'s value to mid


    return new_mins, new_maxes


# Gets 2 points and returns the slope between them
def slope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)


# Gets rgb image and returns it without background
# Can choose how to cut background from img (constant values, from histogram, manually changeable)
def hsv_differentiation(img, is_histogram, set_manually):
    # Color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV
    h1Min = 0
    h1Max = 179
    sMin = 80
    sMax = 255
    vMin = 0
    vMax = 255
    h2Min = h1Min
    h2Max = h1Max

    if is_histogram:
        temp, temp, sMax, sMin, temp, temp = histogram(img)

    if set_manually and not is_histogram:
        h1Min = cv2.getTrackbarPos("Hue1 Min", "Trackbars")
        h1Max = cv2.getTrackbarPos("Hue1 Max", "Trackbars")
        h2Min = cv2.getTrackbarPos("Hue2 Min", "Trackbars")
        h2Max = cv2.getTrackbarPos("Hue2 Max", "Trackbars")
        sMin = cv2.getTrackbarPos("Sat Min", "Trackbars")
        sMax = cv2.getTrackbarPos("Sat Max", "Trackbars")
        vMin = cv2.getTrackbarPos("Val Min", "Trackbars")
        vMax = cv2.getTrackbarPos("Val Max", "Trackbars")


    # HSV
    lower = np.array([h1Min, sMin, vMin])
    upper = np.array([h1Max, sMax, vMax])
    lower2 = np.array([h2Min, sMin, vMin])
    upper2 = np.array([h2Max, sMax, vMax])

    # nomral
    img_mask = cv2.inRange(img_hsv, lower, upper)
    imgMaskRes = cv2.bitwise_and(img, img, mask=img_mask)
    # mask 2
    img_mask2 = cv2.inRange(img_hsv, lower2, upper2)
    imgMask2Res = cv2.bitwise_and(img, img, mask=img_mask2)

    # combined
    both_masks = cv2.bitwise_or(img_mask, img_mask2, mask=None)
    both_masks_res = cv2.bitwise_or(img, img, mask=both_masks)
    # bothMasksRes2 = cv2.bitwise_or(imgMaskRes, imgMask2Res, mask=None)

    # guassian
    guassian_mask = cv2.GaussianBlur(both_masks, (5, 5), cv2.BORDER_DEFAULT)
    guassianMaskRes = cv2.bitwise_and(img, img, mask=guassian_mask)

    # closing
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closing_mask = cv2.morphologyEx(both_masks, cv2.MORPH_CLOSE, closing_kernel)
    closing_mask_res = cv2.bitwise_and(img, img, mask=closing_mask)

    # opening
    opening_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_OPEN, opening_kernel)
    opening_mask_res = cv2.bitwise_and(img, img, mask=opening_mask)

    # use the masks to get avarage color
    average = cv2.mean(both_masks_res, both_masks)
    # Makes a new image for average color
    both_masks_average = img.copy()
    both_masks_average[:] = (average[0], average[1], average[2])

    # use the masks to get avarage color
    average = cv2.mean(closing_mask_res, closing_mask)
    closing_mask_average = img.copy()
    closing_mask_average[:] = (average[0], average[1], average[2])

    # use the masks to get avarage color
    average = cv2.mean(opening_mask_res, opening_mask)
    opening_mask_average = img.copy()
    opening_mask_average[:] = (average[0], average[1], average[2])


    return (img_hsv, opening_mask, opening_mask_res)



def find_max_color(img):
    # Number of bins
    LENGTH = 16
    WIDTH = 16
    HEIGHT = 16
    bins = [LENGTH, WIDTH, HEIGHT]

    # Range of bins
    ranges = [0, 256, 0, 256, 0, 256]
    # Array of Image
    images = [img]
    # Number of channels
    channels = [0, 1, 2]

    # Calculate the Histogram
    hist = cv2.calcHist(images, channels, None, bins, ranges)

    # sorted_index contains the indexes the
    sorted_index = np.argsort(hist.flatten())

    # 1-D index of the max color in histogram
    index = sorted_index[-1]

    # Getting the 3-D index from the 1-D index
    k = index / (WIDTH * HEIGHT)
    j = (index % (WIDTH * HEIGHT)) / WIDTH
    i = index - j * WIDTH - k * WIDTH * HEIGHT

    # Print the max RGB Value
    print("Max RGB Value is = ", [i * 256 / HEIGHT, j * 256 / WIDTH, k * 256 / LENGTH])
    max_color = img.copy()
    max_color[:] = (i * 256 / HEIGHT, j * 256 / WIDTH, k * 256 / LENGTH)

    return max_color
