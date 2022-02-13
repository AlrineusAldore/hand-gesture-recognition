from segmentation.constants import *
from segmentation.helpers import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.signal as signal
from itertools import chain
import time


def histogram(img, plot_histo):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

    # Makes histograms of h, s, v accordingly
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

    if plot_histo:
        start_segmentation_plot(hist_h, hist_s, hist_v)

    h_start, h_end = analyze_histogram(hist_h, plot_histo, 'purple', "smooth h")
    s_start, s_end = analyze_histogram(hist_s, plot_histo, 'orange', "smooth s")
    v_start, v_end = analyze_histogram(hist_v, plot_histo, 'cyan', "smooth v")

    if plot_histo:
        #start = time.time()
        end_segmentation_plot((h_start, h_end), (s_start, s_end), (v_start, v_end))
        #end = time.time()
        #print("time for plot: ", end-start)

    return h_start, h_end, s_start, s_end, v_start, v_end



def analyze_histogram(hist, plot_histo, color, plot_name):
    # First flatten hist since every y value is in a list for some reason. Then turn all the values from ints to floats
    y = np.int_(list(chain.from_iterable(hist.tolist())))

    # Smooth out hist
    yhat = savgol_filter(y, 21, 2)  # type: np.ndarray
    yhat2 = savgol_filter(yhat, 21, 2)  # type: np.ndarray

    if plot_histo:
        x = list(range(0, 256))
        plt.plot(x, yhat2, color=color, label=plot_name)

    mins, maxes = get_useful_extrema(yhat2)
    start, end = find_min_between_max(yhat2, mins, maxes)

    return start, end



# f - function f(x)
def find_min_between_max(f, min_pts, max_pts):
    pts = sorted(min_pts + max_pts)
    # First and second highest max points
    abs_max = f.argmax()
    second_highest = f.argmin()  # Start with the lowest value
    lowest = abs_max  # Start with the highest value

    # Check how many max points have values above NOT_MANY_PIXELS
    high_max_count = 0
    for x in max_pts:
        if f[x] > NOT_MANY_PIXELS:
            if x != abs_max: # Get the last highest max that is not absolute max
                second_highest = x
            high_max_count += 1

    # If there is only 1 significant max point then every value can be the hand
    if len(max_pts) < 2 or high_max_count < 2:
        return get_range_of_max(f, abs_max, pts)

    # If all the other maxes are small (below NOT_MANY_PIXELS)
    if second_highest == f.argmin():
        # Find second highest max point that is not adjacent to first highest
        prev = 0
        for x in max_pts:
            if f[x] > f[second_highest] and f[x] < f[abs_max] and prev != abs_max:
                second_highest = x

            # Only account for prev if there are multiple high points that are not first
            if high_max_count > 2:
                prev = x

    if abs_max < second_highest:
        left = abs_max
        right = second_highest
    else:
        left = second_highest
        right = abs_max

    # Get minimums to the left and right of the 2 max points
    left_i = pts.index(left)
    right_i = pts.index(right)

    # Get min point / zero point to the left of left max
    if left_i == 0:
        start = check_for_value(f, 0, end=left)
    else:
        start = pts[left_i - 1]
    # Get min point / zero point to the right of right max
    if right_i == len(pts) - 1:
        end = check_for_value(f, 0, start=right)
    else:
        end = pts[right_i + 1]

    # Get lowest min between maxes
    for x in min_pts:
        if f[x] < f[lowest] and x > start and x < end:
            lowest = x

    # Return the range of the second max point as it's most likely the hand
    if abs_max < second_highest:
        return lowest, end
    else:
        return start, lowest



# Gets the range of a max point (left of it to right of it)
def get_range_of_max(f, max, pts):
    max_i = pts.index(max)

    # Get min point / zero point to the left of max (whichever is closest to max)
    start = check_for_value(f, 0, end=max, go_backwards=True)
    if max_i != 0:
        left_min = pts[max_i - 1]
        if left_min > start:
            start = left_min

    # Get min point / zero point to the right of max (whichever is closest to max)
    end = check_for_value(f, 0, start=max)
    if max_i != len(pts) - 1:
        right_min = pts[max_i + 1]
        if right_min < end:
            end = right_min

    return start, end




def check_for_value(f, value, start=0, end=256, go_backwards=False):
    """
    Function checks the first occurrence of value in function f(x) within the given range and returns it
    :param f: math function f(x) with x between 0 and 255
    :param value: wanted y value in function
    :param start: from when should we start
    :param end: when should we stop
    :param go_backwards: Whether to check the first from the start or first from the end
    :return: the first x of the wanted y value
    """
    if go_backwards:
        res = start
        for x in range(start, end):
            if int(f[end-x]) == value:
                res = end-x
                break
    else:
        res = end
        for x in range(start, end):
            if int(f[x]) == value:
                res = x
                break

    return res



# Remove any unnecessary extreme points with similar values
def get_useful_extrema(f):
    # All local min & max points
    min_pts = np.array(signal.argrelmin(f), dtype="uint8")
    max_pts = np.array(signal.argrelmax(f), dtype="uint8")

    new_mins = min_pts.tolist()[0]
    new_maxes = max_pts.tolist()[0]
    pts = sorted(new_mins+new_maxes)

    # Go through all pts except first and last
    for i in range(1, len(pts)-1):
        # If extreme points are really close to each other
        if pts[i] - pts[i-1] < 5 and pts[i+1] - pts[i] < 5:
            # And if the slope between them is small enough
            if abs(slope(pts[i-1], pts[i], f=f)) < SMALL_SLOPE and abs(slope(pts[i], pts[i+1], f=f)) < SMALL_SLOPE:
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

    # If endpoints are found to be significant min/max, then add them to new_mins/new_maxes accordingly
    end_mins, end_maxes = check_for_endpoints_extrema(f)

    for x in end_mins:
        index = 0
        if x == 255:
            index = len(new_mins)
        new_mins.insert(index, x)

    for x in end_maxes:
        index = 0
        if x == 255:
            index = len(new_mins)
        new_mins.insert(index, x)


    return new_mins, new_maxes




def check_for_endpoints_extrema(f):
    """
    Checks if endpoints of f(x) are significant min/max and return them if they are
    :param f: Math function f(x)
    :return: significant min/max endpoints in the form of (mins=[], maxes=[])
    """
    maxes = []
    mins = []

    #Get slopes of endpoints with points that has x differences of 3
    start_slope = slope(0, 3, f=f)
    end_slope = slope(252, 255, f=f)

    # Check if start is a significant min/max
    if start_slope > SMALL_SLOPE*2:
        mins.append(0)
    elif start_slope < SMALL_SLOPE*(-2):
        maxes.append(0)

    #Check if end is a significant min/max
    if end_slope > SMALL_SLOPE*2:
        maxes.append(255)
    elif end_slope < SMALL_SLOPE*(-2):
        mins.append(255)

    return mins, maxes



# Gets rgb image and returns it without background
# Can choose how to cut background from img (constant values, from histogram, manually changeable)
def hsv_differentiation(img, is_histogram, set_manually, is_val):
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

    if is_val:
        h1Min, h1Max, sMin, sMax, vMin, vMax = histogram(img, True)
    elif is_histogram:
        h1Min, h1Max, sMin, sMax, temp, temp = histogram(img, PLOT_HISTOGRAMS)

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

    # use the masks to get average color
    average = cv2.mean(both_masks_res, both_masks)
    # Makes a new image for average color
    both_masks_average = img.copy()
    both_masks_average[:] = (average[0], average[1], average[2])

    # use the masks to get average color
    average = cv2.mean(closing_mask_res, closing_mask)
    closing_mask_average = img.copy()
    closing_mask_average[:] = (average[0], average[1], average[2])

    # use the masks to get average color
    average = cv2.mean(opening_mask_res, opening_mask)
    opening_mask_average = img.copy()
    opening_mask_average[:] = (average[0], average[1], average[2])


    return (img_hsv, opening_mask, opening_mask_res)
