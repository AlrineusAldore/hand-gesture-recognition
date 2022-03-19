from segmentation.constants import *
from segmentation.helpers import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from itertools import chain
import time


def histogram(img, plot_histo, colors_space):
    """
    :param img: converted color space img
    :param plot_histo: to plot a histogram or not
    :param colors_space: name of the colors space
    :return: range of values
    """
    val1, val2, val3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Makes histograms of h, s, v accordingly
    hist1 = cv2.calcHist([val1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([val2], [0], None, [256], [0, 256])
    hist3 = cv2.calcHist([val3], [0], None, [256], [0, 256])

    if plot_histo:
        start_segmentation_plot(hist1, hist2, hist3, colors_space)

    start1, end1 = analyze_histogram(hist1, plot_histo, 'purple', "smooth "+colors_space[0])
    start2, end2 = analyze_histogram(hist2, plot_histo, 'orange', "smooth "+colors_space[1])
    start3, end3 = analyze_histogram(hist3, plot_histo, 'cyan', "smooth "+colors_space[2])

    if plot_histo:
        #start = time.time()
        end_segmentation_plot((start1, end1), (start2, end2), (start3, end3))
        #end = time.time()
        #print("time for plot: ", end-start)

    return start1, end1, start2, end2, start3, end3



def analyze_histogram(hist, plot_histo, color, plot_name):
    # First flatten hist since every y value is in a list for some reason. Then turn all the values from ints to floats
    y = np.int_(list(chain.from_iterable(hist.tolist())))

    # Smooth out hist
    yhat = savgol_filter(y, 21, 2)  # type: np.ndarray
    yhat2 = savgol_filter(yhat, 21, 2)  # type: np.ndarray

    if plot_histo:
        x = list(range(0, 256))
        plt.plot(x, yhat2, color=color, label=plot_name)

    minima, maxima = get_useful_extrema(yhat2)
    start, end = find_min_between_max(yhat2, minima, maxima)
    start2, end2 = get_highest_max_range(yhat2, minima, maxima)

    return start, end



def get_highest_max_range(f, min_pts, max_pts):
    pts = sorted(min_pts + max_pts)
    abs_max = f.argmax()
    start, end = get_range_of_max(f, abs_max, pts, max_pts)

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

    # If there is only 1 significant max point then every non-zero value can be the hand
    if len(max_pts) < 2 or high_max_count < 2:
        return get_range_of_max(f, abs_max, pts, max_pts)

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

    start, nothing = get_range_of_max(f, left, pts, max_pts)
    nothing, end = get_range_of_max(f, right, pts, max_pts)

    # Get lowest min between maxes
    for x in min_pts:
        if f[x] < f[lowest] and x > start and x < end:
            lowest = x

    # Return the range of the highest max point as it's most likely the hand
    if abs_max > second_highest:
        return lowest, end
    else:
        return start, lowest




def lab_segmentation(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)






# Gets rgb image and returns it without background
# Can choose how to cut background from img (constant values, from histogram, manually changeable)
def hsv_differentiation(img, is_histo=False, manually=False, is_val=False, has_params=False, params=None, get_range=False, seg_type=0):
    # Color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    if seg_type == 1:
        img2 = lab_img
    elif seg_type == 2:
        img2 = img.copy()
    else:
        img2 = img_hsv

    # HSV
    h1Min = 0
    h1Max = 179
    sMin = 80
    sMax = 255
    vMin = 0
    vMax = 255
    h2Min = h1Min
    h2Max = h1Max

    if has_params:
        h1Min = params[0]
        h1Max = params[1]
        sMin = params[2]
        sMax = params[3]
        vMin = params[4]
        vMax = params[5]
        h2Min = h1Min
        h2Max = h1Max
    elif seg_type == 1:
        h1Min, h1Max, sMin, sMax, vMin, vMax = histogram(lab_img, PLOT_HISTOGRAMS, 'lab')
        h2Min = h1Min
        h2Max = h1Max
    elif seg_type == 2:
        h1Min, h1Max, sMin, sMax, vMin, vMax = histogram(img, PLOT_HISTOGRAMS, 'rgb')
        h2Min = h1Min
        h2Max = h1Max
    elif is_val:
        h1Min, h1Max, sMin, sMax, vMin, vMax = histogram(img_hsv, True, 'hsv')
        h2Min = h1Min
        h2Max = h1Max
    elif is_histo:
        h1Min, h1Max, sMin, sMax, vMin, vMax = histogram(img_hsv, PLOT_HISTOGRAMS, 'hsv')
        h2Min = h1Min
        h2Max = h1Max

    if manually and not is_histo:
        h1Min = cv2.getTrackbarPos("Val1 Min", "Trackbars")
        h1Max = cv2.getTrackbarPos("Val1 Max", "Trackbars")
        h2Min = cv2.getTrackbarPos("Hue2 Min", "Trackbars")
        h2Max = cv2.getTrackbarPos("Hue2 Max", "Trackbars")
        sMin = cv2.getTrackbarPos("Val2 Min", "Trackbars")
        sMax = cv2.getTrackbarPos("Val2 Max", "Trackbars")
        vMin = cv2.getTrackbarPos("Val3 Min", "Trackbars")
        vMax = cv2.getTrackbarPos("Val3 Max", "Trackbars")

    range = (h1Min, h1Max, sMin, sMax, vMin, vMax)

    opening_mask, opening_mask_res = mask_range(img, img2, range, (h2Min, h2Max))

    result = [img2, opening_mask, opening_mask_res]

    if get_range:
        result.append(range)

    return tuple(result)



def mask_range(img, img2, range, hue2):
    h1Min = range[0]
    h1Max = range[1]
    sMin = range[2]
    sMax = range[3]
    vMin = range[4]
    vMax = range[5]
    h2Min = hue2[0]
    h2Max = hue2[1]

    # HSV
    lower = np.array([h1Min, sMin, vMin])
    upper = np.array([h1Max, sMax, vMax])
    lower2 = np.array([h2Min, sMin, vMin])
    upper2 = np.array([h2Max, sMax, vMax])

    # nomral
    img_mask = cv2.inRange(img2, lower, upper)
    imgMaskRes = cv2.bitwise_and(img, img, mask=img_mask)
    # mask 2
    img_mask2 = cv2.inRange(img2, lower2, upper2)
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

    return opening_mask, opening_mask_res




def get_square(img, color):
    divisor = 2.5
    height, width = img.shape[:2]
    h, w = int(height//divisor), int(width//divisor)

    square_img = img.copy()
    square_img = cv2.rectangle(square_img, (0 ,height-h), (w, height), color, 1)
    small = img[height-h:height, 0:w]

    return square_img, small



# Computes the average value of each column
def compute_best_range(ranges):
    new_arr = []
    ncols = 6
    nrows = len(ranges)
    print("\nnum of ranges:", nrows)

    try:
        for col in range(ncols):
            new_arr.append([])
            for row in range(nrows):
                new_arr[col].append(ranges[row][col])
    except Exception as e:
        print("e:", repr(e))

    print("new_arr:")
    avrg = []
    edges = []
    is_min = True
    for row in new_arr:
        print(row)
        avrg.append(sum(row)/len(row))
        if is_min:
            edges.append(min(row))
        else:
            edges.append(max(row))
        is_min = not is_min

    print("avrg:\n", avrg)
    print("edges:\n", edges)

    return avrg, edges
