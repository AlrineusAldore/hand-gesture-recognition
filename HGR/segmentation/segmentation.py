import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from itertools import chain


def histogram(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]

    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    plt.cla()
    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")

    flatten_list = list(chain.from_iterable(s.tolist()))
    count_index_dict = []
    y = []
    x = []
    for i in range(256):
        count_index = flatten_list.count(i)
        y.append(count_index)
        x.append(i)
        count_index_dict.append([len(count_index_dict), i, count_index])

    yhat = savgol_filter(y, 51, 3)  # type: np.ndarray
    plt.plot(x, yhat, color='black', label="smooth s green")
    s_local_min = 0
    for i in range(len(yhat)):
        # print(i, yhat.tolist()[i])
        if (i < len(yhat) - 1
                and yhat.tolist()[i] > 0 and yhat.tolist()[i - 1] > yhat.tolist()[i]
                and yhat.tolist()[i + 1] > yhat.tolist()[i]):
            s_local_min = i

    plt.title(f"hMax:{hist_h.argmax()}, hMin:{hist_h.argmin()}, sMax:{hist_s.argmax()}, sMin:{s_local_min}, vMax:{hist_v.argmax()}, vMin:{hist_v.argmin()}")

    plt.legend()
    plt.pause(0.001)
    return hist_h.argmax(), hist_h.argmin(), hist_s.argmax(), s_local_min, hist_v.argmax(), hist_v.argmin()


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
        h1Max, h1Min, sMax, sMin, vMax, vMin = histogram(img)

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
