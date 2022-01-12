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

    # splot  = sns.distplot(a=h, color='r', label="h", hist=False, kde=True,
    #             kde_kws={'shade': True, 'linewidth': 3})
    # sns.distplot(a=s, color='g', label="s", hist=False, kde=True,
    #             kde_kws={'shade': True, 'linewidth': 3})
    # sns.distplot(a=v, color='b', label="v", hist=False, kde=True,
    #             kde_kws={'shade': True, 'linewidth': 3})

    # [h.get_height() for h in splot.patches]
    plt.legend()
    plt.pause(0.001)
    return hist_h.argmax(), hist_h.argmin(), hist_s.argmax(), s_local_min, hist_v.argmax(), hist_v.argmin()


# Gets rgb image and returns it without background
# Can choose how to cut background from img (constant values, from histogram, manually changeable)
def hsv_differentiation(img, is_histogram, set_manually):
    # Color
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
    imgMask = cv2.inRange(imgHSV, lower, upper)
    imgMaskRes = cv2.bitwise_and(img, img, mask=imgMask)
    # mask 2
    imgMask2 = cv2.inRange(imgHSV, lower2, upper2)
    imgMask2Res = cv2.bitwise_and(img, img, mask=imgMask2)

    # combined
    bothMasks = cv2.bitwise_or(imgMask, imgMask2, mask=None)
    bothMasksRes = cv2.bitwise_or(img, img, mask=bothMasks)
    # bothMasksRes2 = cv2.bitwise_or(imgMaskRes, imgMask2Res, mask=None)

    # guassian
    guassianMask = cv2.GaussianBlur(bothMasks, (5, 5), cv2.BORDER_DEFAULT)
    guassianMaskRes = cv2.bitwise_and(img, img, mask=guassianMask)

    # closing
    closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    openingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closingMask = cv2.morphologyEx(bothMasks, cv2.MORPH_CLOSE, closingKernel)
    closingMaskRes = cv2.bitwise_and(img, img, mask=closingMask)

    # opening
    openingMask = cv2.morphologyEx(closingMask, cv2.MORPH_OPEN, openingKernel)
    openingMaskRes = cv2.bitwise_and(img, img, mask=openingMask)

    # Edges after color
    grayColor = cv2.cvtColor(openingMaskRes, cv2.COLOR_BGR2GRAY)
    ret, binaryColor = cv2.threshold(grayColor, 20, 255, cv2.THRESH_BINARY)

    # use the masks to get avarage color
    average = cv2.mean(bothMasksRes, bothMasks)
    # Makes a new image for average color
    bothMasksAverage = img.copy()
    bothMasksAverage[:] = (average[0], average[1], average[2])

    # use the masks to get avarage color
    average = cv2.mean(closingMaskRes, closingMask)
    closingMaskAverage = img.copy()
    closingMaskAverage[:] = (average[0], average[1], average[2])

    # use the masks to get avarage color
    average = cv2.mean(openingMaskRes, openingMask)
    openingMaskAverage = img.copy()
    openingMaskAverage[:] = (average[0], average[1], average[2])

    stack2 = [[bothMasks, bothMasksRes, bothMasksAverage],
                               [closingMask, closingMaskRes, closingMaskAverage],
                               [openingMask, openingMaskRes, openingMaskAverage]]


    return (imgHSV, openingMask, openingMaskRes)



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

    # sortedIndex contains the indexes the
    sortedIndex = np.argsort(hist.flatten())

    # 1-D index of the max color in histogram
    index = sortedIndex[-1]

    # Getting the 3-D index from the 1-D index
    k = index / (WIDTH * HEIGHT)
    j = (index % (WIDTH * HEIGHT)) / WIDTH
    i = index - j * WIDTH - k * WIDTH * HEIGHT

    # Print the max RGB Value
    print("Max RGB Value is = ", [i * 256 / HEIGHT, j * 256 / WIDTH, k * 256 / LENGTH])
    maxColor = img.copy()
    maxColor[:] = (i * 256 / HEIGHT, j * 256 / WIDTH, k * 256 / LENGTH)

    return maxColor
