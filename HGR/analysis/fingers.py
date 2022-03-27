import cv2
from helpers import autoCropBinImg
import math
import matplotlib.pyplot as plt
import numpy as np
from hand import point
from skimage.morphology import label
from skimage.measure import regionprops


def find_fingers(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    hh, ww = img.shape
    fingers = np.zeros(img.shape, img.dtype)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # to discard small random lines
            x,y,w,h = cv2.boundingRect(cnt)
            #Discard contour that touches bottom of image since it's not a finger
            if y+h != hh:
                cv2.fillPoly(fingers, pts=[cnt],color=255)
                count += 1

    return fingers, count


def getCircle(imgTransformed):
    h = imgTransformed.shape[0]
    w = imgTransformed.shape[1]

    thresh, centerImg = cv2.threshold(imgTransformed, 253, 255, cv2.THRESH_BINARY)

    center = findCenter(centerImg, h, w)

    crop, r = autoCropBinImg(imgTransformed)
    circle = cv2.circle(centerImg, center, r, 255, -1)

    return circle

def findCenter(centerImg, h, w):

    center = 0, 0
    breakNestedLoop = False
    for y in range(0, h):
        for x in range(0, w):
            if centerImg[y, x] != 0:
                center = x, y
                breakNestedLoop = True
                break
        if breakNestedLoop:
            break
    return center

#fings_img is binary img of only fingers after find_fingers()
def get_fingers_data(fings_img):
#    plt.clf()

    distanceList = []
    fings_img = cv2.resize(fings_img, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)

    label_img = label(fings_img)
    regions = regionprops(label_img)

#    fig, ax = plt.subplots()
    #ax.imshow(fings_img, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        x1 = x0 + math.cos(props['Orientation']) * 0.5 * props['MinorAxisLength']
        y1 = y0 - math.sin(props['Orientation']) * 0.5 * props['MinorAxisLength']
        x2 = x0 - math.sin(props['Orientation']) * 0.5 * props['MajorAxisLength']
        y2 = y0 - math.cos(props['Orientation']) * 0.5 * props['MajorAxisLength']

        p0 = point.Point(x0, y0)
        p1 = point.Point(x1, y1)
        p2 = point.Point(x2, y2)

        if(p0.distance_between_two_points(p1) > 6):
            #add to list tuple (finger_length, finger_width)
            distanceList += (p0.distance_between_two_points(p1), p2.distance_between_two_points(p1))

#            ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
#            ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
#            ax.plot(x0, y0, '.g', markersize=15)

#            minr, minc, maxr, maxc = props.bbox
#            bx = (minc, maxc, maxc, minc, minc)
#            by = (minr, minr, maxr, maxr, minr)
#            ax.plot(bx, by, '-b', linewidth=2.5)

#        ax.axis((0, 600, 600, 0))
        #plt.show()
    return distanceList

