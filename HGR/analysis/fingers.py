<<<<<<< HEAD
import cv2
import numpy as np
from helpers import autoCropBinImg
from skimage.measure import regionprops
=======
import math
import cv2
import numpy as np
from helpers import autoCropBinImg
from hand import point
import matplotlib.pyplot as plt

from skimage.draw import ellipse
from skimage.morphology import label
from skimage.measure import regionprops
from scipy.ndimage import geometric_transform
>>>>>>> develop


def find_fingers(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    hh, ww = img.shape
<<<<<<< HEAD
    fingers = np.zeros(img.shape, img.dtype)
=======
    fingers_img = np.zeros(img.shape, img.dtype)
>>>>>>> develop

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # to discard small random lines
            x,y,w,h = cv2.boundingRect(cnt)
            #Discard contour that touches bottom of image since it's not a finger
            if y+h != hh:
<<<<<<< HEAD
                cv2.fillPoly(fingers, pts=[cnt],color=255)
                count += 1

    #cv2.putText(fingers, str(count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
    return fingers, count
=======
                cv2.fillPoly(fingers_img, pts=[cnt],color=255)
                count += 1

    #cv2.putText(fingers, str(count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
    return fingers_img, count
>>>>>>> develop


def getCircle(imgTransformed):
    h = imgTransformed.shape[0]
    w = imgTransformed.shape[1]

    thresh, centerImg = cv2.threshold(imgTransformed, 253, 255, cv2.THRESH_BINARY)

<<<<<<< HEAD
    center = 0, 0

    breakNestedLoop = False
=======
    center = findCenter(centerImg, h, w)

    crop, r = autoCropBinImg(imgTransformed)
    circle = cv2.circle(centerImg, center, r, 255, -1)

    return circle

def findCenter(centerImg, h, w):
    center = 0, 0
    breakNestedLoop = False

>>>>>>> develop
    for y in range(0, h):
        for x in range(0, w):
            if centerImg[y, x] != 0:
                center = x, y
                breakNestedLoop = True
                break
        if breakNestedLoop:
            break
<<<<<<< HEAD

    crop, r = autoCropBinImg(imgTransformed)
    #print("radius:", r)
    circle = cv2.circle(centerImg, center, r, 255, -1)

    return circle


#fings_img is binary img of only fingers after find_fingers()
def get_fingers_data(fings_img):
    props = regionprops(fings_img, [
    'BoundingBox',
    'Centroid',
    'Orientation',
    'MajorAxisLength',
    'MinorAxisLength'])
=======
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




# Dunno if needed
def measure_object():
    image = np.full((600, 600), 0, dtype=np.uint8)

    rr, cc = ellipse(300, 350, 100, 220)
    image[rr,cc] = 1

    image = geometric_transform(image, rotate)

    label_img = label(image)

    props = regionprops(np.squeeze(label_img), [
        'BoundingBox',
        'Centroid',
        'Orientation',
        'MajorAxisLength',
        'MinorAxisLength'
    ])

    plt.imshow(image)

    for prop in props:
        x0 = prop['Centroid'][1]
        y0 = prop['Centroid'][0]
        x1 = x0 + math.cos(prop['Orientation']) * 0.5 * prop['MajorAxisLength']
        y1 = y0 - math.sin(prop['Orientation']) * 0.5 * prop['MajorAxisLength']
        x2 = x0 - math.sin(prop['Orientation']) * 0.5 * prop['MinorAxisLength']
        y2 = y0 - math.cos(prop['Orientation']) * 0.5 * prop['MinorAxisLength']

        plt.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        plt.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        plt.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = prop['BoundingBox']
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        plt.plot(bx, by, '-b', linewidth=2.5)

    plt.gray()
    plt.axis((0, 600, 600, 0))
    plt.show()


def rotate(xy, angle=0.2):
    x, y = xy
    out_x = math.cos(angle) * x - math.sin(angle) * y
    out_y = math.sin(angle) * x + math.cos(angle) * y
    return (out_x, out_y)
>>>>>>> develop
