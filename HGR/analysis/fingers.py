import cv2
import numpy as np
from helpers import autoCropBinImg
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

    #cv2.putText(fingers, str(count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
    return fingers, count


def getCircle(imgTransformed):
    h = imgTransformed.shape[0]
    w = imgTransformed.shape[1]

    thresh, centerImg = cv2.threshold(imgTransformed, 253, 255, cv2.THRESH_BINARY)

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
