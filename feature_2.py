#!/usr/bin/env python3.7
# Import packages
import cv2
import numpy as np


# consts
SCALE_DOWN = 0.6

IMAGE_NAME_1 = 'hand gesture ideal pictures\\palm.jpeg'
IMAGE_NAME_2 = 'hand gesture ideal pictures\\thumbs up.jpg'

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def feature_2_func(img):
    #make a vid 256-500
    #
    img = cv2.resize(img, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_AREA)
    ogImg = img.copy()

    #balck and white image
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    #contors
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(img, [contours], -1, (255, 255, 0), 2)

    #hull (the yellow line)
    hull = cv2.convexHull(contours)
    cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)

    #claculate the angle
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull)
    if defects is not None:
        cnt = 0
    for i in range(defects.shape[0]):  # calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
        if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
            cnt += 1
            cv2.circle(img, far, 4, [0, 0, 255], -1)
    if cnt > 0:
        cnt = cnt + 1
    cv2.putText(img, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    images = stackImages(1.2, [[ogImg, cv2.cvtColor(ogImg, cv2.COLOR_BGR2GRAY), img]])
    cv2.imshow('final_result', images)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    img = cv2.imread(IMAGE_NAME_2)
    #cv2.imshow("original", img)
    feature_2_func(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(3)

if __name__ == "__main__":
    main()



