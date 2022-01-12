import cv2
from helpers import autoCropBinImg

def findFingers(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # to discard small random lines
            cv2.drawContours(img, cnt, -1, 100, 2)
            count += 1

    cv2.putText(img, str(count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)


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
