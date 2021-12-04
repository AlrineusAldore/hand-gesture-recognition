import cv2
import numpy as np
from matplotlib import pyplot as plt

import cython



VID_NAME = "handGestures\\handGesturesVid.mp4"

def main():
    cap = cv2.VideoCapture(VID_NAME)
    n = 0

    #InitializeWindows()

    while cap.isOpened():


        success, img = cap.read()
        #Reset video if it ends
        if not success:
            cap = cv2.VideoCapture(VID_NAME)
            success, img = cap.read()


        #skips 10 frames
        n += 1
        if n % 10 != 0:
            continue
        #cv2.waitKey(0)


        img = img[160:490, 0:330]
        resizedImg = cv2.resize(img, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_AREA)



        #Different imgs types
        blankImg = img.copy()
        blankImg[:] = 0,0,0
        imgBlur = cv2.GaussianBlur(img, (9,9), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        (thresh, imgBinary) = cv2.threshold(imgGray, 255, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        imgEdge = cv2.Canny(imgBlur, 30, 30)
        imgCanvas = blankImg.copy()
        imgContours = img.copy()
        drawContours(imgEdge, imgContours, imgCanvas)


        imgHsv, readyBinary, readyImg, readyContour = hsvDiffrentiation(img)

        imgTransformed = distanceTransform(readyBinary)
        #compare_average_and_dominant_colors(resizedImg)

        contourImg = feature_2_func(readyImg)

        thresh, skeleton = cv2.threshold(imgTransformed, 1, 255, cv2.THRESH_BINARY)



        stack = stackImages(0.6, [[img, contourImg, imgGray, imgBinary],
                                  [imgBlur, imgEdge, imgContours ,imgCanvas],
                                  [imgHsv, readyBinary, readyImg, readyContour]])



        cv2.imshow("stack", stack)
        cv2.imshow("hi", imgTransformed)
        cv2.imshow("hi2", skeleton)
        #cv2.waitKey(100)


        #plt.imshow(stack)
        #plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break




def slow(imgTransformed):
    h = imgTransformed.shape[0]
    w = imgTransformed.shape[1]

    skeleton = imgTransformed.copy()


    for y in range(0, h):
        for x in range(0, w):
            if imgTransformed[y,x] == 240:
                skeleton[y,x] = imgTransformed[y,x]
            else:
                skeleton[y,x] = 0

    return skeleton


def hsvDifferentiation(img):
    # Color
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV
    h1Min = 0
    h1Max = 14
    h2Min = 179
    h2Max = 179
    sMin = 80
    sMax = 255
    vMin = 0
    vMax = 255
    # HSV
    #h1Max = cv2.getTrackbarPos("Hue1 Max", "Trackbars") # 20
    #h2Min = cv2.getTrackbarPos("Hue2 Min", "Trackbars") # 160
    #h1Min = cv2.getTrackbarPos("Hue1 Min", "Trackbars") # 0
    #h2Max = cv2.getTrackbarPos("Hue2 Max", "Trackbars") # 179
    #sMin = cv2.getTrackbarPos("Sat Min", "Trackbars") # 10
    #sMax = cv2.getTrackbarPos("Sat Max", "Trackbars") # 70-100
    #vMin = cv2.getTrackbarPos("Val Min", "Trackbars") # 0-90
    #vMax = cv2.getTrackbarPos("Val Max", "Trackbars") # 255
    # print("Hue:", hMin, hMax, "Sat:", sMin, sMax, "val:", vMin, vMax)
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
    #bothMasksRes2 = cv2.bitwise_or(imgMaskRes, imgMask2Res, mask=None)


    # guassian
    guassianMask = cv2.GaussianBlur(bothMasks, (5, 5), cv2.BORDER_DEFAULT)
    guassianMaskRes = cv2.bitwise_and(img, img, mask=guassianMask)

    # closing
    closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,7))
    openingKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    closingMask = cv2.morphologyEx(bothMasks, cv2.MORPH_CLOSE, closingKernel)
    closingMaskRes = cv2.bitwise_and(img, img, mask=closingMask)

    # opening
    openingMask = cv2.morphologyEx(closingMask, cv2.MORPH_OPEN, openingKernel)
    openingMaskRes = cv2.bitwise_and(img, img, mask=openingMask)


    # Edges after color
    grayColor = cv2.cvtColor(openingMaskRes, cv2.COLOR_BGR2GRAY)
    ret, binaryColor = cv2.threshold(grayColor, 20, 255, cv2.THRESH_BINARY)
    colorEdge = cv2.Canny(binaryColor, 100, 200)

    colorContour = openingMaskRes.copy()
    colorCanvas = img.copy()
    colorCanvas[:] = 0,0,0
    drawContours(colorEdge, colorContour, colorCanvas)

    stack2 = stackImages(0.6, [[bothMasks, bothMasksRes],
                               [closingMask, closingMaskRes],
                               [openingMask, openingMaskRes]])

    #cv2.imshow("stack 2", stack2)

    return (imgHSV, openingMask, openingMaskRes, colorContour)




def empty(a):
    pass


def InitializeWindows():
    # Trackbar
    # hue: 0-30 && 160-179
    # sat: 10-70 || 10-100
    # val: 90-255 || 0-255
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 380)
    cv2.createTrackbar("Hue1 Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Hue1 Max", "Trackbars", 14, 179, empty)
    cv2.createTrackbar("Hue2 Min", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("Hue2 Max", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", 80, 255, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)




def drawContours(img, imgContour, imgCanvas):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print("\n!!!!!!!!!!!!!!!!!!!!start of img!!!!!!!!!!!!:\n")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # to discard small random lines
            #print("cuntour: ", cnt)
            #print('area: ', area)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 5)
            cv2.drawContours(imgCanvas, cnt, -1, (255, 0, 0), 5)

            peri = cv2.arcLength(cnt, True)  # perimeter
            approx = cv2.approxPolyDP(cnt, 0.2*peri, True)  # Points
            #print("approx: ", approx)
            cv2.drawContours(imgContour, approx, -1, (0, 255, 0), 9)
            cv2.drawContours(imgCanvas, approx, -1, (0, 255, 0), 9)



def deleteBackground(img):
    hh, ww = img.shape[:2]

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([250, 250, 250])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("background black", result)



def compare_average_and_dominant_colors(img):

    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    average = img.mean(axis=0).mean(axis=0)
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    # Makes a new image for average color
    imgAverage = img.copy()
    imgAverage[:] = average
    cv2.imshow("average", cv2.resize(imgAverage, None, fx=2, fy=2, interpolation=cv2.INTER_AREA))

    # Makes a new image for dominant color
    imgDom = img.copy()
    imgDom[:] = dominant
    cv2.imshow("dominant", cv2.resize(imgDom, None, fx=2, fy=2, interpolation=cv2.INTER_AREA))

    #print("dominant: " , dominant)
    #print("average : ", average)




def distanceTransform(binary):

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    transformed = cv2.normalize(src=dist, dst=dist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return transformed


def feature_2_func(img):
    #make a vid 256-500
    #
    newImg = img.copy()

    #balck and white image
    hsvim = cv2.cvtColor(newImg, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    #contors
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(newImg, [contours], -1, (255, 255, 0), 2)

    #hull (the yellow line)
    hull = cv2.convexHull(contours)
    cv2.drawContours(newImg, [hull], -1, (0, 255, 255), 2)

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
            cv2.circle(newImg, far, 4, [0, 0, 255], -1)
    if cnt > 0:
        cnt = cnt + 1
    cv2.putText(newImg, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return newImg



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





if __name__ == "__main__":
    main()
