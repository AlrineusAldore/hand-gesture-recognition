import cv2
import numpy as np

# consts
SCALE_DOWN = 0.6

def main():
    cap = cv2.VideoCapture(0)
    #cap.set(10, 100)
    InitializeWindows()

    while True:
        success, img = cap.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
        (thresh, imgBinary) = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        imgContour1 = img.copy()
        imgContour2 = img.copy()

        # Makes a new black image with same resolution
        imgCanvas1 = img.copy()
        imgCanvas1[:] = 0,0,0
        imgCanvas2 = imgCanvas1.copy()

        normalEdge = cv2.Canny(img, 100, 100)
        grayEdge = cv2.Canny(imgGray, 100, 100)
        blurEdge = cv2.Canny(imgBlur, 50, 50)
        blurEdge2 = cv2.Canny(imgBlur, 30, 30)

        drawContours(blurEdge, imgContour1, imgCanvas1)
        drawContours(blurEdge2, imgContour2, imgCanvas2)

        # Treshold (binary)
        ret, grayTresh = cv2.threshold(imgGray, 130, 255, cv2.THRESH_BINARY)



        # Color
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # HSV
        h1Min = cv2.getTrackbarPos("Hue1 Min", "Trackbars") # 0
        h1Max = cv2.getTrackbarPos("Hue1 Max", "Trackbars") # 20
        h2Min = cv2.getTrackbarPos("Hue2 Min", "Trackbars") # 160
        h2Max = cv2.getTrackbarPos("Hue2 Max", "Trackbars") # 179
        sMin = cv2.getTrackbarPos("Sat Min", "Trackbars") # 10
        sMax = cv2.getTrackbarPos("Sat Max", "Trackbars") # 70-100
        vMin = cv2.getTrackbarPos("Val Min", "Trackbars") # 0-90
        vMax = cv2.getTrackbarPos("Val Max", "Trackbars") # 255
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
        ret, binaryColor = cv2.threshold(grayColor, 50, 255, cv2.THRESH_BINARY)
        colorEdge = cv2.Canny(binaryColor, 100, 200)

        colorContour = openingMaskRes.copy()
        colorCanvas = img.copy()
        colorCanvas[:] = 0,0,0
        drawContours(colorEdge, colorContour, colorCanvas)


        stack = stackImages(0.4, ([img, imgHSV, colorEdge, blurEdge, blurEdge2],
                                  [bothMasks, bothMasksRes, colorContour, imgContour1, imgContour2],
                                  [closingMask, closingMaskRes, colorCanvas, imgCanvas1, imgCanvas2],
                                  [openingMask, openingMaskRes, binaryColor, imgMask, imgMask2]))

        cv2.imshow("stack", stack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



def InitializeWindows():
    # Trackbar
    # hue: 0-30 && 160-179
    # sat: 10-70 || 10-100
    # val: 90-255 || 0-255
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 380)
    cv2.createTrackbar("Hue1 Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Hue1 Max", "Trackbars", 20, 179, empty)
    cv2.createTrackbar("Hue2 Min", "Trackbars", 160, 179, empty)
    cv2.createTrackbar("Hue2 Max", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", 18, 255, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", 100, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)


def findHand(img):
    scaledImg = cv2.resize(img, None, fx=SCALE_DOWN, fy=SCALE_DOWN, interpolation=cv2.INTER_LINEAR)
    return scaledImg


def empty(a):
    pass


def drawContours(img, imgContour, imgCanvas):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print("\nstart of img:\n")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # to discard small random lines
            #print('area: ', area)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 5)
            cv2.drawContours(imgCanvas, cnt, -1, (255, 0, 0), 5)

            peri = cv2.arcLength(cnt, True)  # perimeter
            approx = cv2.approxPolyDP(cnt, 0.2*peri, True)
            #print("approx: ", approx)
            cv2.drawContours(imgContour, approx, -1, (0, 255, 0), 9)
            cv2.drawContours(imgCanvas, approx, -1, (0, 255, 0), 9)







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
