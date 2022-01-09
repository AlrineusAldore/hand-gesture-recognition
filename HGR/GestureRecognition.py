import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns
import cython
#Recist Code



VID_NAME = "handGestures\\handGesturesVid.mp4"
SET_VALUES_MANUALLY = True

def main():
    cap = cv2.VideoCapture(0)
    n = 0

    if SET_VALUES_MANUALLY:
        InitializeWindows()

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

        #avarage color
        imgHsv, readyBinary, readyImg, readyContour = hsvDifferentiation(img)

        #find_max_color(img)

        imgTransformed = distanceTransform(readyBinary)

        contourImg = feature_2_func(readyImg)
        #compare_average_and_dominant_colors(contourImg)

        #Find the center of the hand from the distance transformation
        thresh, centerImg = cv2.threshold(imgTransformed, 253, 255, cv2.THRESH_BINARY)


        circle = readyBinary.copy()#getCircle(imgTransformed)

        fingers = cv2.subtract(readyBinary, circle, mask=None)
        findFingers(fingers)

        stack = stackImages(0.6, [[img, contourImg, imgGray, imgBinary],
                                  [imgTransformed, centerImg, circle ,fingers],
                                  [imgHsv, readyBinary, readyImg, readyContour]])



        cv2.imshow("stack", stack)
        #cv2.imshow("hi2", autoCropBinImg(imgTransformed))
        #cv2.waitKey(0)


        #plt.imshow(stack)
        #plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def histogram(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    #trying to smooth out the histograms
    #sns.histplot(hist_b, color="blue", label="100% Equities", kde=True, stat="density", linewidth=256)
    #sns.histplot(hist_g, color="green", label="100% Equities", kde=True, stat="density", linewidth=256)
    #sns.histplot(hist_r, color="red", label="100% Equities", kde=True, stat="density", linewidth=256)
    plt.plot(hist_r, color='r', label="r")
    plt.plot(hist_g, color='g', label="g")
    plt.plot(hist_b, color='b', label="b")
    plt.legend()
    plt.show()

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")
    plt.legend()
    plt.title(f"{hist_h.argmax()}, {hist_h.argmin()}, {hist_s.argmax()}, {hist_s.argmin()}, {hist_v.argmax()}, {hist_v.argmin()}")
    plt.show()

    return hist_h.argmax(), hist_h.argmin(), hist_s.argmax(), hist_s.argmin(), hist_v.argmax(), hist_v.argmin()


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

    center = 0,0

    breakNestedLoop = False
    for y in range(0, h):
        for x in range(0, w):
            if centerImg[y,x] != 0:
                center = x,y
                breakNestedLoop = True
                break
        if breakNestedLoop:
            break

    crop, r = autoCropBinImg(imgTransformed)
    print("radius:", r)
    circle = cv2.circle(centerImg, center, r, 255, -1)

    return circle


def autoCropBinImg(bin):
    white_pt_coords=np.argwhere(bin)

    crop = bin  #in case bin is completely black

    if cv2.countNonZero(bin) != 0:
        min_y = min(white_pt_coords[:,0])
        min_x = min(white_pt_coords[:,1])
        max_y = max(white_pt_coords[:,0])
        max_x = max(white_pt_coords[:,1])

        r = int(min([max_y - min_y, max_x - min_x]) // 2)

        crop = bin[min_y:max_y,min_x:max_x]

    return crop, r



def find_max_color(img):
    # Number of bins
    LENGTH = 16
    WIDTH = 16
    HEIGHT = 16
    bins = [LENGTH, WIDTH, HEIGHT];

    # Range of bins
    ranges = [0, 256, 0, 256, 0, 256];
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

    stack3 = stackImages(0.2, [maxColor, img])

    cv2.imshow("stack 3", stack3)

def hsvDifferentiation(img):
    # Color
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # histogram
    h1Max, h1Min, sMax, sMin, vMax, vMin = histogram(img)

    # HSV
    h1Min = 0
    sMin = 80
    vMin = 0
    h1Max = 179
    sMax = 255
    vMax = 255
    h2Min = h1Min
    h2Max = h1Max

    # HSV
    if SET_VALUES_MANUALLY:
        h1Max = cv2.getTrackbarPos("Hue1 Max", "Trackbars")  # 20
        h2Min = cv2.getTrackbarPos("Hue2 Min", "Trackbars")  # 160
        h1Min = cv2.getTrackbarPos("Hue1 Min", "Trackbars")  # 0
        h2Max = cv2.getTrackbarPos("Hue2 Max", "Trackbars")  # 179
        sMin = cv2.getTrackbarPos("Sat Min", "Trackbars")  # 10
        sMax = cv2.getTrackbarPos("Sat Max", "Trackbars")  # 70-100
        vMin = cv2.getTrackbarPos("Val Min", "Trackbars")  # 0-90
        vMax = cv2.getTrackbarPos("Val Max", "Trackbars")  # 255
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

    #use the masks to get avarage color
    average = cv2.mean(bothMasksRes, bothMasks)
    # Makes a new image for average color
    bothMasksAverage = img.copy()
    bothMasksAverage[:] = (average[0], average[1], average[2])

    #use the masks to get avarage color
    average = cv2.mean(closingMaskRes, closingMask)
    closingMaskAverage = img.copy()
    closingMaskAverage[:] = (average[0], average[1], average[2])

    #use the masks to get avarage color
    average = cv2.mean(openingMaskRes, openingMask)
    openingMaskAverage = img.copy()
    openingMaskAverage[:] = (average[0], average[1], average[2])

    stack2 = stackImages(0.9, [[bothMasks, bothMasksRes, bothMasksAverage],
                               [closingMask, closingMaskRes, closingMaskAverage],
                               [openingMask, openingMaskRes, openingMaskAverage]])

    cv2.imshow("stack 2", stack2)

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
    cv2.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)


def drawContours(img, imgContour, imgCanvas):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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

    cv2.imshow("resized imaged", resized)

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

    normalized = cv2.normalize(src=dist, dst=dist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    transformed = (normalized*255).astype(np.uint8)
    #for arr in normalized:
    #    transformed.append((arr*255).astype(int))

    #print("dist: \n",dist, "\n\n\n\n\n")
    #print("transformed: \n",transformed, "\n\n\n\n\n")
    #print("normalized: \n",normalized, "\n\n\n\n\n")

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
