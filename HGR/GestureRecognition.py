import segmentation.segmentation as sgm
import cv2
import numpy as np
import matplotlib.pyplot as plt

# import cython

VID_NAME = "Videos\\handGesturesVid.mp4"
SET_VALUES_MANUALLY = False


def main():
    cap = cv2.VideoCapture(VID_NAME)
    n = 0
    plt.show()
    plt.ion()

    if SET_VALUES_MANUALLY:
        InitializeWindows()


    while cap.isOpened():

        success, img = cap.read()

        # Reset video if it ends
        if not success:
            cap = cv2.VideoCapture(VID_NAME)
            success, img = cap.read()

        # skips 10 frames
        n += 1
        if n % 10 != 0:
            continue

        img = img[160:490, 0:330]
        img = cv2.resize(img, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)

        # Different imgs types
        blankImg = img.copy()
        blankImg[:] = 0, 0, 0
        imgBlur = cv2.GaussianBlur(img, (9, 9), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        (thresh, imgBinary) = cv2.threshold(imgGray, 255, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        imgEdge = cv2.Canny(imgBlur, 30, 30)
        imgCanvas = blankImg.copy()
        imgContours = img.copy()
        drawContours(imgEdge, imgContours, imgCanvas)

        # avarage color
        imgHsv, readyBinary, readyImg = sgm.hsvDifferentiation(img, False, SET_VALUES_MANUALLY)
        stackHisto = stackImages(2, [[imgHsv, readyBinary, readyImg],
                                       list(sgm.hsvDifferentiation(img, True, SET_VALUES_MANUALLY))])
        # sgm.find_max_color(img)

        img_transformed = distanceTransform(readyBinary)

        contourImg = feature_2_func(readyImg)
        # compare_average_and_dominant_colors(contourImg)

        thresh, skeleton = cv2.threshold(img_transformed, 1, 255, cv2.THRESH_BINARY)

        stack = stackImages(1.5, [[img, contourImg, imgGray, imgBinary],
                                  [imgBlur, imgEdge, imgContours, imgCanvas],
                                  [imgHsv, readyBinary, readyImg, img]])

        #cv2.imshow("stack", stack)
        cv2.imshow("stack", stackHisto)
        # cv2.imshow("hi2", autoCropBinImg(imgTransformed))
        # cv2.waitKey(0)

        # plt.imshow(stack)
        # plt.show()

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break



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
    print("radius:", r)
    circle = cv2.circle(centerImg, center, r, 255, -1)

    return circle


def autoCropBinImg(bin):
    white_pt_coords = np.argwhere(bin)

    crop = bin  # in case bin is completely black

    if cv2.countNonZero(bin) != 0:
        min_y = min(white_pt_coords[:, 0])
        min_x = min(white_pt_coords[:, 1])
        max_y = max(white_pt_coords[:, 0])
        max_x = max(white_pt_coords[:, 1])

        r = int(min([max_y - min_y, max_x - min_x]) // 2)

        crop = bin[min_y:max_y, min_x:max_x]

    return crop, r


def slow(imgTransformed):
    h = imgTransformed.shape[0]
    w = imgTransformed.shape[1]

    skeleton = imgTransformed.copy()

    for y in range(0, h):
        for x in range(0, w):
            if imgTransformed[y, x] == 240:
                skeleton[y, x] = imgTransformed[y, x]
            else:
                skeleton[y, x] = 0

    return skeleton




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

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # to discard small random lines
            # print("cuntour: ", cnt)
            # print('area: ', area)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 2)
            cv2.drawContours(imgCanvas, cnt, -1, (255, 0, 0), 2)

            peri = cv2.arcLength(cnt, True)  # perimeter
            approx = cv2.approxPolyDP(cnt, 0.2 * peri, True)  # Points
            # print("approx: ", approx)
            cv2.drawContours(imgContour, approx, -1, (0, 255, 0), 3)
            cv2.drawContours(imgCanvas, approx, -1, (0, 255, 0), 3)


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

    # print("dominant: " , dominant)
    # print("average : ", average)


def distanceTransform(binary):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    transformed = cv2.normalize(src=dist, dst=dist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return transformed


def feature_2_func(img):
    # make a vid 256-500
    newImg = img.copy()

    # balck and white image
    hsvim = cv2.cvtColor(newImg, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    # contors
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours == ():
        print("tuple empty!")
        return newImg
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(newImg, [contours], -1, (255, 255, 0), 2)

    # hull (the yellow line)
    hull = cv2.convexHull(contours)
    cv2.drawContours(newImg, [hull], -1, (0, 255, 255), 2)

    # claculate the angle
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


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


if __name__ == "__main__":
    main()
