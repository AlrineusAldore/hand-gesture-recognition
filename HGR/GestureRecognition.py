import segmentation.segmentation as sgm
import frame.frame as fr
import helpers
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
        helpers.InitializeWindows()


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

        frame = fr.Frame()

        img = img[160:490, 0:330]
        img = cv2.resize(img, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
        frame.append(img)

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
        imgHsv, readyBinary, readyImg = sgm.hsv_differentiation(img, False, SET_VALUES_MANUALLY)
        #stackHisto = stackImages(2, [[imgHsv, readyBinary, readyImg], list(sgm.hsv_differentiation(img, True, False))])

        # sgm.find_max_color(img)

        img_transformed = distanceTransform(readyBinary)

        contourImg = feature_2_func(readyImg)
        frame.append(contourImg)
        # compare_average_and_dominant_colors(contourImg)

        #Find the center of the hand from the distance transformation
        thresh, centerImg = cv2.threshold(img_transformed, 253, 255, cv2.THRESH_BINARY)

        circle = getCircle(img_transformed)

        fingers = cv2.subtract(readyBinary, circle, mask=None)
        findFingers(fingers)

        frame.lst += [imgGray, imgBinary]
        frame.lst += [img_transformed, centerImg, circle ,fingers]
        frame.lst += [imgHsv, readyBinary, readyImg]
        frame.auto_organize()

        #stack = stackImages(1.5, [[img, contourImg, imgGray, imgBinary],
        #                          [img_transformed, centerImg, circle ,fingers],
        #                          [imgHsv, readyBinary, readyImg, img]])
        stack = frame.stack(1.5)

        cv2.imshow("stack", stack)

        #cv2.imshow("stack", stackHisto)
        # cv2.imshow("hi2", autoCropBinImg(imgTransformed))
        # cv2.waitKey(0)

        # plt.imshow(stack)
        # plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


#fingers
def findFingers(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # to discard small random lines
            cv2.drawContours(img, cnt, -1, 100, 2)
            count += 1

    cv2.putText(img, str(count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)


#fingers
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


#helpers
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


#helpers/none
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


#helpers
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


#points
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
    cv2.drawContours(newImg, [contours], -1, (255, 255, 0), 1)

    # hull (the yellow line)
    hull = cv2.convexHull(contours)
    cv2.drawContours(newImg, [hull], -1, (0, 255, 255), 1)

    # claculate the angle
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull)
    if defects is None:
        return
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
            cv2.circle(newImg, far, 2, [0, 0, 255], -1)
    if cnt > 0:
        cnt = cnt + 1
    cv2.putText(newImg, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

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
