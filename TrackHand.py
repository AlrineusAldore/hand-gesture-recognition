import cv2
import numpy as np

# consts
SCALE_DOWN = 0.6

def main():
    cap = cv2.VideoCapture(0)
    #cap.set(10, 100)

    while True:
        success, img = cap.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
        (thresh, imgBinary) = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        imgContour1 = img.copy()
        imgContour2 = img.copy()

        imgCanvas1 = img.copy()
        imgCanvas1[:] = 0,0,0
        imgCanvas2 = imgCanvas1.copy()


        normalBorder = cv2.Canny(img, 100, 100)
        grayBorder = cv2.Canny(imgGray, 50, 50)
        blurEdge = cv2.Canny(imgBlur, 50, 50)
        blurEdge2 = cv2.Canny(imgBlur, 30, 30)

        drawContours(blurEdge, imgContour1, imgCanvas1)
        drawContours(blurEdge2, imgContour2, imgCanvas2)

        ret, grayTresh = cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY)
        ret, grayTresh2 = cv2.threshold(imgGray, 130, 255, cv2.THRESH_BINARY)
        ret, blurTresh = cv2.threshold(blurEdge, 120, 255, cv2.THRESH_BINARY)
        ret, blurTresh2 = cv2.threshold(blurEdge, 250, 255, cv2.THRESH_BINARY)

        stack = stackImages(0.5, ([img, imgGray, blurEdge, blurEdge2],
                                  [normalBorder, grayBorder, imgContour1, imgContour2],
                                  [grayTresh2, imgBinary, imgCanvas1, imgCanvas2]))

        cv2.imshow("stack", stack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def findHand(img):
    scaledImg = cv2.resize(img, None, fx=SCALE_DOWN, fy=SCALE_DOWN, interpolation=cv2.INTER_LINEAR)
    return scaledImg


def drawContours(img, imgContour, imgCanvas):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("\nstart of img:\n")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # to discard small random lines
            print('area: ', area)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 5)
            cv2.drawContours(imgCanvas, cnt, -1, (255, 0, 0), 5)

            peri = cv2.arcLength(cnt, True)  # perimeter
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
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
