import segmentation.segmentation as sgm
import frame.frame as fr
import analysis.fingers as fings
import analysis.points as pts
import analysis.general as general
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
        helpers.drawContours(imgEdge, imgContours, imgCanvas)

        # avarage color
        imgHsv, readyBinary, readyImg = sgm.hsv_differentiation(img, False, SET_VALUES_MANUALLY)
        #stackHisto = stackImages(2, [[imgHsv, readyBinary, readyImg], list(sgm.hsv_differentiation(img, True, False))])

        # sgm.find_max_color(img)

        img_transformed = general.distanceTransform(readyBinary)

        contourImg = pts.find_lower_points(readyImg)
        frame.append(contourImg)
        # compare_average_and_dominant_colors(contourImg)

        #Find the center of the hand from the distance transformation
        thresh, centerImg = cv2.threshold(img_transformed, 253, 255, cv2.THRESH_BINARY)

        circle = fings.getCircle(img_transformed)

        fingers = cv2.subtract(readyBinary, circle, mask=None)
        fings.findFingers(fingers)

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




if __name__ == "__main__":
    main()
