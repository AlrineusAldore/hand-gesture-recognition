import segmentation.segmentation as sgm
import frame.frame as fr
import analysis.fingers as fings
import analysis.points as pts
import analysis.general as general
import helpers
import cv2
import numpy as np
import matplotlib.pyplot as plt

#recist code

# import cython

VID_NAME = "Videos\\handGesturesVid.mp4"
SET_VALUES_MANUALLY = False


def main():
    plt.show()
    plt.ion()

    if SET_VALUES_MANUALLY:
        helpers.InitializeWindows()


    analyze_capture(VID_NAME, False)  # Analyzing a video
    #analyze_capture(0, False)  # Analyzing camera



# Fully analyzes a whole capture
def analyze_capture(cap_path, pre_recorded):
    cap = cv2.VideoCapture(cap_path)
    n = 0

    #loop forever
    while cap.isOpened():

        success, img = cap.read()

        # Reset video if it ends
        if not success:
            cap = cv2.VideoCapture(cap_path)
            success, img = cap.read()

        # skips 10 frames if not live
        if pre_recorded:
            n += 1
            if n % 10 != 0:
                cap.grab()
                continue
            n = 0

        frame = fr.Frame()

        img = img[160:490, 0:330]
        img = cv2.resize(img, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
        frame.append(img)

        blank_img = img.copy()
        blank_img[:] = 0, 0, 0

        # Separate hand from background through hsv difference
        img_hsv, ready_binary, ready_img = sgm.hsv_differentiation(img, False, SET_VALUES_MANUALLY)


        img_transformed = general.distanceTransform(ready_binary)

        lower_points_img = pts.find_lower_points(ready_img)
        frame.append(lower_points_img)

        # Find the center of the hand from the distance transformation
        thresh, center_img = cv2.threshold(img_transformed, 253, 255, cv2.THRESH_BINARY)

        circle = fings.getCircle(img_transformed)

        fingers = cv2.subtract(ready_binary, circle, mask=None)
        fingers = fings.findFingers(fingers)


        frame.lst += [img_transformed, center_img, circle, fingers]
        frame.lst += [img_hsv, ready_binary, ready_img]
        frame.auto_organize()


        stack = frame.stack(1.5)

        cv2.imshow("stack", stack)

        #histo = fr.Frame([img_hsv, ready_binary, ready_img] + list(sgm.hsv_differentiation(img, True, False)))
        #cv2.imshow("histo", histo.stack(2))
        #cv2.waitKey(0)

        #if 'q' is pressed, close all windows and break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break





if __name__ == "__main__":
    main()
