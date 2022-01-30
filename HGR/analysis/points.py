import cv2
import numpy as np


def find_lower_points(img):
    # Make a vid 256-500
    new_img = img.copy()

    # Black and white image
    hsvim = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skin_region_hsv = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skin_region_hsv, (2, 2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    # Contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours == ():
        return new_img
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(new_img, [contours], -1, (255, 255, 0), 1)

    # Hull (the yellow line)
    hull = cv2.convexHull(contours)
    cv2.drawContours(new_img, [hull], -1, (0, 255, 255), 1)

    # Calculate the angle
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull)
    if defects is None:
        return
    cnt = 0
    for i in range(defects.shape[0]):  # Calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # Cosine theorem
        if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
            cnt += 1
            cv2.circle(new_img, far, 2, [0, 0, 255], -1)
    if cnt > 0:
        cnt = cnt + 1
    cv2.putText(new_img, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    return new_img, cnt
