import cv2
import numpy as np
def draw_squer(img):
    start_point = (240, 350)
    end_point = (110,220)
    color = (255, 0, 0)
    thickness = 2
    image = cv2.rectangle(img, start_point, end_point, color, thickness)
    return image

def get_dominant_color(img):
    # Cropping an image to get part of hand and not the background
    cropped_image = img[240:350, 110:220]
    cv2.imshow("cropped", cropped_image)
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    imgDom = img.copy()
    imgDom[:] = dominant
    cv2.imshow("dominant", imgDom)

    return dominant

