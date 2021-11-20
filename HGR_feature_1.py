#!/usr/bin/env python3.7
# Import packages
import cv2
import numpy as np
#import matplotlib.pyplot as plt

# consts
SCALE_DOWN = 0.6
IMAGE_NAME_1 = 'hand gesture ideal pictures\\palm.jpeg'
IMAGE_NAME_2 = 'hand gesture ideal pictures\\thumbs up.jpg'

# Makes the white background of the image black.
def delete_backgroung(img):
    hh, ww = img.shape[:2]

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([200, 200, 200])
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

def biner_and_contour(img):
    #balck and white image
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)

    transformed = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    norm_image = cv2.normalize(src=transformed, dst=transformed, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow("transformed", norm_image)

    #contors
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(img, [contours], -1, (255, 255, 0), 2)
    cv2.imshow("contours", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    cv2.imshow("average", imgAverage)

    # Makes a new image for dominant color
    imgDom = img.copy()
    imgDom[:] = dominant
    cv2.imshow("dominant", imgDom)

    print("dominant: " , dominant)
    print("average : ", average)

def resize_image_to_defolte_size(img, height, width):
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)
    return resized

def main():

    img = cv2.imread(IMAGE_NAME_1)
    img = resize_image_to_defolte_size(img, 450, 350)
    cv2.imshow("original", img)

    # Cropping an image to get part of hand and not the background
    cropped_image = img[240:350, 110:220]
    cv2.imshow("cropped", cropped_image)

    compare_average_and_dominant_colors(cropped_image)
    delete_backgroung(img)
    biner_and_contour(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
