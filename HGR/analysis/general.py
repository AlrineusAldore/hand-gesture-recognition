import cv2
import numpy as np


# Gets binary img and returns a distance-transformed image
def distanceTransform(binary):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    normalized = cv2.normalize(src=dist, dst=dist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    transformed = (normalized*255).astype(np.uint8)

    return transformed
