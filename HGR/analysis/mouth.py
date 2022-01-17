import analysis.points as pts
import cv2
import mouse
def show_extreme_points(img, readyBinary):
    extLeft, extRight, extTop, extBot = pts.extreme_points(readyBinary)
    cv2.circle(img, extLeft, 3, (0, 0, 255), -1)
    cv2.circle(img, extRight, 3, (0, 255, 0), -1)
    cv2.circle(img, extTop, 3, (255, 0, 0), -1)
    cv2.circle(img, extBot, 3, (255, 255, 0), -1)
    return img

def show_north_extreme_points(img, readyBinary):
    extLeft, extRight, extTop, extBot = pts.extreme_points(readyBinary)
    cv2.circle(img, extTop, 3, (255, 0, 0), -1)
    #set_mouse_point(extTop)
    return img

def set_mouse_point(mouse_point):
    curr = mouse.get_position()
    mouse.drag(curr[0], curr[1], mouse_point[0], mouse_point[1], absolute=False, duration=0.1)

