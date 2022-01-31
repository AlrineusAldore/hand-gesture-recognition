import analysis.points as pts
import cv2
import mouse


# Draws extreme points on image and returns it
def show_extreme_points(img, binary):
    ext_left, ext_right, ext_top, ext_bot = pts.extreme_points(binary)
    cv2.circle(img, ext_left, 3, (0, 0, 255), -1)
    cv2.circle(img, ext_right, 3, (0, 255, 0), -1)
    cv2.circle(img, ext_top, 3, (255, 0, 0), -1)
    cv2.circle(img, ext_bot, 3, (255, 255, 0), -1)
    return img


# If one finger is up, show north point and move mouse to it
def show_north_extreme_points(img, binary, fingers_count):
    if (fingers_count == 1):
        ext_left, ext_right, ext_top, ext_bot = pts.extreme_points(binary)
        cv2.circle(img, ext_top, 3, (255, 0, 0), -1)
        set_mouse_point(ext_top)
    return img


# Drags the mouse to a given point while holding the button
def set_mouse_point(mouse_point):
    curr = mouse.get_position()
    # Needs to change to mouse.move() with combination of mouse.press() and mouse.release()
    mouse.drag(curr[0], curr[1], mouse_point[0], mouse_point[1], absolute=False, duration=0.1)

