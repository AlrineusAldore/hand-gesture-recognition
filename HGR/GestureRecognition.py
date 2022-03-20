from constants import EMPTY_HISTO
from analysis import mouse_handler
from gui import gui_handler as gui
from cython_funcs import helpers_cy as cy
import segmentation.segmentation as sgm
import segmentation.region_segmentation as rsgm
import stack.stack as stk
import analysis.fingers as fings
import analysis.points as pts
import analysis.general as general
import commands.commands_handler as cmds
import helpers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

#recist code


VID_NAME = "Videos\\handGesturesVid2.mp4"
SET_VALUES_MANUALLY = False

stage = [1]
ranges = {"hsv":[], "lab":[], "rgb":[]}
clock_has_not_started = [True]


def main():
    plt.show()
    plt.ion()

    if SET_VALUES_MANUALLY:
        helpers.InitializeWindows()

    #app = gui.make_gui()

    #fings.measure_object()

    #print("cython output:", cy.test(5))
    #analyze_capture(VID_NAME, 0, app)  # Analyzing a video with gui
    #analyze_capture(VID_NAME, 0, 0)  # Analyzing a video
    analyze_capture(0, 0, 0)  # Analyzing camera



# Fully analyzes a whole capture
def analyze_capture(cap_path, frames_to_skip, app):
    cap = cv2.VideoCapture(cap_path)
    n = 0
    cmds_handler = cmds.CommandsHandler()

    #loop forever
    while cap.isOpened():
        ##start_tot = time.time()

        success, img = cap.read()

        # Reset video if it ends
        if not success:
            cap = cv2.VideoCapture(cap_path)
            success, img = cap.read()

        # skips N frames
        if frames_to_skip > 1:
            n += 1
            if n % frames_to_skip != 0:
                cap.grab()
                continue
            n = 0

        img = cv2.flip(img, 1) #unmirror the image

        analyze_frame(img, cmds_handler)
        #cv2.waitKey(0)

        ##end_tot = time.time()
        ##print("time for everything:", end_tot-start_tot)
        ##print("percentage of time of stack compared to everything:", str(int((end-start)/(end_tot-start_tot))*100) + "%")

        #if 'q' is pressed, close all windows and break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



def analyze_frame(img, cmds_handler, is_edge_seg=False, is_manual=False):
    #img = img[160:490, 0:330]
    #cv2.imshow("og", img)
    img = cv2.resize(img, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)
    scale = 1.9


    if is_edge_seg:
        stack = stk.Stack(edge_segmentation(img))
        scale = 3
    elif is_manual:
        # Separate hand from background through hsv difference but manually
        img_hsv, ready_binary, ready_img, range = sgm.hsv_differentiation(img, manually=SET_VALUES_MANUALLY)

        stack, data = analyze_segmentated_img(ready_img, ready_binary)
        execute_commands(data, cmds_handler)
    else:
        stack = segmentate(img)


    #app.frame.panel.put_img(stack)
    cv2.imshow("stack", stack.to_viewable_stack(scale))
    pass  # Breakpoint for each frame



def segmentate(img):
    stack = None
    global ranges
    main_area_img = edge_segmentation(img)
    b, no_low_sat = rsgm.threshold_white(main_area_img)

    # In stage 1, just show that we are preparing stage 2
    if stage[0] == 1:
        if clock_has_not_started[0]:
            # Be in stage 1 for 3 seconds
            #t = threading.Thread(target=helpers.timer, args=(6, stage, clock_has_not_started))
            #t.start()
            if cv2.waitKey(1) & 0xFF == ord('g'):
                print("g pressed")
                stage[0] += 1
                clock_has_not_started[0] = True
            #clock_has_not_started[0] = False
        stack = stage1(no_low_sat)
    # In stage 2, get the ranges through the histogram
    elif stage[0] == 2:
        if clock_has_not_started[0] and stage[0] == 2:
            # Be in stage 1 for 3 seconds
            t = threading.Thread(target=helpers.timer, args=(3, stage, clock_has_not_started))
            t.start()
            clock_has_not_started[0] = False
        stack, color_spaces_ranges = stage2(no_low_sat)
        if color_spaces_ranges is not None:
            ranges["hsv"].append(color_spaces_ranges)
            #ranges["lab"].append(color_spaces_ranges[1])
            #ranges["rgb"].append(color_spaces_ranges[2])
    # In stage 3, use the calculated range from the ranges
    elif stage[0] == 3:
        if clock_has_not_started[0]:
            ranges["hsv"] = sgm.compute_best_range(ranges["hsv"])
            #ranges["lab"] = sgm.compute_best_range(ranges["lab"])
            #ranges["rgb"] = sgm.compute_best_range(ranges["rgb"])
            clock_has_not_started[0] = False
        stack = stage3(main_area_img)
        stack2 = stage3(no_low_sat)
        stack.append(stack2.lst[3])
        stack.append(stack2.lst[0])
        stack.append(stack2.lst[5])


    return stack


def stage1(img):
    color = (255, 0, 0) # stage 1 is blue
    square_img, small = sgm.get_square(img, color)

    try:
        hsv_small, hsv_small_no_bg, hsv_small_bin = list(sgm.hsv_differentiation(small, seg_type=0, is_plot=False))
        #lab_small, lab_small_no_bg, lab_small_bin = list(sgm.hsv_differentiation(small, seg_type=1))
        #rgb_small, rgb_small_no_bg, rgb_small_bin = list(sgm.hsv_differentiation(small,  seg_type=2))

        stack = stk.Stack([square_img, small, hsv_small, hsv_small_no_bg, hsv_small_bin], size=(1,5))
    except Exception as e:
        stack = stk.Stack([square_img, small], size=(1, 5), is_filler_empty=True)
        if e.args[0] != EMPTY_HISTO:
            print(helpers.get_line_num(), ". e:", repr(e))



    return stack



def stage2(img):
    color = (0, 255, 0) #stage 2 is green
    square_img, small = sgm.get_square(img, color)

    try:
        hsv_small, hsv_small_no_bg, hsv_small_bin, hsv_range = list(sgm.hsv_differentiation(small, get_range=True, seg_type=0))
        #lab_small, lab_small_no_bg, lab_small_bin, lab_range = list(sgm.hsv_differentiation(small, get_range=True, seg_type=1))
        #rgb_small, rgb_small_no_bg, rgb_small_bin, rgb_range = list(sgm.hsv_differentiation(small, get_range=True, seg_type=2))

        stack = stk.Stack([square_img, small, hsv_small, hsv_small_no_bg, hsv_small_bin], size=(1,5))
    except Exception as e:
        stack = stk.Stack([square_img, small], size=(1, 5), is_filler_empty=True)
        if e.args[0] != EMPTY_HISTO:
            print(helpers.get_line_num(), ". e:", repr(e))

        return stack, None

    return stack, (hsv_range)


def stage3(img):
    # Separate hand from background through hsv difference
    hsv_img, hsv_avg_bin, hsv_avg_no_bg = sgm.hsv_differentiation(img, has_params=True, params=ranges["hsv"][0], seg_type=0)
    hsv_img, hsv_edge_bin, hsv_edge_no_bg = sgm.hsv_differentiation(img, has_params=True, params=ranges["hsv"][1], seg_type=0)
    #lab_img, lab_avg_bin, lab_avg_no_bg = sgm.hsv_differentiation(img, has_params=True, params=ranges["lab"][0], seg_type=1)
    #lab_img, lab_edge_bin, lab_edge_no_bg = sgm.hsv_differentiation(img, has_params=True, params=ranges["lab"][1], seg_type=1)
    #rgb_img, rgb_avg_bin, rgb_avg_no_bg = sgm.hsv_differentiation(img, has_params=True, params=ranges["rgb"][0], seg_type=2)
    #rgb_img, rgb_edge_bin, rgb_edge_no_bg = sgm.hsv_differentiation(img, has_params=True, params=ranges["rgb"][1], seg_type=2)

    stack = stk.Stack([img, hsv_img, hsv_avg_bin,
                       hsv_avg_no_bg, hsv_edge_bin, hsv_edge_no_bg], size=(2,3))

    return stack


def edge_segmentation(img):
    scale = 31
    img_blur = cv2.GaussianBlur(img,(scale,scale),1)

    blur_vars = rsgm.region_based_segmentation(img_blur)
    normalized = helpers.normalize_zero1_to_zero255(blur_vars[1])

    main_area_img = cv2.bitwise_and(img, img, mask=normalized)

    return main_area_img







def stage0(img, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = img.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(img, bg, aWeight)

def segment_bg(img, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), img)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def analyze_segmentated_img(img, binary):
    data = {}  # To store all the data and use it to execute the right command
    stack = stk.Stack()

    blank_img = img.copy()
    blank_img[:] = 0, 0, 0

    img_transformed = general.distanceTransform(binary)

    lower_points_img, fings_count_pts = pts.find_lower_points(img)
    stack.append(lower_points_img)

    # Find the center of the hand from the distance transformation
    thresh, center_img = cv2.threshold(img_transformed, 253, 255, cv2.THRESH_BINARY)

    circle = fings.getCircle(img_transformed)

    fingers = cv2.subtract(binary, circle, mask=None)
    fingers, fings_count = fings.find_fingers(fingers)
    cv2.putText(fingers, str(fings_count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)

    img_hsv_pts = mouse_handler.show_extreme_points(img.copy(), binary)
    img_pts = mouse_handler.show_north_extreme_points(img.copy(), binary, fings_count)
    stack.append(img)



    stack.lst += [img_transformed, center_img, circle, fingers]
    stack.lst += [binary, img]
    stack.lst += [img_hsv_pts, img_pts]
    stack.auto_organize()


    #  Add number of fingers up to data
    if (fings_count == fings_count_pts):
        data["fings_count"] = fings_count
    else:
        data["fings_count"] = None


    return stack, data





def execute_commands(data, cmds_handler):
     #  Do command
    cmds_handler.update_data(data)
    result = cmds_handler.check_commands()

    #  Makes an image showing only the used command
    #command_img = blank_img.copy()
    #cv2.putText(command_img, result, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    #stack.append(command_img)

    return result







if __name__ == "__main__":
    main()
