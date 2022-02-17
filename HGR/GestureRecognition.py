import segmentation.segmentation as sgm
import segmentation.region_segmentation as rsgm
import stack.stack as stk
import analysis.fingers as fings
import analysis.points as pts
from analysis import mouse_handler
import analysis.general as general
import helpers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
from gui import gui_handler as gui
import commands.commands_handler as cmds
from cython_funcs import helpers_cy as cy
import time

#recist code


VID_NAME = "Videos\\handGesturesVid2.mp4"
SET_VALUES_MANUALLY = False

stage = [1]
ranges = []
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

        analyze_frame(img, cmds_handler, True)
        #cv2.waitKey(0)

        ##end_tot = time.time()
        ##print("time for everything:", end_tot-start_tot)
        ##print("percentage of time of stack compared to everything:", str(int((end-start)/(end_tot-start_tot))*100) + "%")

        #if 'q' is pressed, close all windows and break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



def analyze_frame(img, cmds_handler, is_histo):
    #img = img[160:490, 0:330]
    img = cv2.resize(img, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)


    if not is_histo:
        # Separate hand from background through hsv difference
        img_hsv, ready_binary, ready_img, range = sgm.hsv_differentiation(img, manually=SET_VALUES_MANUALLY)

        stack, data = analyze_segmentated_img(ready_img, ready_binary)
        execute_commands(data, cmds_handler)
    else:
        stack = segmentate(img)

    #app.frame.panel.put_img(stack)
    cv2.imshow("stack", stack.to_viewable_stack(2))



def segmentate(img):
    stack = None
    global ranges

    # In stage 1, just show that we are preparing stage 2
    if stage[0] == 1:
        print("stage 1")
        if clock_has_not_started[0]:
            print("start clock of stage 1")
            # Be in stage 1 for 3 seconds
            t = threading.Thread(target=helpers.timer(6, stage, clock_has_not_started))
            t.start()
            print("end clock of stage 1")
            clock_has_not_started[0] = False
        stack = stage1(img)
    # In stage 2, get the ranges through the histogram
    elif stage[0] == 2:
        print("stage 2")
        if clock_has_not_started[0]:
            print("start clock of stage 2")
            # Be in stage 1 for 3 seconds
            t = threading.Thread(helpers.timer(5, stage, clock_has_not_started))
            t.start()
            clock_has_not_started[0] = False
        stack, range = stage2(img)
        ranges.append(range)
    # In stage 3, use the calculated range from the ranges
    elif stage[0] == 3:
        print("stage 3")
        if clock_has_not_started[0]:
            print("compute stage 3")
            ranges = sgm.compute_best_range(ranges)
            clock_has_not_started[0] = False
        stack = stage3(img)


    return stack



def stage1(img):
    color = (255, 0, 0) # stage 1 is blue
    square_img, small = sgm.get_square(img, color)

    stack = stk.Stack([square_img, small])

    return stack


def stage2(img):
    color = (0, 255, 0) #stage 2 is green
    square_img, small = sgm.get_square(img, color)

    small_hsv, small_no_bg, small_binary, range = list(sgm.hsv_differentiation(small, is_histo=True))

    stack = stk.Stack([square_img, small, small_hsv, small_no_bg, small_binary])

    return stack, range


def stage3(img):
    # Separate hand from background through hsv difference
    img_hsv, binary, img_no_bg, r = sgm.hsv_differentiation(img, has_params=True, params=ranges)

    region_seg = list(rsgm.region_based_segmentation(img))
    #vis = cv2.hconcat(region_seg)
    normalized = (region_seg[0]*255).astype(np.uint8)

    stack = stk.Stack([img, img_hsv, binary, img_no_bg])
    stack.append(normalized)

    return stack




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
