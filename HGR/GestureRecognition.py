from constants import EMPTY_HISTO
from analysis import mouse_handler
from info_handlers.data import Data
from info_handlers.stack import Stack
from gui import gui_handler as gui
#from cython_funcs import helpers_cy as cy
from segmentation.helpers import open_close
import segmentation.color_segmentation as csgm
import segmentation.segmentation as sgm
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
import sql.mySQL as sqlit

#db = sqlit.database()

#recist code


VID_NAME = "Videos\\handGesturesVid.mp4"
SET_VALUES_MANUALLY = False

stage = [1]
ranges = {"hsv":[], "lab":[], "rgb":[]}
clock_has_not_started = [True]
bg = None



def main():
    plt.show()
    plt.ion()

    if SET_VALUES_MANUALLY:
        helpers.InitializeWindows()

    #app = gui.make_gui()

    #fings.measure_object()

    #print("cython output:", cy.test(5))
    #analyze_capture(VID_NAME, 0, app)  # Analyzing a video with gui
    #analyze_capture(VID_NAME, 0)  # Analyzing a video
    analyze_capture(0, 0)  # Analyzing camera



# Fully analyzes a whole capture

def analyze_capture(cap_path, frames_to_skip, app=None):
    cap = cv2.VideoCapture(cap_path)
    n = 0
    cmds_handler = cmds.CommandsHandler()
    has_started = False

    #loop forever
    while cap.isOpened():
        ##start_tot = time.time()

        success, img = cap.read()

        # Reset video if it ends
        if not success:
            cap = cv2.VideoCapture(cap_path)
            success, img = cap.read()

        img = cv2.flip(img, 1) #unmirror the image

        # Only start once 's' is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            has_started = True
            cv2.destroyWindow("start")
        if not has_started:
            cv2.imshow("start", img)
            continue

        n += 1
        # skips N frames
        if frames_to_skip > 1:
            if n % frames_to_skip != 0:
                cap.grab()
                continue
            #n = 0


        if n < 30:
            analyze_frame(img, cmds_handler, is_calc_avg_bg=True)
        else:
            analyze_frame(img, cmds_handler, is_calc_avg_bg=False)
        #cv2.waitKey(0)

        ##end_tot = time.time()
        ##print("time for everything:", end_tot-start_tot)
        ##print("percentage of time of stack compared to everything:", str(int((end-start)/(end_tot-start_tot))*100) + "%")

        #if 'q' is pressed, release cap, close all windows, and break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break



def analyze_frame(img, cmds_handler, is_calc_avg_bg=False, is_manual=False):
    #img = img[160:490, 0:330]
    #cv2.imshow("og", img)
    img = cv2.resize(img, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)
    scale = 1.4


    if is_calc_avg_bg:
        # Convert to gray and blur
        gray = helpers.get_gray_blurred_img(img)
        run_avg(gray, aWeight=0.5)

        # noinspection PyUnresolvedReferences
        stack = Stack([img, gray, bg.astype(np.uint8)], size=(1,3))
        scale = 3
    elif is_manual:
        # Separate hand from background through hsv difference but manually
        img_hsv, ready_binary, ready_img = csgm.hsv_differentiation(img, manually=SET_VALUES_MANUALLY)

        stack, data = analyze_segmented_img(ready_img, ready_binary)
        execute_commands(data, cmds_handler)
    else:
        stack = segment(img)

    if stack.hand_img is not None:
        im = stack.hand_img
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        try:
            stack, data = analyze_segmented_img(im, binary)
            execute_commands(data, cmds_handler)
        except Exception as e:
            stack = Stack([im])
            print(e)

    #app.frame.panel.put_img(stack)
    cv2.imshow("stack", stack.to_viewable_stack(scale))
    pass  # Breakpoint for each frame



def segment(img):
    stack = None
    global ranges
    stack = subtract_bg(img, threshold=30)
    #return stack
    stack2 = subtract_bg(img, threshold=20, is_sat=True)

    arm_img = stack.lst[5]
    arm_img2 = stack2.lst[5]

    arm_both_bins = cv2.bitwise_and(stack.lst[4], stack2.lst[4])
    arm_both_imgs = cv2.bitwise_and(img, img, mask=arm_both_bins)

    #combine 2 subtracted bg images that used different thresholds
    stack3 = Stack([stack.lst[1], stack2.lst[1], stack.lst[2], stack2.lst[2], stack.lst[4], stack2.lst[4], arm_both_bins, arm_both_imgs])
    #cv2.imshow("hihihi", stack.to_viewable_stack(2))


    main_area_img = arm_both_imgs

    stack = Stack([img, arm_both_imgs, edge_segmentation(arm_both_imgs), edge_segmentation(img)])


    b, no_low_sat = sgm.threshold_low_sat(main_area_img)
    no_white_bin, no_white = sgm.threshold_white_rgb(main_area_img, thresh=150)
    no_white_bin, no_white2 = sgm.threshold_white_rgb(main_area_img)

    stack.append(main_area_img)
    stack.append(no_white)
    stack.append(no_white2)

    #cv2.imshow("hihi", stack.to_viewable_stack(2))
    wanted_img = arm_both_imgs


    # In stage 1, just show that we are preparing stage 2
    if stage[0] == 1:
        if clock_has_not_started[0]:
            if cv2.waitKey(1) & 0xFF == ord('g'):
                print("g pressed")
                stage[0] += 1
                clock_has_not_started[0] = True
        stack = stage1(wanted_img)
    # In stage 2, get the ranges through the histogram
    elif stage[0] == 2:
        if clock_has_not_started[0] and stage[0] == 2:
            # Be in stage 1 for 3 seconds
            t = threading.Thread(target=helpers.timer, args=(3, stage, clock_has_not_started))
            t.start()
            clock_has_not_started[0] = False
        stack, color_spaces_ranges = stage2(wanted_img)
        if color_spaces_ranges is not None:
            ranges["hsv"].append(color_spaces_ranges)
    # In stage 3, use the calculated range from the ranges
    elif stage[0] == 3:
        if clock_has_not_started[0]:
            ranges["hsv"] = csgm.compute_best_range(ranges["hsv"])
            clock_has_not_started[0] = False
        stack = stage3(wanted_img, non_seg_img=img)


    return stack


def stage1(img):
    color = (255, 0, 0) # stage 1 is blue
    square_img, small = csgm.get_square(img, color)

    try:
        hsv_small, hsv_small_no_bg, hsv_small_bin = list(csgm.hsv_differentiation(small, seg_type=0, is_plot=False))
        #lab_small, lab_small_no_bg, lab_small_bin = list(sgm.hsv_differentiation(small, seg_type=1))
        #rgb_small, rgb_small_no_bg, rgb_small_bin = list(sgm.hsv_differentiation(small,  seg_type=2))

        stack = Stack([square_img, small, hsv_small, hsv_small_no_bg, hsv_small_bin])
    except Exception as e:
        stack = Stack([square_img, small], size=(2, 3), is_filler_empty=True)
        if e.args[0] != EMPTY_HISTO:
            print(helpers.get_line_num(), ". e:", repr(e))



    return stack


def stage2(img):
    color = (0, 255, 0) #stage 2 is green
    square_img, small = csgm.get_square(img, color)

    try:
        hsv_small, hsv_small_no_bg, hsv_small_bin, hsv_range = list(csgm.hsv_differentiation(small, get_range=True, seg_type=0))
        #lab_small, lab_small_no_bg, lab_small_bin, lab_range = list(sgm.hsv_differentiation(small, get_range=True, seg_type=1))
        #rgb_small, rgb_small_no_bg, rgb_small_bin, rgb_range = list(sgm.hsv_differentiation(small, get_range=True, seg_type=2))

        stack = Stack([square_img, small, hsv_small, hsv_small_no_bg, hsv_small_bin])
    except Exception as e:
        stack = Stack([square_img, small], size=(2, 3), is_filler_empty=True)
        if e.args[0] != EMPTY_HISTO:
            print(helpers.get_line_num(), ". e:", repr(e))

        return stack, None

    return stack, (hsv_range)


def stage3(img, non_seg_img):
    # Separate hand from background through hsv difference
    hsv_img, hsv_edge_bin, hsv_edge_no_bg = csgm.hsv_differentiation(img, has_params=True, params=ranges["hsv"][1], seg_type=0)
    #hue = hsv_img[:, :, 0] # to view hue with breakpoint

    dilated_bin = cv2.dilate(hsv_edge_bin, (15,15), iterations=7)
    dilated_img = cv2.bitwise_and(non_seg_img, non_seg_img, mask=dilated_bin)
    no_white_bin, no_white = sgm.threshold_white_rgb(dilated_img, thresh=180)


    sat_bin, sat_img = sgm.threshold_low_sat(no_white, thresh=15)
    opened_sat_bin, opened_sat = open_close(non_seg_img, sat_bin)


    hand_bin = helpers.get_biggest_object(opened_sat_bin)
    hand_img = cv2.bitwise_and(no_white, no_white, mask=hand_bin)

    no_dark_bin, no_dark_img = sgm.threshold_dark_spots(hand_img, thresh=45)
    hand_bin2 = helpers.get_biggest_object(no_dark_bin)
    hand_img2 = cv2.bitwise_and(no_white, no_white, mask=hand_bin2)

    stack = Stack([img, hsv_img, hsv_edge_no_bg,
                   dilated_img, hand_img, no_dark_img, hand_img2], size=(2,4), hand_img=hand_img2)
    stack.hand_img=None
    return stack



def edge_segmentation(img):
    scale = 31
    img_blur = cv2.GaussianBlur(img,(scale,scale),1)

    blur_vars = sgm.region_based_segmentation(img_blur)
    normalized = helpers.normalize_zero1_to_zero255(blur_vars[1])

    main_area_img = cv2.bitwise_and(img, img, mask=normalized)

    return main_area_img







def run_avg(img_gray, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = img_gray.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(img_gray, bg, aWeight)



def subtract_bg(img, threshold=50, is_sat=False):
    global bg
    gray = helpers.get_gray_blurred_img(img)
    if is_sat:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
        gray = cv2.GaussianBlur(gray, (3,3), 0)
    # find the absolute difference between background and current frame
    # noinspection PyUnresolvedReferences
    diff = cv2.absdiff(bg.astype("uint8"), gray)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff.copy(), threshold, 255, cv2.THRESH_BINARY)[1]

    img_max_contour, arm_bin = do_segment_thingy(img, gray, thresholded)

    arm_img = cv2.bitwise_and(img, img, mask=arm_bin)

    stack = Stack([img, diff, thresholded, img_max_contour, arm_bin, arm_img])

    return stack




def do_segment_thingy(img, gray, edge_or_thresh):
    arm_bin = helpers.get_blank_img(gray)
    img_max_contour = helpers.get_blank_img(img)

    # get the contours in the thresholded/edge image
    cnts, _ = cv2.findContours(edge_or_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if there are contours, get arm
    if cnts is not None and len(cnts) > 0:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        helpers.draw_contours(img, img_canvas=img_max_contour, contours=[segmented], min_area=0, draw_pts=False)
        cv2.fillPoly(arm_bin, pts=[segmented], color=255)


    return img_max_contour, arm_bin





def analyze_segmented_img(img, binary):
    data = Data()  # To store all the data and use it to execute the right command

    blank_img = img.copy()
    blank_img[:] = 0, 0, 0

    img_transformed = general.distanceTransform(binary)

    lower_points_img, fings_count_pts = pts.find_lower_points(img)

    # Find the center of the hand from the distance transformation
    thresh, center_img = cv2.threshold(img_transformed, 253, 255, cv2.THRESH_BINARY)

    circle = fings.getCircle(img_transformed)

    fingers = cv2.subtract(binary, circle, mask=None)
    fingers, fings_count = fings.find_fingers(fingers)
    #cv2.putText(fingers, str(fings_count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)

    img_hsv_pts = mouse_handler.show_extreme_points(img.copy(), binary)
    img_pts = mouse_handler.show_north_extreme_points(img.copy(), binary, fings_count)


    fings_width_len_list = str(fings.get_fingers_data(fingers))
    #db.update("fings_width_len_list", '\"' + fings_width_len_list + '\"')

    stack = Stack([img, binary, lower_points_img, img_transformed,
                       center_img, circle, fingers, img_hsv_pts, img_pts])


    #  Add number of fingers up to data
    if (fings_count == fings_count_pts):
        data.fings_count = fings_count

    data.fings_count = fings_count

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
