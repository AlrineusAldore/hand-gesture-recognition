import segmentation.segmentation as sgm
import frame.frame as fr
import analysis.fingers as fings
import analysis.points as pts
from analysis import mouse_handler
import analysis.general as general
import helpers
import cv2
import matplotlib.pyplot as plt
from gui import gui_handler as gui
import commands.commands_handler as cmds
#from cython_funcs import helpers_cy as cy
import sql.mySQL as sqlit
db = sqlit.database()

#recist code


VID_NAME = "Videos\\handGesturesVid.mp4"
SET_VALUES_MANUALLY = False


def main():
    plt.show()
    plt.ion()

    if SET_VALUES_MANUALLY:
        helpers.InitializeWindows()

    #app = gui.make_gui()

    #print("cython output:", cy.test(5))
    #analyze_capture(VID_NAME, 0, app)  # Analyzing a video with gui
    analyze_capture(VID_NAME, 5, 0)  # Analyzing a video
    #analyze_capture(0, 0, 0)  # Analyzing camera



# Fully analyzes a whole capture
def analyze_capture(cap_path, frames_to_skip, app):
    cap = cv2.VideoCapture(cap_path)
    n = 0
    data = {}  # To store all the data and use it to execute the right command
    cmds_handler = cmds.CommandsHandler(data)

    #loop forever
    while cap.isOpened():

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

        frame = fr.Frame()

        img = img[160:490, 0:330]
        img = cv2.resize(img, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)

        blank_img = img.copy()
        blank_img[:] = 0, 0, 0

        # Separate hand from background through hsv difference
        img_hsv, ready_binary, ready_img = sgm.hsv_differentiation(img, False, SET_VALUES_MANUALLY)


        img_transformed = general.distanceTransform(ready_binary)

        lower_points_img, fings_count_pts = pts.find_lower_points(ready_img)
        frame.append(lower_points_img)

        # Find the center of the hand from the distance transformation
        thresh, center_img = cv2.threshold(img_transformed, 253, 255, cv2.THRESH_BINARY)

        circle = fings.getCircle(img_transformed)

        fingers = cv2.subtract(ready_binary, circle, mask=None)
        fingers, fings_count = fings.find_fingers(fingers)

        img_hsv_pts = mouse_handler.show_extreme_points(img.copy(), ready_binary)
        img_pts = mouse_handler.show_north_extreme_points(img.copy(), ready_binary, fings_count)
        frame.append(img)

        fings_width_len_list = str(fings.get_fingers_data(fingers))
        db.update("fings_width_len_list", '\"' + fings_width_len_list + '\"')
        frame.lst += [img_transformed, center_img, circle, fingers]
        frame.lst += [img_hsv, ready_binary, ready_img]
        frame.lst += [img_hsv_pts, img_pts]
        frame.auto_organize()


        #  Add number of fingers up to data
        if (fings_count == fings_count_pts):
            data["fings_count"] = fings_count
        else:
            data["fings_count"] = None


        #  Do command
        cmds_handler.update_data(data)
        result = cmds_handler.check_commands()

        #  Makes an image showing only the used command
        command_img = blank_img.copy()
        cv2.putText(command_img, result, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        frame.append(command_img)

        stack = frame.stack(1.5)

        #app.frame.panel.put_img(stack)
        cv2.imshow("stack", stack)

        #histo = fr.Frame([img_hsv, ready_binary, ready_img] + list(sgm.hsv_differentiation(img, True, False)))
        #cv2.imshow("histo", histo.stack(2))
        #cv2.waitKey(0)

        #if 'q' is pressed, close all windows and break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        db.printData()

    return frame




if __name__ == "__main__":
    main()
