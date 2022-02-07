from gui.constants import *
from gui.capture_hanlder import *
import cv2
import numpy as np
import wx


class MyApp(wx.App):
    def __init__(self):
        super().__init__(clearSigInt=True)

        self.init_frame()


def make_gui():
    capture = cv2.VideoCapture(0)
    app = wx.App()
    frame = wx.Frame(None)
    ShowCapture(frame, capture)

    frame.Show()
    app.MainLoop()
    return app


def update_img(img):
    pass


def cvimage_to_wx(cv2_image):
    height, width = cv2_image.shape[:2]

    info = np.iinfo(cv2_image.dtype) # Get the information of the incoming image type
    data = cv2_image.astype(np.float64) / info.max # normalize the data to 0 - 1
    data = 255 * data # Now scale by 255
    cv2_image = data.astype(np.uint8)

    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return wx.Bitmap.FromBuffer(width, height, cv2_image_rgb)
