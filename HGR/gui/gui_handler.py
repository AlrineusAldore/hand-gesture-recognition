from gui.constants import *
<<<<<<< HEAD
from gui.capture_hanlder import *
import cv2
=======
from gui.capture_panel import *
from gui.control_panel import *
import cv2
import matplotlib
>>>>>>> develop
import numpy as np
import wx


<<<<<<< HEAD
class MyApp(wx.App):
    def __init__(self):
        super().__init__(clearSigInt=True)

        self.init_frame()


def make_gui():
    capture = cv2.VideoCapture(0)
    app = wx.App()
    frame = wx.Frame(None)
    ShowCapture(frame, capture)

=======
class main_panel(wx.Frame):
    def __init__(self):
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        height, width = frame.shape[:2]
        wx.Frame.__init__(self, parent=None, title="main", size=(width*2, height))

        splitter = wx.SplitterWindow(self)
        top = capture_panel(splitter, capture)
        bottom = control_panel(splitter, capture)

        splitter.SplitVertically(top, bottom)
        splitter.SetMinimumPaneSize(width)

        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
    def OnCloseWindow(self, e):
        dial = wx.MessageDialog(None, "Are you sure you want to exit?", 'Question',
                                wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
        ret = dial.ShowModal()
        if ret == wx.ID_YES:
            self.Destroy()
        else:
            e.Veto()


def make_gui():
    app = wx.App()
    frame = main_panel()
>>>>>>> develop
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
<<<<<<< HEAD
    return wx.Bitmap.FromBuffer(width, height, cv2_image_rgb)
=======
    return wx.Bitmap.FromBuffer(width, height, cv2_image_rgb)
>>>>>>> develop
