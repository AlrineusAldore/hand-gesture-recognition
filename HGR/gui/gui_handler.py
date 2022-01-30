from gui.constants import *
import cv2
import numpy as np
import wx


class MyApp(wx.App):
    def __init__(self):
        super().__init__(clearSigInt=True)

        self.init_frame()

    def init_frame(self):
        self.frame = MyFrame(parent=None, title="framu", pos=(500, 500))
        self.frame.Show()



class MyFrame(wx.Frame):
    def __init__(self, parent, title, pos):
        super().__init__(parent=parent, title=title, pos=pos)

        self.panel = MyPanel(parent=self)



class MyPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        #text = "Put your hand in the square for five seconds after pressing the button"
        #self.welcomeText = wx.StaticText(self, id=wx.ID_ANY, label=text, pos=(50, 50))

    def put_img(self, img, coords=(0,0)):
        img = cvimage_to_wx(img)
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        dc.DrawBitmap(img, coords)
        #self.currentBitmap = img
        #self.Refresh()
        #wx.PaintDC.DrawBitmap(img)
        #wx.StaticBitmap(self, -1, img, coords, (img.GetWidth(), img.GetHeight()))


def make_gui():
    app = MyApp()
    #app.MainLoop()

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
