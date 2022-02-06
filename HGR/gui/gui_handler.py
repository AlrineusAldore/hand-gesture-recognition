# Constants for pygame
import gui.get_squer_color
import gui.constants
import cv, cv2
import numpy as np
import wx

class MyApp(wx.App):
    def __init__(self):
        super().__init__(clearSigInt=True)
        capture = cv2.VideoCapture(0)

        frame = wx.Frame(None)

        cap = ShowCapture(frame, capture)
        frame.Show()




class ShowCapture(wx.Panel):
    def __init__(self, parent, capture, fps=15):
        wx.Panel.__init__(self, parent)
        text = "Put your hand in the square for five seconds after pressing the button"
        self.welcomeText = wx.StaticText(self, id=wx.ID_ANY, label=text, pos=(50, 50))

        take_a_picture = wx.Button(self, label="take a picture")
        take_a_picture.Bind(wx.EVT_BUTTON, self.take_a_picture)

        self.capture = capture
        ret, frame = self.capture.read()

        height, width = frame.shape[:2]
        parent.SetSize((width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.bmp = wx.BitmapFromBuffer(width, height, frame)

        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.NextFrame)


    def take_a_picture(self, event):
        """"""
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = gui.get_squer_color.draw_squer(frame)
            gui.get_squer_color.get_dominant_color(frame)
            self.bmp.CopyFromBuffer(frame)
            self.Refresh()
        self.Close()


    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def NextFrame(self, event):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = gui.get_squer_color.draw_squer(frame)
            self.bmp.CopyFromBuffer(frame)
            self.Refresh()


def make_gui():
    app = MyApp()
    app.MainLoop()

    return app