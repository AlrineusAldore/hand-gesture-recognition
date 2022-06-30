import wx
import cv2
import time
import numpy as np
import matplotlib
import wx.lib.platebtn as platebtn

start = time.time()
elapsed = 0

class capture_panel(wx.Panel):
    def __init__(self, parent, capture, fps=15):
        wx.Panel.__init__(self, parent = parent)
        self.capture = capture
        ret, frame = self.capture.read()

        height, width = frame.shape[:2]
        parent.SetSize(width+20, height+20)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.SetBackgroundColour("red")

        self.bmp = wx.BitmapFromBuffer(width, height, frame)

        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)

        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_TIMER, self.nextFrame)

    def onPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def nextFrame(self, event):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = drawRectangle(frame)[0]
            self.bmp.CopyFromBuffer(frame)
            self.Refresh()

class capture_panel_hand(wx.Panel):
    def __init__(self, parent, capture, fps=15):
        wx.Panel.__init__(self, parent = parent)

        self.capture = capture
        ret, frame = self.capture.read()

        height, width = frame.shape[:2]
        parent.SetSize(width+20, height+20)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.SetBackgroundColour("red")

        self.bmp = wx.BitmapFromBuffer(width, height, frame)

        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)

        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_TIMER, self.nextFrame)

        self.SetBackgroundColour("red")
        wx.Button(self, -1, "filter", pos=(100, 100), style=platebtn.PB_STYLE_GRADIENT)
        wx.Button(self, -1, "filter1", pos=(200, 100))
        wx.Button(self, -1, "filter2", pos=(300, 100))
        wx.Button(self, -1, "filter3", pos=(400, 100))
        wx.Button(self, -1, "filter4", pos=(100, 200))
        wx.Button(self, -1, "filter5", pos=(200, 100))

    def onPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def nextFrame(self, event):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = drawRectangle(frame)[1]
            self.bmp.CopyFromBuffer(frame)
            self.Refresh()

def drawRectangle(frame):
    startPoint = (300,400)
    endPoint = (100, 200)
    color = (255, 0, 0) # blue rectangle
    # Cropping an image to get part of hand and not the background
    w = int(frame.shape[1])
    h = int(frame.shape[0])
    dim = (w, h)

    cropped_image = frame[200:400, 100:300]
    cropped_image = cv2.rectangle(cropped_image, (0,150),(200, 200), color, -1)

    resized = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)

    return cv2.rectangle(frame, endPoint,startPoint, color, 2), resized

def stopwatch(seconds, elapsed, start):
    elapsed = time.time() - start
    if elapsed < seconds:
        return False
    return True