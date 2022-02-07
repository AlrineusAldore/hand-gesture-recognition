import wx
import cv2
import numpy as np

class ShowCapture(wx.Panel):
    def __init__(self, parent, capture, fps=15):
        wx.Panel.__init__(self, parent)

        self.capture = capture
        ret, frame = self.capture.read()
        height, width = frame.shape[:2]
        parent.SetSize((width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
            frame = drawRectangle(frame)
            self.bmp.CopyFromBuffer(frame)
            self.Refresh()

def drawRectangle(frame):
    startPoint = (240,350)
    endPoint = (110, 220)
    color = (255, 0, 0) # blue rectangle
    # Cropping an image to get part of hand and not the background

    cropped_image = frame[240:350, 110:220]
    cv2.imshow("cropped", cropped_image)
    return cv2.rectangle(frame, startPoint, endPoint, color, 2)

