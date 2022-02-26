import wx
from gui.capture_panel import *
import cv2
#from analysis.GestureRecognition import *
import wx.lib.platebtn as platebtn

class control_panel(wx.Panel):
    def __init__(self, parent, capture, fps=15):
        wx.Panel.__init__(self, parent = parent)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)

        self.capture = capture
        ret, frame = self.capture.read()

        height, width = frame.shape[:2]
        parent.SetSize(width, height)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.bmp = wx.BitmapFromBuffer(width, height, frame)

        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)

        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_TIMER, self.nextFrame)

        test = []
        button_id = []
        for i in range(10):
            test.append(i)
            button_id.append(wx.NewId())

        self.button = []
        for i in range(len(test)):
            self.button.append(wx.Button(self, button_id[i], label=(str(test[i]))))
            self.button[i].Bind(wx.EVT_BUTTON, self.OnButton)

        wrapper = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.FlexGridSizer(rows=2, cols=5, vgap=5, hgap=5)

        for i in self.button:
            sizer.Add(i, 0, wx.ALL, 0)

        sizer.AddGrowableRow(1, 1)
        sizer.AddGrowableRow(0, 1)

        for col in range(sizer.Cols):
            sizer.AddGrowableCol(col, 1)

        wrapper.Add(sizer, proportion=1, flag=wx.ALL | wx.EXPAND)
        self.SetSizer(wrapper)

    def OnButton(self, event):
        Id = event.GetId()
        Obj = event.GetEventObject()
        #here you need to connect to functions from gestureRecognition
        print("Button Id", Id)
        print("Button Pressed:", Obj.GetLabelText())

    def OnSize(self, size):
        self.Layout()
        self.Refresh()

    def OnEraseBackground(self, evt):
        pass

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

class control_panel_2(wx.Panel):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent = parent)
        wx.Button(self, -1, "Button 3")
