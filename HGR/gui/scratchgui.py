import wx
FILE_NAME = "C:\\Users\\magshimim\\PycharmProjects\\karmiel-504-hgr-v\\hand gesture ideal pictures\\palm.jpeg"
class MainPanel(wx.Panel):
    def __init__(self, parent, bg_img=FILE_NAME):
        wx.Panel.__init__(self, parent=parent)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.bg = wx.Bitmap(bg_img)
        self._width, self._height = self.bg.GetSize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        hSizer = wx.BoxSizer(wx.HORIZONTAL)

        for num in range(4):
            btn = wx.Button(self, label="Button %s" % num)
            sizer.Add(btn, 0, wx.ALL, 5)
        hSizer.Add((1,1), 1, wx.EXPAND)
        hSizer.Add(sizer, 0, wx.TOP, 100)
        hSizer.Add((1,1), 0, wx.ALL, 75)
        self.SetSizer(hSizer)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

    def OnSize(self, size):
        self.Layout()
        self.Refresh()

    def OnEraseBackground(self, evt):
        pass

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        self.Draw(dc)

    def Draw(self, dc):
        cliWidth, cliHeight = self.GetClientSize()
        if not cliWidth or not cliHeight:
            return
        dc.Clear()
        xPos = (cliWidth - self._width)/2
        yPos = (cliHeight - self._height)/2
        dc.DrawBitmap(self.bg, xPos, yPos)

app = wx.App()
frame = wx.Frame(None, size=(400,300))
panel = MainPanel(frame)
frame.Show()
app.MainLoop()