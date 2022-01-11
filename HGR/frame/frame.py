import cv2
import numpy as np
import math


class Frame():
    def __init__(self, lst=None, size=(1,1)):
        if lst is None:
            lst = []
        self.lst = lst
        self.size = size
        self.img_stack = None

        self.organize(size)


    def organize(self, size):
        if len(self.lst) == 0:
            return
        rows = size[0]
        cols = size[1]

        if rows*cols >= len(self.lst):
            self.img_stack = []
            temp = []
            for i in range(rows):
                for j in range(cols):
                    temp.append(self.lst[i*rows + j])
                self.img_stack.append(temp)
            self.size = (rows, cols)
        else:
            self.auto_organize()


    # Automatically organizes stack as approximately sqrt(len)*sqrt(len)
    def auto_organize(self):
        #Get root of length and fractional digits
        rows = math.sqrt(len(self.lst))
        rem = rows % 1
        cols = rows

        # If number whole, remain with root as rows & cols
        # If not whole, then have root+1 as cols
        # Rows are root+1 if rem >0.45, otherwise rows=root
        if rem != 0:
            rows = round(rows)
            if rem > 0.45:
                rows += 1
                cols = rows
            else:
                cols = rows+1

        self.organize((rows, cols))

    # Appends a new img to the list
    def append(self, img):
        self.lst.append(img)
        self.organize(self.size)


    # Resizes frame stack
    def resize(self, size):
        if self.lst is not None:
            self.organize(size)


    #turns the img_stack into an actual stack of imgs with size of scale
    def stack(self, scale):
        img_array = self.img_stack

        rows = len(img_array)
        cols = len(img_array[0])
        rowsAvailable = isinstance(img_array[0], list)
        width = img_array[0][0].shape[1]
        height = img_array[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    if img_array[x][y].shape[:2] == img_array[0][0].shape [:2]:
                        img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                    else:
                        img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                    if len(img_array[x][y].shape) == 2: img_array[x][y]= cv2.cvtColor( img_array[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(img_array[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if img_array[x].shape[:2] == img_array[0].shape[:2]:
                    img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
                else:
                    img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None,scale, scale)
                if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
            hor= np.hstack(img_array)
            ver = hor
        return ver
