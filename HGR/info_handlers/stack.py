import cv2
import numpy as np
import math
from helpers import get_blank_img


class Stack():
    def __init__(self, lst=None, size=(1,1), is_filler_empty=False, hand_img=None):
        if lst is None:
            lst = []
        self.lst = lst
        self.size = size
        self.img_stack = None
        self.is_filler_empty = is_filler_empty
        self.hand_img = hand_img

        self.organize(size)


    def organize(self, size):
        if len(self.lst) == 0:
            return
        rows = size[0]
        cols = size[1]

        if rows*cols >= len(self.lst):
            filler_img = self.__create_filler_img()
            self.img_stack = []
            for i in range(rows):
                temp = []
                for j in range(cols):
                    # After lst runs out, start copying filler image to fill up space
                    try:
                        temp.append(self.lst[i*cols + j])
                    except:
                        temp.append(filler_img)
                self.img_stack.append(temp)
            self.size = (rows, cols)
        else:
            self.auto_organize()


    # Automatically organizes stack with size of approximately sqrt(len)*sqrt(len)
    def auto_organize(self):
        #Get root of length and fractional digits
        length = len(self.lst)
        rows = math.sqrt(length)
        rem = rows % 1

        # If number whole, remain with root as rows & cols
        # If not whole, then have root+1 as cols, and rows appropriately too
        if rem != 0:
            rows = int(rows)
            if rows * (rows+1) < length:
                rows += 1
                cols = rows
            else:
                cols = rows+1
        else:
            rows = math.ceil(rows)
            cols = rows

        self.organize((rows, cols))

    # Appends a new img to the list
    def append(self, img):
        self.lst.append(img)
        self.organize(self.size)


    # Resizes stack stack
    def resize(self, size):
        if self.lst is not None:
            self.organize(size)


    def __create_filler_img(self):
        filler_img = get_blank_img(self.lst[0])

        if not self.is_filler_empty:
            cv2.putText(filler_img, "filler", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        return filler_img


    #turns the img_stack into an actual stack of imgs with size of scale
    def to_viewable_stack(self, scale):
        img_array = self.img_stack.copy()

        rows = len(img_array)
        cols = len(img_array[0])
        rowsAvailable = isinstance(img_array[0], list)
        width = img_array[0][0].shape[1]
        height = img_array[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    hi = img_array[x][y].shape[:2]
                    bye = img_array[0][0].shape[:2]
                    if hi == bye:
                        img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                    else:
                        img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                    if len(img_array[x][y].shape) == 2:
                        img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
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
