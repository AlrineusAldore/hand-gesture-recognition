import cv2
import numpy as np
import math


class Frame():
    def __init__(self, *lst ,img=None, size=(1,1)):
        self.img = img
        self.lst = lst
        self.size = size
        self.count = len(lst)
        self.img_stack = None

        self.organize(size)

    def organize(self, *size):
        tot_size = size[0]*size[1]
        #
        if tot_size >= len(self.lst):
            self.img_stack = []
            temp = []
            for i in range(size[0]):
                for j in range(size[1]):
                    temp.append(self.lst[i*size[0] + j])
        else:
            self.auto_organize()





    def auto_organize(self):
        root = math.sqrt(self.count)
        rem = root % 1

        if rem != 0:
            if rem >= 0.5:
                root = math.ceil(root)
            else:
                root = round(root)



    def stack(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor= np.hstack(imgArray)
            ver = hor
        return ver
