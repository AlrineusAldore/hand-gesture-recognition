from constants import *
import pygame
import cv2
import numpy as np
import wx


class MyApp(wx.App):
    super().__init__(clearSigInt=True)

def make_gui():
    screen=pygame.display.set_mode((SC_WIDTH, SC_HEIGHT))
    screen.fill(0) #set pygame screen to black
    #frame=getCamFrame(color,camera)
    #screen=blitCamFrame(frame,screen)
    pygame.display.flip()



def update_img(img):
    pass





def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "RGB")
