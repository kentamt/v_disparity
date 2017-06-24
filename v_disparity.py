#!/usr/bin/env python2
#-*- coding: utf-8 -*-

import cv2
import numpy as np
from numpy.random import *


def draw_object(img, x, y, w=50, h=100):
    color = img[y, x]
    img[y-h:y, x-w//2:x+w//2] = color


height = 480
width = 640
max_disp = 200
img = np.zeros(( height, width), np.uint8)
for y in range(height)[::-1]:
        img[y, ...] = int(float(y) / height * max_disp) + abs(normal(0,5))


draw_object(img, 275, 175, w=20, h=50)
draw_object(img, 300, 200, w=20, h=50)
draw_object(img, 300, 300, w=50, h=50)
draw_object(img, 300, 400, w=300, h=50)
draw_object(img, 100, 150, w=20, h=50)


# V-disparity
vhist_vis = np.zeros((height, max_disp), np.float)
for i in range(height):
    vhist_vis[i, ...] = cv2.calcHist(images=[img[i, ...]], channels=[0], mask=None, histSize=[max_disp], ranges=[0, max_disp]).flatten() / float(height)

vhist_vis = np.array(vhist_vis * 255, np.uint8)
vblack_mask = vhist_vis < 5
vhist_vis = cv2.applyColorMap(vhist_vis, cv2.COLORMAP_JET)
vhist_vis[vblack_mask] = 0

# U-disparity
uhist_vis = np.zeros((max_disp, width), np.float)
for i in range(width):
    uhist_vis[..., i] = cv2.calcHist(images=[img[..., i]], channels=[0], mask=None, histSize=[max_disp], ranges=[0, max_disp]).flatten() / float(width)
    
uhist_vis = np.array(uhist_vis * 255, np.uint8)
ublack_mask = uhist_vis < 5
uhist_vis = cv2.applyColorMap(uhist_vis, cv2.COLORMAP_JET)
uhist_vis[ublack_mask] = 0


# save result
img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
cv2.imwrite('disparity.png', img)
cv2.imwrite('v_disparity.png', vhist_vis)
cv2.imwrite('u_disparity.png', uhist_vis)
