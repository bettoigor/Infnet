#!/usr/bin/env python
import numpy as np
import cv2

img = cv2.imread('fig.png',0)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
th,img = cv2.threshold(img,
                        0,255,
                        cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow('Gray',img)

cv2.waitKey()
