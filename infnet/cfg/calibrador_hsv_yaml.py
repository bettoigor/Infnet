# -*- coding: utf-8 -*-
#############################################
#                                           #
# Script to camera calibration using HSV    #
# color space. The script outputs a .YAML   #
# file used to segmetation in the node.     #
#                                           #
# Author: Adalberto Oliveira                #
# Autonomous Vehicle - Infnet               #
# Version: 1.0                              #
# Date: 03-03-2021                          #
#                                           #
#############################################

import cv2, sys, time
import numpy as np

def nothing(x):
   pass

def create_control(name):
    # creating control window
    cv2.namedWindow(name)
    cv2.createTrackbar('Hue Minimo',name,0,255,nothing)
    cv2.createTrackbar('Hue Maximo',name,0,255,nothing)
    cv2.createTrackbar('Saturation Minimo',name,0,255,nothing)
    cv2.createTrackbar('Saturation Maximo',name,0,255,nothing)
    cv2.createTrackbar('Value Minimo',name,0,255,nothing)
    cv2.createTrackbar('Value Maximo',name,0,255,nothing)

# starting
num_mask = int(input('How many masks do you want to create?: '))
file_name = input('Enter file name: ')
print('Press <Esc> when you finish')

# creating capture
videoIn = 0 if len(sys.argv)<2 else int(sys.argv[1])
cap = cv2.VideoCapture(videoIn)

# creatig file content
mask_file = []
mask_file.append('num_masks: '+str(num_mask)+ '\n')

for i in range(0,num_mask):
    print('Creating mask ',i+1)
    window_name = 'Mask '+ str(i+1)
    create_control(window_name)
    while True:

        # reading camera frame
        _,cv_image = cap.read() 

        # converting to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # getting Trackbars values
        hMin = cv2.getTrackbarPos('Hue Minimo',window_name)
        hMax = cv2.getTrackbarPos('Hue Maximo',window_name)
        sMin = cv2.getTrackbarPos('Saturation Minimo',window_name)
        sMax = cv2.getTrackbarPos('Saturation Maximo',window_name)
        vMin = cv2.getTrackbarPos('Value Minimo',window_name)
        vMax = cv2.getTrackbarPos('Value Maximo',window_name)

        # creating and arry with max and min values
        lower=np.array([hMin,sMin,vMin])
        upper=np.array([hMax,sMax,vMax])

        # detecting color
        mask = cv2.inRange(hsv, lower, upper)

        # showing output
        mask_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        output = np.hstack((cv_image, mask_show))
        cv2.imshow(window_name,output)

        # waiting key to finish
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            # saving mask params
            mask_file.append('mask_'+str(i+1)+':\n')
            mask_file.append('  low: ['+ str(hMin) + ', ' + str(sMin) + ', ' + str(vMin) + ']\n')
            mask_file.append('  high: ['+ str(hMax) + ', ' + str(sMax) + ', ' + str(vMax) + ']\n')

            # closing window
            cv2.destroyAllWindows()
            break

# creating yaml file
arq = open(file_name+".yaml", 'w')
arq.writelines(mask_file)
arq.close()
print(mask_file)
cv2.destroyAllWindows()
