#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
################################
#	                                                                #
# GPS fake: service for Aruco Tag position 	#
#	                                                                #
# Author: Adalberto Oliveira	                                #
# Masterin in robotic - PUC- Rio - BR	                #
# Vesion: 1.0	                                                #
# Date: 9-19-2018	                                                #
#	                                                                #
################################
'''

import rospy, cv2, math, time, sys
import numpy as np
import cv2.aruco as ar
from sensor_msgs.msg import NavSatFix, Image
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
from gps_cam.msg import GpsCam
from gps_cam.srv import GpsCamPose

def calc_xyth(xi,yi,xf,yf, x3, y3, rate):
    global centerX
    global centerY
    
    xFrame = int((((xf - xi)/2) + xi) - centerX)
    yFrame = int(centerY - (((yf - yi)/2) + yi))
    theta =  math.atan2((x3 - xi),(y3 - yi))

    x = round(xFrame * rate,2)
    y = round(yFrame * rate, 2)
    
    return x, y, theta


def get_coordinates(arCodes):
    global rate
    
    if len(arCodes[0]) > 0:
        idCodes = [0] * len(arCodes[1])
        coordinates = [[0 for x in range(5)] for y in range(len(arCodes[1]))]
        for i in range(len(arCodes[0]) ): 
            xi = arCodes[0][i][0][0][0]
            yi = arCodes[0][i][0][0][1]
            xf = arCodes[0][i][0][2][0]
            yf = arCodes[0][i][0][2][1]
            x3 = arCodes[0][i][0][3][0]
            y3 = arCodes[0][i][0][3][1]
            x, y, theta = calc_xyth(xi,yi,xf,yf, x3, y3, rate)
            thetaDeg = round((theta * 180)/3.141592)    #degrees
            coordinates[i]= [x, y, thetaDeg]
            idCodes[i] = arCodes[1][i]

    else:
        coordinates = []
        idCodes = []

    return idCodes, coordinates
 
def get_frame():
    
    global camera

    ret, frame = camera.read()


    return frame
    

def get_tags():
    
    #frame = get_frame()
    dic = ar.getPredefinedDictionary(ar.DICT_ARUCO_ORIGINAL) #aruco dictionary
    ret, frame = camera.read()
    cv2.imshow('Camera',frame)
    cv2.waitKey(2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    arCodes = ar.detectMarkers(gray,dic)
    numId = len(arCodes[0])
    print numId
    return arCodes


def set_init_param(argv):

    if len(argv) < 2:
        h = 46 # distance between gound and camera 
        apperture = 0.376909270325 # aperture for 4tech camera (DO NOT CHANGE!!!!)
        video = 0 # webcam port
        print 'Using camera default parameters\nAltitude: ', h,' \nAperture: ',apperture,' \nVideo: ', video
    else:
        if len(argv) == 2:
            h = float(argv[1])
            apperture = 0.376909270325 # aperture for 4tech camera (DO NOT CHANGE!!!!)
            video = 0 # webcam port
            print 'Change: \nAltitude: ', h
        elif len(argv) == 3:
            h = float(argv[1])
            apperture = float(argv[2])
            video = 0
            print 'Changes: \nAltitude: ', h,' and Aperture: ',apperture
        else:# len(argv) == 4:
            h = float(argv[1])
            apperture = float(argv[2])
            video = int(argv[3])
            print 'Using camera user parameters\nAltitude: ', h,' \nAperture: ',apperture,' \nVideo: ', video
        
    rate = ((math.tan(apperture)) * h *2)/640        
    camera = cv2.VideoCapture(video)

    return rate, camera

def get_tag_position(tag):
    
    idCodes, coord  = get_coordinates(get_tags())
   
    if (tag in idCodes):
        index = idCodes.index(tag)
        coordinate = coord[index]
    else:
        coordinate = False
    
    return coordinate


def callback_gps(msg):

    position = get_tag_position(msg.tag)
    #print msg.tag, ', ',position 
    gps = Pose2D()
    if not position:
        gps.theta = 200
    else:
        gps.x = position[0]
        gps.y = position[1]
        gps.theta = position[2]
        
    return gps


def test():    
    
    tag = int(input('Type the tag for monitoring: '))
    while(True):
        #print get_coordinates(get_tags())
        position = get_tag_position(tag)
        if  not position:
            print 'not found'
        else:
            x = position[0]
            y = position[1]
            th = position[2]
            print 'Requested Position: \nX: ',x, '\nY: ', y, '\nTheta: ', th


def gps_server():
    
    rospy.init_node('gps_cam_service', anonymous=True)
    rospy.Service('gps_position', GpsCamPose, callback_gps)
    print 'Ready for requests!'
    #rospy.spin()
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        get_frame()
        get_tags()
        rate.sleep()
   
centerX = 320
centerY = 240
argv = sys.argv
rate, camera = set_init_param(argv)

if __name__ == '__main__':
    try:
        gps_server()
    except Exception as e:
        print 'Program finished.\nCause: ', e
        sys.exit(1)

