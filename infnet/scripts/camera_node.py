#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################################
#                                           #
# Node to control the simulator TurtleSim.  #
# This node receives a goal coordinate from #
# topc /goal and controls the turtle.       #
#                                           #
# Author: Adalberto Oliveira                #
# Autonomous Vehicle - Infnet	            #
# Version: 1.0                              #
# Date: 02-03-2021                          #
#                                           #
#############################################


# importing libraries
import rospy, time, sys, cv2
import numpy as np
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def get_img(video):

    # Lendo a camera
    _,cv_image = video.read() 

    # Criando o conversor de imagem para ros msg	
    bridge = CvBridge()

    img_msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
    cv2.namedWindow('Webcam')
    cv2.imshow('Webcam',cv_image)
    cv2.waitKey(5)
    
    
    return img_msg


def camera_main():
    """
    This function is called from the main conde and calls 
    all work methods of fucntions into the codde.
    """

    # Global variables
    global cap

    # Initializing ros node
    rospy.init_node('camera_node', anonymous=True)    # node name
    
   # Publishers
    pub_image = rospy.Publisher('image_raw', Image, queue_size=10)  # send control signals

    # control rate
    rate = rospy.Rate(30)   # run the node at 15H
    pub_img = Image()

    # main loop
    while not rospy.is_shutdown():

        print('Camera node running ok!')
        #get_img(cap)
        pub_img = get_img(cap)

        pub_image.publish(pub_img)
        
        rate.sleep()
    




video = int(sys.argv[1])   # webacam

# Criando a porta de captura de video 
cap = cv2.VideoCapture(video)

if __name__ == '__main__':
    camera_main()


