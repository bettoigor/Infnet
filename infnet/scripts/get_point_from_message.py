#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#############################################
#                                           #
# Node to receive image messagens and send  #
# the centroind and base point coordinates  #
# of segmented images.                      #
#                                           #
# Changes:                                  #
#    * Using new libraty image_lib_v2;      #
#    * Updated to Python3 and ROS Noetic    #
#                                           #
# Author: Adalberto Oliveira                #
# Autonomous Vehicle - Infnet	            #
# Version: 1.1                              #
# Date: 13 mar 2021                         #
#                                           #
#############################################


# importing libraries
import rospy, time, sys, cv2
import numpy as np
import image_lib_v2 as img
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback_img(msg):

    global input_img

    # creating ros bridge
    bridge = CvBridge()

    # receive the ros image mesage and convert to bgr, ohta and hsv  
    input_img = bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")   

def camera_main():

    """
    This function is called from the main conde and calls 
    all work methods of fucntions into the codde.
    """

    # Global variables
    global cap
    global mask_h
    global mask_l
    global num_masks
    global input_img
    global show_img


    # Initializing ros node
    rospy.init_node('camera_node', anonymous=True)    # node name
    
    # Publishers
    pub_goal_centroid = rospy.Publisher('goal_centroid', Pose2D, queue_size=10)  # send control signals
    pub_goal_base = rospy.Publisher('goal_base', Pose2D, queue_size=10)  # send control signals
    
    # Subscribers
    rospy.Subscriber('image_raw', Image, callback_img)
 
    # control rate
    rate = rospy.Rate(30)   # run the node at 15H
    pub_img = Image()
    input_img = []

    # variables of flow control 
    time.sleep(1)

    print('Starting goal puslisher')
    # main loop
    while not rospy.is_shutdown():

        # Creating variables
        mask = []
        centroid = []
        base = []

        for i in range(num_masks):        
            try:
                
                # Creating masks         
                mask.append(img.get_mask(input_img,mask_l[i],mask_h[i],im_blur=True))

                # Getting centroids    
                cent_, img_cont = img.get_centroid(input_img,mask[i])

                # Getting base points
                b_, img_cont = img.get_base(img_cont,mask[i])
                
                cv2.namedWindow('Centroides')
                cv2.imshow('Centroides',img_cont)
                cv2.waitKey(1) 

                # add points
                centroid.append(cent_)
                base.append(b_)
                print('mask OK!')
            except:
                
                centroid = None
                base = None
                img_cont = input_img
                print('mask nok')

        # Prepating publishing objects
        goal_centroid = Pose2D()

        if centroid is not None:
            goal_centroid.x = centroid[0][0]
            goal_centroid.y = centroid[0][1]
            goal_centroid.theta = 1            

       
        goal_base = Pose2D()       
        if base is not None:
            goal_base.x = base[0][0]
            goal_base.y = base[0][1]
            goal_base.theta = 1
            
        # Publishin... 
        pub_goal_centroid.publish(goal_centroid)
        pub_goal_base.publish(goal_base)


        
        # showing images
        if show_img:
            cv2.namedWindow('Centroides')
            cv2.imshow('Centroides',img_cont)
            cv2.waitKey(1)   
        
        rate.sleep()


################### MAIN CODE  ##############################
# Loading initial values of global variables
show_img = int(sys.argv[1])
print('Showing image:',show_img)

# loading params from rosparam
num_masks = rospy.get_param('/num_masks')

# creating masks
mask_h = np.empty([num_masks,3],dtype=np.uint8)
mask_l = np.empty([num_masks,3],dtype=np.uint8)

for i in range(0,num_masks):
    mask_l[i,:] = rospy.get_param('/mask_'+str(i+1)+'/low')
    mask_h[i,:] = rospy.get_param('/mask_'+str(i+1)+'/high')


if __name__ == '__main__':
    camera_main()


