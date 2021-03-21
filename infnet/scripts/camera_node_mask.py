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
import image_lib as img
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



def camera_main():
    """
    This function is called from the main conde and calls 
    all work methods of fucntions into the codde.
    """

    # Global variables
    global cap
    global mask_h
    global mask_l

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
        
        cv_image, pub_img = img.get_img_ros(cap)

        try:
        
            mask_laranja = img.get_mask(cv_image,
                                        mask_l[0],mask_h[0],
                                        im_blur=True)
            mask_verde = img.get_mask(cv_image,
                                        mask_l[1],mask_h[1],
                                        im_blur=True)  
            
            cent_l, img_cont = img.get_centroid(cv_image,
                                                mask_laranja, 
                                                put_text=True,
                                                drawn_contour=False)
            cent_v, img_cont = img.get_centroid(img_cont,
                                                mask_verde, 
                                                put_text=True,
                                                drawn_contour=False)

            base_l, img_cont = img.get_base(img_cont,mask_laranja, put_text=True)
            base_v, img_cont = img.get_base(img_cont,mask_verde, put_text=True)
        
    

        except:
            cent_v = [0,0]
            cent_l = [0,0]
            base_v = [0,0]
            base_l = [0,0]
            img_cont = cv_image

        cv2.namedWindow('Original')
        cv2.imshow('Original',cv_image)
        
        print(cent_v, cent_l)
        print(base_v, base_l)

        cv2.waitKey(5)

        cv2.namedWindow('Centroides')
        cv2.imshow('Centroides',img_cont)

        pub_image.publish(pub_img)
        
        rate.sleep()




# loading params from rosparam
num_masks = rospy.get_param('/num_masks')

# creating masks
mask_h = np.empty([num_masks,3],dtype=np.uint8)
mask_l = np.empty([num_masks,3],dtype=np.uint8)
for i in range(0,num_masks):
    mask_l[i,:] = rospy.get_param('/mask_'+str(i+1)+'/low')
    mask_h[i,:] = rospy.get_param('/mask_'+str(i+1)+'/high')



video = int(sys.argv[1])   # webacam

# Criando a porta de captura de video 
cap = cv2.VideoCapture(video)

if __name__ == '__main__':
    camera_main()


