#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importing libraries
import rospy, time
from geometry_msgs.msg import  PointStamped
import tf2_ros as tf2
import tf2_geometry_msgs


# Transformation variables
goal_stamped = PointStamped()
img_goal = PointStamped()


if __name__ == '__main__':

    # Initializing ros node
    rospy.init_node('turtle_control', anonymous=True)    # node name
    
    # creating transformation engine
    tfBuffer = tf2.Buffer()
    listener = tf2.TransformListener(tfBuffer)

    # control rate
    #rate = rospy.Rate(30)   # run the node at 15H

    # getting points in the camera frame
    x = float(input('Coordinate x in the camera frame:'))
    y = float(input('Coordinate y in the camera frame:'))
    z = float(input('Coordinate z in the camera frame:'))

    # filling the object 
    img_goal.header.stamp = rospy.Time()    # getting time stamp
    img_goal.header.frame_id = 'camera_link'
    img_goal.point.x = x
    img_goal.point.y = y
    img_goal.point.z = z

    goal_stamped = tfBuffer.transform(img_goal, "odom")

    print('*** Transformation from Camera to Odom Frame:\
    \nCamera Point:',img_goal,
    '\nOdom Point:',goal_stamped)   

