#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#############################################
#                                           #
# Node for world coordinate transform       #
# for Turtlebot3 Gazebo                     #
#                                           #
# Author: Adalberto Oliveira                #
# Autonomous Vehicle - Infnet               #
# Version: 1.0                              #
# Date: 21 mar 2021                         #
#                                           #
#############################################

import rospy, tf, time, angles, math, sys
import tf2_ros as tf2
from geometry_msgs.msg import Twist, Pose2D, PointStamped 
from nav_msgs.msg import Odometry


def world_broadcaster(msg):

    time_stamp = msg.header.stamp #rospy.Time.now()
    
    turtlebot3_base_link_quat = tf.transformations.quaternion_from_euler(0,0,0)
    turtlebot3_base_link = tf.TransformBroadcaster()
    turtlebot3_base_link.sendTransform((0,0,0),
                                        turtlebot3_base_link_quat,
                                        time_stamp,
                                        "base_link",
                                        "base_footprint")


    turtlebot3_camera_link_quat = tf.transformations.quaternion_from_euler(0,0,0)	
    turtlebot3_camera_link = tf.TransformBroadcaster()
    turtlebot3_camera_link.sendTransform((0.073, -0.011, 0.084),
                                        turtlebot3_camera_link_quat,
                                        time_stamp,
                                        "camera_link",
                                        "base_link")


def init_world():
	rospy.init_node('tf_broadcaster', anonymous=True) #nome do nó
	rospy.Subscriber('odom', 
					 Odometry, 
					 world_broadcaster)

	rospy.spin()
	

if __name__ == '__main__':
    print('Broadcasting transformations...')
    init_world()
