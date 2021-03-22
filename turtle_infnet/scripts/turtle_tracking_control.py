#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#############################################
#                                           #
# Node to control the simulator TurtleSim.  #
# This node controls a pursuiting system    #
# sending controls for a turtle to tracking #
# a master.                                 #
#                                           #
# Author: Adalberto Oliveira                #
# Autonomous Vehicle - Infnet	            #
# Version: 1.0                              #
# Date: 02-07-2021                          #
#                                           #
#############################################


# importing libraries
import rospy, time, sys, math, angles
import control_lib as ctl
import numpy as np
from geometry_msgs.msg import Pose2D, Twist
from turtlesim.msg import Pose
from std_msgs.msg import Bool, Int32

############ WORK FUNCTIONS ##################


def callback_master_odom(msg):
    """
    This function receives the robot position and saves it
    in a global variable Pose2D
    """

    global master_pose

    master_pose.x = round(msg.x,4)
    master_pose.y = round(msg.y,4)
    master_pose.theta = round(msg.theta,4)

def callback_donkey_odom(msg):
    """
    This function receives the robot position and saves it
    in a global variable Pose2D
    """

    global donkey_pose

    donkey_pose.x = round(msg.x,4)
    donkey_pose.y = round(msg.y,4)
    donkey_pose.theta = round(msg.theta,4)



def traking_control():
    """
    This function is called from the main conde and calls 
    all work methods of fucntions into the codde.
    """

    # Global variables
    global master_pose
    global donkey_pose  
    global delta_tracking 
    global gains
    global error_int

    # Initializing ros node
    rospy.init_node('turtle_control', anonymous=True)    # node name
    
    # Subscribers
    master_odom_sub = rospy.Subscriber('master_pose', Pose, callback_master_odom)    # receives thr robot odometry
    donkey_odom_sub = rospy.Subscriber('donkey_pose', Pose, callback_donkey_odom)    # receives thr robot odometry
    
    # Publishers
    cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)  # send control signals

    # control rate
    rate = rospy.Rate(15)   # run the node at 15Hz
        
    # main loop
    while not rospy.is_shutdown():

        # Computing the control signal
        control_signal, errors = ctl.tracking_control(donkey_pose, 
                                 master_pose, 
                                 delta_tracking,
                                 gains,
                                 error_int,
                                 max_vel)
        error_int = errors[0]
        print('Control signal',errors)
    
        # Publishin the control to the robot
        cmd_vel.publish(control_signal)

        rate.sleep()
    



############ MAIN CODE #######################
# initializing Global variables
master_pose = Pose2D()  # master odometry
donkey_pose = Pose2D()  # donkey odometry

max_lin_vel = 4
max_ang_vel = 2

max_vel = [max_lin_vel, max_ang_vel]

K_v = float(sys.argv[1])    # Control gain for linear velocity
K_int = float(sys.argv[2])  # Integral control gain
K_omega = float(sys.argv[3])    # Control gain for angular velocity
delta_tracking = float(sys.argv[4]) # Distance from master

gains = [K_v, K_int, K_omega]
error_int = 0
#lim_theta = float(sys.argv[3]) # limite velue for theta (from launch file) 

print('#############################################')
print('#                                           #')
print('# Node to control the simulator TurtleSim.  #')
print('# This node receives a goal coordinate from #')
print('# topc /goal and controls the turtle.       #')
print('#                                           #')
print('# Author: Adalberto Oliveira                #')
print('# Autonomous Vehicle - Infnet               #')
print('# Version: 1.0                              #')
print('# Date: 02-03-2021                          #')
print('#                                           #')
print('#############################################')



if __name__ == '__main__':
    traking_control()
    '''
    try:
        traking_control()
    except:
        print('Node ended.')
    '''
