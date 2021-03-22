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
import rospy, time, sys, math, angles, control_lib, tf
import tf2_ros as tf2
import numpy as np
from geometry_msgs.msg import Pose2D, Twist
from turtlesim.msg import Pose
from std_msgs.msg import Bool, Int32
from nav_msgs.msg import Odometry

############ WORK FUNCTIONS ##################

def callback_goal(msg):
    """
    This function receives the goal and saves it 
    in a globa variable goal
    """

    global goal

    goal = msg

def callback_odom(msg):
   
    global robot_pose
    
    robot_pose.x = round(msg.pose.pose.position.x, 2)
    robot_pose.y = round(msg.pose.pose.position.y, 2)
    
    q = [msg.pose.pose.orientation.x, 
        msg.pose.pose.orientation.y, 
        msg.pose.pose.orientation.z, 
        msg.pose.pose.orientation.w]

    euler = tf.transformations.euler_from_quaternion(q)
    
    theta = round(euler[2],2)
    
    robot_pose.theta = theta

def callback_control_type(msg):
    """
    This function receives the type of controller:
    0: Cartesian control
    1: Polar control
    """

    global ctrl_type 

    ctrl_type = msg.data

def control_robot():
    """
    This function is called from the main conde and calls 
    all work methods of fucntions into the codde.
    """

    # Global variables
    global lim_x
    global lim_y
    global lim_theta
    global goal
    global robot_pose
    global K_v
    global K_omega
    global ctrl_type

    # Initializing ros node
    rospy.init_node('turtle_control', anonymous=True)    # node name
    
    # Subscribers
    goal_sub = rospy.Subscriber('goal',Pose2D, callback_goal)   # receives the goal coordinates
    odom_sub = rospy.Subscriber('odom', Odometry, callback_odom)    # receives thr robot odometry
    control_type = rospy.Subscriber('control_type',Int32,callback_control_type) #receives c.t.

    # Publishers
    cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)  # send control signals

    # control rate
    rate = rospy.Rate(15)   # run the node at 15H

    # main loop
    while not rospy.is_shutdown():

        # Computing the control signal
        #control_signal = control()
        
        # Selecting the controller
        if ctrl_type == 1:
            print('\nPolar Control')
            control_signal = control_lib.polar_control(robot_pose,goal,K_rho,K_alpha, K_beta)
        else:
            print('Cartesian Control')
            control_signal = control_lib.cartesian_control(robot_pose,goal,K_v,K_omega)
        

        # Publishin the control to the robot
        cmd_vel.publish(control_signal)
        print 'Robot Pose:',robot_pose
        print 'Goal: ', goal
        '''
        print('\nGoal:\n',goal,
              '\nPosition:\n',robot_pose,
              '\nControl:',control_signal.linear.x,control_signal.angular.z)
        '''
        rate.sleep()
    



############ MAIN CODE #######################
# initializing Global variables
goal = Pose2D()
robot_pose = Pose2D()
goal.x = 0
goal.y = 0
goal.theta = 0
ctrl_type = 0

K_v = float(sys.argv[1])   # Control gain for linear velocity
K_omega = float(sys.argv[2])   # Control gain for angular velocity
K_rho = float(sys.argv[3]) 
K_alpha = float(sys.argv[4]) 
K_beta = float(sys.argv[5]) 

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
    control_robot()
    '''
    try:
        control_robot()
    except:
        print('Node ended.')
    '''
