#!/usr/bin/env python3
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
import rospy, time, sys, math, angles, control_lib
import numpy as np
from geometry_msgs.msg import Pose2D, Twist
from turtlesim.msg import Pose
from std_msgs.msg import Bool, Int32

############ WORK FUNCTIONS ##################

def callback_goal(msg):
    """
    This function receives the goal and saves it 
    in a globa variable goal
    """

    global goal

    goal = msg



def callback_odom(msg):
    """
    This function receives the robot position and saves it
    in a global variable Pose2D
    """

    global robot_pose

    robot_pose.x = round(msg.x,4)
    robot_pose.y = round(msg.y,4)
    robot_pose.theta = round(msg.theta,4)

def callback_control_type(msg):
    """
    This function receives the type of controller:
    0: Cartesian control
    1: Polar control
    """

    global ctrl_type 

    ctrl_type = msg.data



def control():
    """
    This function computes the control signal to guides the 
    robot to the desired goal. It's based on the Cartesian
    Control Algorithm
    """
    global robot_pose
    global goal
    global K_v
    global K_omega

    # Computing the position error
    error_x = goal.x - robot_pose.x
    error_y = goal.y - robot_pose.y
    error_lin = round(math.sqrt(error_x**2 + error_y**2),3)
    v = K_v*error_lin

    # Computing the heading
    heading = math.atan2(error_y,error_x)
    error_th = round(angles.shortest_angular_distance(robot_pose.theta,heading),3)
    
    omega = K_omega*error_th

    print('Error lin:',error_lin,
            'Errol heading:',error_th)


    u = Twist()

    u.linear.x = v
    u.angular.z = omega

    return u

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
    odom_sub = rospy.Subscriber('pose', Pose, callback_odom)    # receives thr robot odometry
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

        print('\nGoal:\n',goal,
              '\nPosition:\n',robot_pose,
              '\nControl:',control_signal.linear.x,control_signal.angular.z)
        
        rate.sleep()
    



############ MAIN CODE #######################
# initializing Global variables
goal = Pose2D()
robot_pose = Pose2D()
goal.x = 5.5
goal.y = 5.5
goal.theta = 0
ctrl_type = 1

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
