#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#############################################
#                                           #
# Node to send goal coordinates. This node  #
# will not receive any message from other   #
# nodes.                                    #
#                                           #
# Author: Adalberto Oliveira                #
# Autonomous Vehicle - Infnet	            #
# Version: 1.0                              #
# Date: 02-03-2021                          #
#                                           #
#############################################


import rospy, time, sys, math, angles
import numpy as np
from geometry_msgs.msg import Pose2D, Twist
from turtlesim.msg import Pose



def cartesian_control(robot_pose,goal,K_v,K_omega):
    """
    This function computes the control signal to guides the 
    robot to the desired goal. It's based on the Cartesian
    Control Algorithm
    """

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


def polar_control(robot_pose,goal,K_rho,K_alpha, K_beta):
    """
    This function computes the control signal to guides the 
    robot to the desired goal. It's based on the Cartesian
    Control Algorithm
    """

    # recovering the variables
    x = robot_pose.x
    y = robot_pose.y
    theta = robot_pose.theta

    x_d = goal.x
    y_d = goal.y
    theta_d = goal.theta

    # Computing the position error
    delta_x = x_d - x
    delta_y = y_d - y
    heading = math.atan2(delta_y,delta_x)

    # Creating the new variables
    rho  = round(math.sqrt(delta_x**2 + delta_y**2),3)
    alpha = round(angles.shortest_angular_distance(theta,heading),3)
    beta = round(-theta - alpha + theta_d,3)
    #beta =  round(angles.shortest_angular_distance(theta_d, theta),3)

    # Computing control signals
    v = round(K_rho*rho,3)
    
    omega = round(K_alpha*alpha + K_beta*beta,4)

    print('Rho:',rho,
    		'\nAlpha:',alpha,
            '\nBeta:',beta)

    u = Twist()

    u.linear.x = v
    u.angular.z = omega

    return u


def tracking_control(donkey_pose, master_pose, delta_tracking, gains, error_int, max_vel):
    """
    This function computes the control signal to guides the 
    robot to the desired goal. It's based on the Cartesian
    Control Algorithm
    """
    
    # Recovering control gains
    K_v = gains[0]
    K_int = gains[1]
    K_omega = gains[2]

    # Recovering velocitie limits
    max_lin_vel = max_vel[0]
    max_ang_vel = max_vel[1]

    # Computing the position error
    error_x = master_pose.x - donkey_pose.x
    error_y = master_pose.y - donkey_pose.y
    error_lin = round(math.sqrt(error_x**2 + error_y**2)-delta_tracking,3)
    error_int += error_lin*0.066

    # Computing the heading
    heading = math.atan2(error_y,error_x)
    error_th = round(angles.shortest_angular_distance(donkey_pose.theta, heading),3)
    
    errors = [error_int, error_lin, error_th]

    # Computing control signals
    v = K_v*error_lin + K_int*error_int
    v = np.sign(v)*max_lin_vel if v > max_lin_vel else v

    omega = K_omega*error_th
    omega = np.sign(omega)*max_ang_vel if omega > max_ang_vel else omega

    u = Twist()

    u.linear.x = v
    u.angular.z = omega

    return u, errors
