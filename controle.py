#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, time, sys, math, angles
from geometry_msgs.msg import Pose2D, Twist
from turtlesim.msg import Pose


def callback_robot_pose(msg):
    global robot_odom
 

    robot_odom.x = msg.x
    robot_odom.y = msg.y
    robot_odom.theta = msg.theta

def callback_robot_goal(msg):
    global goal

    goal = msg

def robot_command(robot_odom,goal,gain):
    
    # recuperando as coordenadas do robo
    x = robot_odom.x
    y = robot_odom.y
    theta = robot_odom.theta

    # recuperand o goal
    x_d = goal.x
    y_d = goal.y
    theta_d = goal.theta

    # recuperando os ganhos
    K_v = gain[0]
    K_omega = gain[1]

    # definindo os erros
    delta_x = x_d - x
    delta_y = y_d - y

    erro_p = round(math.sqrt(delta_x**2 + delta_y**2),3)
    
    heading = round(math.atan2(delta_y,delta_x),3)

    erro_theta = angles.shortest_angular_distance(theta,heading)

    v = K_v*erro_p
    omega = K_omega*erro_theta
    
    robot_vel = Twist()
    robot_vel.linear.x = v
    robot_vel.angular.z = omega

    return robot_vel


def main_control():
    global robot_odom
    global gain
    global goal

    rospy.init_node('turtle_control', anonymous=True)    # node name
    robot_pose = rospy.Subscriber('/turtle1/pose',Pose,callback_robot_pose)
    robot_goal = rospy.Subscriber('/goal',Pose2D,callback_robot_goal)
    pub_cmd_vel = rospy.Publisher('/turtle1/cmd_vel',Twist,queue_size=10)
    

    rate = rospy.Rate(15)

    cmd_vel = Twist()
    
    while not rospy.is_shutdown():


        cmd_vel = robot_command(robot_odom,goal,gain)
        print(cmd_vel)

        pub_cmd_vel.publish(cmd_vel)

        rate.sleep()

############### Main code ################
robot_odom = Pose2D()
goal = Pose2D()
k_v = 0.5
k_w = 0.8
gain = [k_v,k_w]

if __name__ == '__main__':
    main_control()
