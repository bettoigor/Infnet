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


# importing libraries
import rospy, time, sys, math
import numpy as np
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Bool, Int32

############ WORK FUNCTIONS ##################

def sendGoal():
    """
    This function is called from the main conde and calls 
    all work methods of fucntions into the codde.
    """

    # Global variables
    global lim_x
    global lim_y
    global lim_theta

    # Initializing ros node
    rospy.init_node('send_goal', anonymous=True)    # node name
    goal_pub = rospy.Publisher('goal',Pose2D, queue_size=10, latch=True)
    control_type_pub = rospy.Publisher('control_type',Int32, queue_size=10, latch=True)
    
    rate = rospy.Rate(15)   # run the node at 15Hz
 
    # Creating the object Pose2D
    goal = Pose2D() # destination
    control_type = Int32()
    # main loop
    while not rospy.is_shutdown():


        # Prompts the user for destination goal
        x = input('\nPlease enter destinatin coord \'x\':')       # receiving x
        y = input('Please enter destinatin coord \'y\':')       # receiving y
        theta = input('Please enter destinatin coord \'theta\':')  # receiving theta
        ctrl_type =input('Controlle type (0: Cartesia; 1: Polar):')  # controller type
        
        # Converting from String to float and check integrity
        try:
            x = float(x)
            x = lim_x if x > abs(lim_x) else abs(x)
        except:
            x = 0

        try:
            y = float(y)
            y = lim_y if y > abs(lim_y) else abs(y)
        except:
            y = 0
        try:
            theta = float(theta)
            theta = np.sign(theta)*lim_theta if abs(theta) > lim_theta else theta
        except:
            theta = 0

        try:
            ctrl_type = int(ctrl_type)
            ctrl_type = ctrl_type if (ctrl_type==0 or ctrl_type==1) else 0
        except:
            ctrl_type = 0


        # filling in the objetc Pose2D
        goal.x = x
        goal.y = y
        goal.theta = theta

        # filling in the object Int32
        control_type.data = ctrl_type

        # Publising goal
        goal_pub.publish(goal)
        control_type_pub.publish(ctrl_type)

        print('\nSending Goal:\n', goal,
              '\nController:',ctrl_type)
        
        rate.sleep()




############ MAIN CODE #######################
# initializing Global variables
lim_x = float(sys.argv[1])   # limite velue for x (from launch file)
lim_y = float(sys.argv[2])   # limite velue for y (from launch file)
lim_theta = float(sys.argv[3]) # limite velue for theta (from launch file) 

print('#############################################')
print('#                                           #')
print('# Node to send goal coordinates. This node  #')
print('# will not receive any message from other   #')
print('# nodes.                                    #')
print('#                                           #')
print('# Author: Adalberto Oliveira                #')
print('# Autonomous Vehicle - Infnet               #')
print('# Version: 1.0                              #')
print('# Date: 02-03-2021                          #')
print('#                                           #')
print('#############################################')


print ('System Limits\nx:',lim_x,
        '\ny:',lim_y,
        '\ntheta:',lim_theta)

if __name__ == '__main__':
    sendGoal()
    '''
    try:
        sendGoal()
    except:
        print('Node ended.')
    '''
