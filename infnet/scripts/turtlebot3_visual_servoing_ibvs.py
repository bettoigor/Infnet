#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#############################################
#                                           #
# Node to control the Turtlebot3 simulator. #
# This node receives image poit and sends   #
# control signal to the cmd_vel topic.      #
#                                           #
# Author: Adalberto Oliveira                #
# Autonomous Vehicle - Infnet	            #
# Version: 1.2                              #
# Date: 13 mar 2021                         #
#                                           #
#############################################


# importing libraries
import rospy, time, sys, math, control_lib, tf
import numpy as np
from geometry_msgs.msg import Pose2D, Twist
from turtlesim.msg import Pose
from std_msgs.msg import Bool, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
import image_geometry

############ WORK FUNCTIONS ##################

def callback_camera_info(msg):

    global model
    global camera_matrix


    model.fromCameraInfo(msg)
    
    K = np.array(msg.K).reshape([3, 3])
    f = K[0][0]
    u0 = K[0][2]
    v0 = K[1][2]

    camera_matrix[0] = f
    camera_matrix[1] = u0
    camera_matrix[2] = v0


def callback_img_point(msg):
    """
    This function receives the goal and saves it 
    in a globa variable goal
    """

    global camera_height
    global image_point
    global mask_is_true

    # recovering point
    u = msg.x
    v = msg.y
    base_point = [u, v]
    mask_is_true = msg.theta
    distance = 0

    try:
        # finding distance to the point 
        pixel_rectified = model.rectifyPoint(base_point)
        line = model.projectPixelTo3dRay(pixel_rectified)
        th = math.atan2(line[2],line[1])
        distance = math.tan(th) * camera_height

        image_point.x = u
        image_point.y = v
        image_point.theta = distance

    except:
        pass


def callback_odom(msg):
   
    global robot_pose
    
    robot_pose.x = round(msg.pose.pose.position.x, 3)
    robot_pose.y = round(msg.pose.pose.position.y, 3)
    
    q = [msg.pose.pose.orientation.x, 
        msg.pose.pose.orientation.y, 
        msg.pose.pose.orientation.z, 
        msg.pose.pose.orientation.w]

    euler = tf.transformations.euler_from_quaternion(q)
    
    theta = round(euler[2],3)
    
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
    global img_goal
    global image_point
    global robot_pose
    global gains_cart
    global ctrl_type
    global max_lin
    global max_ang
    global goal
    global camera_matrix
    global mask_is_true

    # Initializing ros node
    rospy.init_node('turtle_control', anonymous=True)    # node name
    
    # Subscribers
    rospy.Subscriber('img_point',Pose2D, callback_img_point)   # receives the goal coordinates
    rospy.Subscriber('odom', Odometry, callback_odom)    # receives thr robot odometry
    rospy.Subscriber('control_type',Int32,callback_control_type) #receives c.t.
    rospy.Subscriber('camera_info',CameraInfo, callback_camera_info)   # receives the goal coordinates

    # Publishers
    cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)  # send control signals

    # control rate
    rate = rospy.Rate(30)   # run the node at 15H

    # main loop
    while not rospy.is_shutdown():

        # Computing the control signal
        control_signal = Twist()
        
        # calling IBVS
        try:
            if mask_is_true:
                control_signal = control_lib.ibvs(img_goal, image_point, camera_matrix, gains_cart,vel_lim)
                
            else:
                control_signal = Twist()
                control_signal.linear.x = 0.
                control_signal.angular.z = 0.5
        except:
            pass

        #print control_signal
        cmd_vel.publish(control_signal)

        print('\rDistance to the target:',image_point.theta, end='\r')

        rate.sleep()
    



############ MAIN CODE #######################
# initializing Global variables
# Readin from launch
K_eu = float(sys.argv[1])   # Control gain for linear velocity
K_ev = float(sys.argv[2])   # Control gain for angular velocity
X_goal = float(sys.argv[3])
Y_goal = float(sys.argv[4])
max_lin = float(sys.argv[5])
max_ang = float(sys.argv[6])
ctrl_type = float(sys.argv[7])
camera_height = float(sys.argv[8])

# Inner values
robot_pose = Pose2D()
image_point = Pose2D()
gains_cart = [K_eu, K_ev]
img_goal = Pose2D()
img_goal.x = X_goal
img_goal.y = Y_goal
camera_matrix = np.zeros((3,1))
vel_lim = [max_lin, max_ang]
mask_is_true = False

# creating a camera model
model = image_geometry.PinholeCameraModel()


print('#############################################',
      '\n#                                           #',
      '\n# Node to control the Turtlebot3 simulator. #',
      '\n# This node receives image poit and sends   #',
      '\n# control signal to the cmd_vel topic.      #',
      '\n#                                           #',
      '\n# Author: Adalberto Oliveira                #',
      '\n# Autonomous Vehicle - Infnet               #',
      '\n# Version: 1.2                              #',
      '\n# Date: 13 mar 2021                         #',
      '\n#                                           #',
      '\n#############################################')




if __name__ == '__main__':
    control_robot()
    '''
    try:
        control_robot()
    except:
        print('Node ended.')
    '''
