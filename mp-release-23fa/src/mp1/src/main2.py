import rospy
import numpy as np
import argparse

from gazebo_msgs.msg import  ModelState
from controller import vehicleController
import time
from waypoint_list import WayPoints
from util import euler_to_quaternion, quaternion_to_euler
# from line_fit import lineFit 
from line_fit2 import create_waypoints 
from std_msgs.msg import Float32MultiArray

#---------------------
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
#---------------------

# from studentVision import lanenet_detector

global waypoints

def callback_function(msg):
    global waypoints
    data = msg.data
    waypoints = [[data[0], data[1]] , [data[2], data[3]]] 

def subscriber_node():
    global waypoints
    rospy.init_node('subscriber_node')
    msg = rospy.Subscriber('chatter', Float32MultiArray, run_model)
    rospy.spin()


def run_model(msg):
    print('run model ==================================')
    global waypoints

    rospy.init_node("model_dynamics")
    controller = vehicleController()

    def shutdown():
        """Stop the car when this ROS node shuts down"""
        controller.stop()
        rospy.loginfo("Stop the car")

    rospy.on_shutdown(shutdown)

    rate = rospy.Rate(100)  # 100 Hz
    rospy.sleep(0.0)
    start_time = rospy.Time.now()
    prev_wp_time = start_time

    data = msg.data
    waypoints = [[data[0], data[1]] , [data[2], data[3]]] 

    while not rospy.is_shutdown():
        rate.sleep()  # Wait a while before trying to get a new state

        # Get the current position and orientation of the vehicle
        currState =  controller.getModelState()
        # img = laneDet.img_callback()
        if not currState.success:
            continue

        controller.execute(currState, waypoints)     # main2
    

if __name__ == "__main__":
    try:
        subscriber_node()
        run_model(msg)
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutting down")