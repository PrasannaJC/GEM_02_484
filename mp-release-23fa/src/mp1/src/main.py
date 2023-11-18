import rospy
import numpy as np
import argparse
from controller import vehicleController
import time
from waypoint_list import WayPoints
from util import euler_to_quaternion, quaternion_to_euler
# from line_fit import lineFit 
from line_fit import create_waypoints 
from std_msgs.msg import Float32MultiArray

#---------------------
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
#---------------------

import rospy
from std_msgs.msg import Float32MultiArray

class ControllerNode:
    def __init__(self):
        self.waypoints = None
        rospy.init_node('controller_node')
        rospy.Subscriber('chatter', Float32MultiArray, self.callback_function)
        self.controller = vehicleController()

    def callback_function(self, msg):
        data = msg.data
        self.waypoints = [[data[0], data[1]], [data[2], data[3]]]

    def run_model(self):
        def shutdown():
            """Stop the car when this ROS node shuts down"""
            self.controller.stop()
            rospy.loginfo("Stop the car")

        rospy.on_shutdown(shutdown)

        rate = rospy.Rate(100)  # 100 Hz
        rospy.sleep(0.0)
        start_time = rospy.Time.now()
        prev_wp_time = start_time

        while not rospy.is_shutdown():
            rate.sleep()  
            
            if self.waypoints is not None:
                self.controller.execute(self.waypoints)

    def start(self):
        self.run_model()

if __name__ == "__main__":
    try:
        node = ControllerNode()
        node.start()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutting down")

