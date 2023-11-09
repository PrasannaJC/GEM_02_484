import rospy
import numpy as np
import argparse

from gazebo_msgs.msg import  ModelState
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

def callback_function(msg):
    data = msg.data
    waypoints = [[data[0], data[1]] , [data[2], data[3]]] 
    run_model(waypoints)

def run_model(waypoints):
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

    while not rospy.is_shutdown():
        rate.sleep()  

        currState =  controller.getModelState()
        if not currState.success:
            continue

        controller.execute(currState, waypoints)

def main():
    rospy.init_node('main_node')
    rospy.Subscriber('chatter', Float32MultiArray, callback_function)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutting down")



# ==============================================================
# global waypoints

# def callback_function(msg):
#     data = msg.data
#     waypoints = [[data[0], data[1]] , [data[2], data[3]]] 
#     # print(data[0], data[1], data[2], data[3])
#     print(waypoints)

# def subscriber_node():
#     global waypoints
#     rospy.init_node('subscriber_node')
#     msg = rospy.Subscriber('chatter', Float32MultiArray, callback_function)
#     # print(waypoints.layout.data)
#     rospy.spin()


# def run_model():
#     rospy.init_node("model_dynamics")
#     controller = vehicleController()

#     waypoints = WayPoints()
#     pos_list = waypoints.getWayPoints()
#     pos_idx = 1

#     target_x, target_y = pos_list[pos_idx]

#     def shutdown():
#         """Stop the car when this ROS node shuts down"""
#         controller.stop()
#         rospy.loginfo("Stop the car")

#     rospy.on_shutdown(shutdown)

#     rate = rospy.Rate(100)  # 100 Hz
#     rospy.sleep(0.0)
#     start_time = rospy.Time.now()
#     prev_wp_time = start_time

    

#     while not rospy.is_shutdown():
#         rate.sleep()  # Wait a while before trying to get a new state

#         # Get the current position and orientation of the vehicle
#         currState =  controller.getModelState()
        
#         if not currState.success:
#             continue

#         # Compute relative position between vehicle and waypoints
#         # distToTargetX = abs(target_x - currState.pose.position.x)
#         # distToTargetY = abs(target_y - currState.pose.position.y)
#         distToTargetX = target_x
#         distToTargetY = target_y

#         cur_time = rospy.Time.now()

#         if (distToTargetX < 2 and distToTargetY < 2):
#             # If the vehicle is close to the waypoint, move to the next waypoint
#             prev_pos_idx = pos_idx
#             pos_idx = pos_idx+1

#             if pos_idx == len(pos_list): #Reached all the waypoints
#                 print("Reached all the waypoints")
#                 total_time = (cur_time - start_time).to_sec()
#                 print(total_time)
#                 return True, pos_idx, total_time

#             target_x, target_y = pos_list[pos_idx]

#             time_taken = (cur_time- prev_wp_time).to_sec()
#             prev_wp_time = cur_time
#             print(f"Time Taken: {round(time_taken, 2)}", "reached",pos_list[prev_pos_idx][0],pos_list[prev_pos_idx][1],"next",pos_list[pos_idx][0],pos_list[pos_idx][1])

#         # controller.execute(currState, [target_x, target_y], pos_list[pos_idx:])
#         # controller.execute(currState, [target_x, target_y], lineFit.create_waypoints(self))
#         controller.execute(currState, [target_x, target_y], waypoints)

# if __name__ == "__main__":
#     try:
#         subscriber_node()
#         status, num_waypoints, time_taken = run_model()

#     except rospy.exceptions.ROSInterruptException:
#         rospy.loginfo("Shutting down")