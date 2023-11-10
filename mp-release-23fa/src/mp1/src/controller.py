import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from scipy.integrate import ode
from std_msgs.msg import Float32MultiArray

import math
from util import euler_to_quaternion, quaternion_to_euler
import matplotlib.pyplot as plt

import time

def func1(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1]
    curr_theta = vars[2]

    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]


class vehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = True
        self.accelerations = []
        self.x = []
        self.y = []


    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp


    # Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    def extract_vehicle_info(self, currentPose):

        pos_x, pos_y, vel, yaw = 0, 0, 0, 0

        # Extract x and y pos, calculate velocity and yaw
        pos_x = currentPose.pose.position.x
        pos_y = currentPose.pose.position.y
        vel = math.sqrt(currentPose.twist.linear.x**2+currentPose.twist.linear.y**2)
        euler_angles = quaternion_to_euler(currentPose.pose.orientation.x, currentPose.pose.orientation.y, currentPose.pose.orientation.z, currentPose.pose.orientation.w)
        yaw = euler_angles[2]
                

        return pos_x, pos_y, vel, yaw # note that yaw is in radian

    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    import numpy as np

    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):
        global curve
        ####################### TODO: Your TASK 2 code starts Here #######################
        if len(future_unreached_waypoints) < 2: # Make sure there's enough points to determine velocity
            target_velocity = 1
        
        else:
            p1 = future_unreached_waypoints[0]
            p2 = future_unreached_waypoints[1]

            # Calculate difference between angle of waypoints and current yaw to determine if we are on a curve
            d_theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) - curr_yaw
            d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))

            # Difference in angle(radians) to signify a curve (tunable parameter)
            threshold = 0.12 

            if abs(d_theta) > threshold: # Check if we are on a curve and decrease target
                target_velocity = 0.5
                curve = True
            else:
                target_velocity = 1
                curve = False
            
            # Limit acceleration at start
            # if curr_vel < 5:
            #     target_velocity = 6
            # elif curr_vel < 8:
            #     target_velocity = 9

        ####################### TODO: Your TASK 2 code ends Here #######################
        return target_velocity



    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, future_unreached_waypoints):

        ####################### TODO: Your TASK 3 code starts Here #######################
        global curve

        # If we are on a curve, lookahead distance is close for sharp turns
        # if curve:
        #     lookahead = future_unreached_waypoints[0]
        
        # # If we aren't on a curve, use a deeper point to prevent unnecessary zig-zag
        # else:
        #     try: # Make sure there's enough waypoints
        #         lookahead = future_unreached_waypoints[1]
        #     except IndexError:
        #         lookahead = future_unreached_waypoints[0]
        
        lookahead = future_unreached_waypoints[0]

        curr_x = 250
        curr_y = 480
        curr_yaw = 0
        # Distance between lookahead point and current position        
        ld = math.sqrt((lookahead[0] - curr_x)**2 + (lookahead[1] - curr_y)**2)

        # Find angle car is rotated away from lookahead
        alpha = np.arctan2(lookahead[1] - curr_y, lookahead[0] - curr_x) - curr_yaw

        # Pure pursuit equation
        target_steering = 30 * np.arctan(2*self.L*np.sin(alpha) / ld)

        print(target_steering* 180/math.pi)
        
        self.x.append(curr_x)
        self.y.append(curr_y)


        ####################### TODO: Your TASK 3 code starts Here #######################
        return target_steering

    def execute_origin(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz

        self.accelerations.append(acceleration)

        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)


        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)

        self.prev_vel = curr_vel

    def execute(self, currentPose, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)
        curr_x = 250
        curr_y = 480
        curr_yaw = 0
        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz

        self.accelerations.append(acceleration)

        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, future_unreached_waypoints)


        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)

        self.prev_vel = curr_vel


    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        # print(self.x)
        # print(self.y)

        self.controlPub.publish(newAckermannCmd)
