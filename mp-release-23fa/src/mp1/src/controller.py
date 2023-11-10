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
        self.L = 65 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = True
        self.accelerations = []
        self.x = []
        self.y = []
        self.fix_x = 380
        self.fix_y = 480
        self.fix_yaw = np.pi/2

        

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
        global xp
        global cy
        # pos_x, pos_y, vel, yaw = 0, 0, 0, 0

        # Extract x and y pos, calculate velocity and yaw
        xp = currentPose.pose.position.x
        # pos_y = currentPose.pose.position.y
        vel = math.sqrt(currentPose.twist.linear.x**2+currentPose.twist.linear.y**2)
        euler_angles = quaternion_to_euler(currentPose.pose.orientation.x, currentPose.pose.orientation.y, currentPose.pose.orientation.z, currentPose.pose.orientation.w)
        cy = euler_angles[2]
                
        pos_x = self.fix_x
        pos_y = self.fix_y
        yaw = self.fix_yaw
        

        return pos_x, pos_y, vel, yaw, xp, cy # note that yaw is in radian

    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    import numpy as np



    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, future_unreached_waypoints):

        global curve
        global xp

        # If we are on a curve, lookahead distance is close for sharp turns
        # if curve:
        #     lookahead = future_unreached_waypoints[0]
        
        # # If we aren't on a curve, use a deeper point to prevent unnecessary zig-zag
        # else:
        #     try: # Make sure there's enough waypoints
        #         lookahead = future_unreached_waypoints[1]
        #     except IndexError:
        #         lookahead = future_unreached_waypoints[0]
        
        lookahead = future_unreached_waypoints[1]

        curr_x = self.fix_x
        curr_y = self.fix_y
        curr_yaw = self.fix_yaw
        
        # Distance between lookahead point and current position        
        ld = math.sqrt((lookahead[0] - curr_x)**2 + (lookahead[1] - curr_y)**2)

        # Find angle car is rotated away from lookahead
        alpha = np.arctan2( -lookahead[1] + curr_y, lookahead[0] - curr_x) - curr_yaw

        # Pure pursuit equation
        target_steering = np.arctan(2*self.L*np.sin(alpha) / ld)
        if abs(target_steering) > 0.1:
            curve = True
        else:
            curve = False
                    
        # print('alpha: ', alpha*180/np.pi, '   steering: ', target_steering* 180/np.pi)
        


        return target_steering, curve
    
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints, curve):

        # if len(future_unreached_waypoints) < 2: # Make sure there's enough points to determine velocity
        #     target_velocity = 3
        
        # else:
        #     p1 = future_unreached_waypoints[0]
        #     p2 = future_unreached_waypoints[1]

        #     # Calculate difference between angle of waypoints and current yaw to determine if we are on a curve
        #     d_theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) - curr_yaw
        #     d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))
            
        #     # Difference in angle(radians) to signify a curve (tunable parameter)
        #     threshold = 0.12 

        #     if abs(d_theta) > threshold: # Check if we are on a curve and decrease target
        #         target_velocity = 2
        #         curve = True
        #     else:
        #         target_velocity = 5
        #         curve = False
        
        if curve:
            target_velocity = 2
        else:
            target_velocity = 5
        print(target_velocity)
                    

        return target_velocity


    def execute(self, currentPose, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw, x_pos, yp = self.extract_vehicle_info(currentPose)
        curr_x = self.fix_x
        curr_y = self.fix_y
        curr_yaw = self.fix_yaw
        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz

        self.accelerations.append(acceleration)
        
        target_steering, curve = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, future_unreached_waypoints)
        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints, curve)


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

        self.controlPub.publish(newAckermannCmd)