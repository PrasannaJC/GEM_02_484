import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
from util import euler_to_quaternion, quaternion_to_euler
import time
import matplotlib.pyplot as plt

class vehicleController():
    
    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = True

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


    ## Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    # and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    def extract_vehicle_info(self, currentPose):

    ####################### TODO: Your TASK 1 code starts Here #######################
        pos_x, pos_y, vel, yaw = 0, 0, 0, 0
        pos_x = currentPose.pose.position.x
        pos_y = currentPose.pose.position.y
        x_orientation = currentPose.pose.orientation.x
        y_orientation = currentPose.pose.orientation.y
        z_orientation = currentPose.pose.orientation.z
        w_orientation = currentPose.pose.orientation.w
        quat = quaternion_to_euler(x_orientation,y_orientation,z_orientation,w_orientation)
        yaw = quat[2]
        vel = currentPose.twist.linear
        #print(vel)
        ####################### TODO: Your Task 1 code ends Here #######################
        return pos_x, pos_y, vel, yaw # note that yaw is in radian
   
    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):
        #target_velocity=12
    ####################### TODO: Your TASK 2 code starts Here #######################
        straight_speed = 12  
        turn_speed = 8      
        #print(curr_vel.x)
        delta_x = future_unreached_waypoints[0][0] - curr_x
        delta_y = future_unreached_waypoints[0][1] - curr_y

        distance = math.sqrt(delta_x**2 + delta_y**2)
        angular_velocity = (delta_x * math.sin(curr_yaw) - delta_y * math.cos(curr_yaw)) / distance
        curr_vel_mag = math.sqrt(curr_vel.x**2+curr_vel.y**2+curr_vel.z**2)
        curvature = abs(angular_velocity)/curr_vel_mag

        if curvature > 0.005:
           target_velocity = turn_speed
        else:
            target_velocity=straight_speed
        #acceleration_value = (target_velocity**2-curr_vel**2)/(2*distance)
        #print(acceleration_value)
        #print(curvature)
        #target_velocity = max(target_velocity, curr_vel)
        ####################### TODO: Your TASK 2 code ends Here ######################
        #print(future_unreached_waypoints[0][0], future_unreached_waypoints[0][1],target_velocity,delta_x, delta_y)
        return target_velocity

    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):

        ####################### TODO: Your TASK 3 code starts Here #######################
        #target_steering = 0
        next_waypoint_x = future_unreached_waypoints[0][0]
        next_waypoint_y= future_unreached_waypoints[0][1]
        next_desirable_wp_x=(curr_x+next_waypoint_x)/2
        next_desirable_wp_y=(curr_y+next_waypoint_y)/2
        #next_desirable_wp_x=next_waypoint_x
        #next_desirable_wp_y=next_waypoint_y
        delta_x= curr_x - next_desirable_wp_x
        delta_y= curr_y - next_desirable_wp_y
        ld= math.sqrt(delta_x**2+delta_y**2)
        
        closest_waypoint = (next_desirable_wp_x,next_desirable_wp_y)
        total_yaw = math.atan2(closest_waypoint[1] - curr_y, closest_waypoint[0] - curr_x) 
        alpha = total_yaw-curr_yaw
        
        L = 1.75 
        target_steering = math.atan2(2 * L * math.sin(alpha), ld)
         

        ####################### TODO: Your TASK 3 code starts Here #######################
        return target_steering


    def execute(self, currentPose, target_point, future_unreached_waypoints):
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
            if isinstance(self.prev_vel, int):
                self.prev_vel= curr_vel
                time=0
                count =0
            else:
                acceleration_x = (curr_vel.x- self.prev_vel.x) * 100 # Since we are running in 100Hz
                acceleration_y = (curr_vel.y-self.prev_vel.y) * 100
                acceleration_mag = math.sqrt(acceleration_x**2+acceleration_y**2)
                print(acceleration_mag)
                #if acceleration_mag>5:
                    
                    #count=count+1
                
                #time=time+1/100
                #plt.plot(acceleration_mag)
                #plt.title('Acceleration mag vs time')
                self.prev_vel=curr_vel
                
        #print(count)  


        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)


        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)


    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.controlPub.publish(newAckermannCmd)