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

# Python Headers
import os 
import csv
import scipy.signal as signal


# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64
# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


def func1(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1]
    curr_theta = vars[2]

    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]


class PID(object):

    def __init__(self, kp, ki, kd, wg=None):

        self.iterm  = 0
        self.last_t = None
        self.last_e = 0
        self.kp     = kp
        self.ki     = ki
        self.kd     = kd
        self.wg     = wg
        self.derror = 0

    def reset(self):
        self.iterm  = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e, fwd=0):

        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            de = (e - self.last_e) / (t - self.last_t)

        if abs(e - self.last_e) > 0.5:
            de = 0

        self.iterm += e * (t - self.last_t)

        # take care of integral winding-up
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg
            elif self.iterm < -self.wg:
                self.iterm = -self.wg

        self.last_e = e
        self.last_t = t
        self.derror = de

        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de

class OnlineFilter(object):

    def __init__(self, cutoff, fs, order):
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Get the filter coefficients 
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Initialize
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted

class vehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        # self.L = 65 # Wheelbase, can be get from gem_control.py
        self.L = 97
        self.log_acceleration = True
        self.accelerations = []
        self.x = []
        self.y = []
        self.fix_x = 640
        self.fix_y = 720
        self.fix_yaw = np.pi/2
        
        
        self.gem_enable = False
        self.pacmod_enable = False

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2  # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear = True
        self.accel_cmd.ignore = True

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0  # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 2.0  # radians/second
        
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0
        
        # PID controller for speed
        self.pid_speed = PID(0.5, 0.0, 0.1)  # Tune these parameters
        
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        # Publishers
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)

        # Commands
        self.accel_cmd = PacmodCmd()
        self.steer_cmd = PositionWithSpeed()


        

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
        f_angle = np.arctan(2*self.L*np.sin(alpha) / ld)
        if abs(f_angle) > 0.1:
            curve = True
        else:
            curve = False
                    
        # print('alpha: ', alpha*180/np.pi, '   steering: ', target_steering* 180/np.pi)

        f_angle = f_angle/np.pi*180
        
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0

        return steer_angle, curve
    
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints, curve):

        if curve:
            target_velocity = 2
        else:
            target_velocity = 5

        target_velocity = 20
        print(target_velocity)


        return target_velocity
    
    
    def speed_callback(self, msg):
        self.current_speed = round(msg.vehicle_speed, 3)  # Update with the correct attribute

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

        
        target_steering, curve = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, future_unreached_waypoints)
        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints, curve)


        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)
        
        current_time = rospy.get_time()
        filt_vel     = self.speed_filter.get_data(self.speed)
        print(filt_vel)
        target_acceleration = self.pid_speed.get_control(current_time, target_velocity - filt_vel)

        # Publish acceleration command
        self.accel_cmd.f64_cmd = target_acceleration  # Make sure this is the correct field
        self.accel_pub.publish(self.accel_cmd)

        # Convert and publish steering angle
        # Assuming target_steering is in degrees and needs conversion
        steering_radians = np.radians(target_steering)
        self.steer_cmd.angular_position = steering_radians
        self.steer_pub.publish(self.steer_cmd)

        self.prev_vel = curr_vel


    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0

        current_time = rospy.get_time()
        filt_vel     = self.speed_filter.get_data(self.speed)
        stop_accel = self.pid_speed.get_control(current_time, 0 - filt_vel)
        self.accel_cmb.f64_cmd = stop_accel
        
        self.accel_cmd(self.accel_cmd)
        self.controlPub.publish(newAckermannCmd)
