import rospy
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
        self.prev_vel = 0
        # self.L = 65 # Wheelbase, can be get from gem_control.py
        self.L = 210
        self.log_acceleration = True
        self.accelerations = []
        self.x = []
        self.y = []
        self.fix_x = 640
        self.fix_y = 720
        self.fix_yaw = np.pi/2
        
        
        self.gem_enable = True
        self.pacmod_enable = True

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = True

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
        self.pid_speed = PID(0.82, 0.0, 0.01)  # Tune these parameters
        
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        # Publishers
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)

        # Commands
        self.accel_cmd = PacmodCmd()
        self.steer_cmd = PositionWithSpeed()

    
    def enable_callback(self, msg):
        self.pacmod_enable = msg.data    
        
    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3)

    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, future_unreached_waypoints):
        
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
        print(steer_angle)
        if steer_angle > 0:
            steer_angle = 0.8*steer_angle
        return steer_angle, curve
    
    def longititudal_controller(self, curve):

        if curve:
            target_velocity = 1.2
        else:
            target_velocity = 1.3

        return target_velocity
    

    def execute(self, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None
        
        curr_x = self.fix_x
        curr_y = self.fix_y
        curr_yaw = self.fix_yaw

        self.gear_cmd.ui16_cmd = 3 # switch to forward gear
        self.brake_cmd.enable  = True
        self.brake_cmd.clear   = False
        self.brake_cmd.ignore  = False
        self.brake_cmd.f64_cmd = 0.0

        # enable gas 
        self.accel_cmd.enable  = True
        self.accel_cmd.clear   = False
        self.accel_cmd.ignore  = False
        self.accel_cmd.f64_cmd = 0.0
        self.steer_cmd.angular_velocity_limit = 2.0 # radians/second
        
        self.gear_pub.publish(self.gear_cmd)

        self.gem_enable = True

        target_steering, curve = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, future_unreached_waypoints)
        target_velocity = self.longititudal_controller(curve)
        
        current_time = rospy.get_time()
        filt_vel = self.speed_filter.get_data(self.speed)
        target_acceleration = self.pid_speed.get_control(current_time, target_velocity - filt_vel)
        # print("Filtered Velocity: ", filt_vel)
        # print("Target Accel: ", target_acceleration)

        # Publish acceleration command
        self.accel_cmd.f64_cmd = target_acceleration  # Make sure this is the correct field
        self.accel_pub.publish(self.accel_cmd)


        # Convert and publish steering angle
        # Assuming target_steering is in degrees and needs conversion
        steering_radians = np.radians(target_steering)
        self.steer_cmd.angular_position = steering_radians
        self.steer_pub.publish(self.steer_cmd)



    def stop(self):
        # current_time = rospy.get_time()
        # filt_vel     = self.speed_filter.get_data(self.speed)
        # stop_accel = self.pid_speed.get_control(current_time, 0 - filt_vel)
        # self.accel_cmd.f64_cmd = stop_accel
        
        # self.accel_pub.publish(self.accel_cmd)
        pass