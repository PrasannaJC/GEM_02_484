import rospy
# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64
# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt
import time

class testCar(object):
    def __init__(self):
        self.accel = 0.0
        self.accel_sub = rospy.Subscriber('/pacmod/as_rx/accel_cmd', PacmodCmd, self.accel_callback)
        
        self.speed_pub = rospy.Publisher("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, queue_size=1)
        
        self.current_speed = 0.0
        
        print(self.current_speed)
        
    
    def accel_callback(self, msg):
        print('callback')
        self.accel = round(msg.f64_cmd, 3)
        self.current_speed = 100 * self. accel + self.current_speed
        print('Accel: ', self.accel)
        print('\n Speed: ', self.current_speed)
        speed_rpt = VehicleSpeedRpt()
        speed_rpt.vehicle_speed = self.current_speed
        self.speed_pub.publish(speed_rpt)
        
if __name__ == '__main__':
    myCar = testCar()
    while True:
        time.sleep(10)

    
        
        
