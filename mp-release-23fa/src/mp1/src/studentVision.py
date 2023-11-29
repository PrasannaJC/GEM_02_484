import time
import math
import numpy as np
import cv2
import rospy

# from line_fit import lineFit 
from line_fit import line_fit, tune_fit, bird_fit, final_viz, create_waypoints
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
from std_msgs.msg import Float32MultiArray



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()

        # rosbag
        self.sub_image = rospy.Subscriber('/zed2/zed_node/right_raw/image_raw_color', Image, self.img_callback, queue_size=1)
        
        self.pub_image = rospy.Publisher('/lane_detection/annotate', Image, queue_size=1)

        self.pub_bird = rospy.Publisher(
            "/lane_detection/birdseye", Image, queue_size=1)
        self.pub_waypoints = rospy.Publisher( 'chatter', Float32MultiArray, queue_size=10)
    
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        
        self.last_target = np.array([0, 0])
        self.target_change_threshold = 900  # set a threshold for maximum allowed change

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()

        mask_image, bird_image, waypoints = self.detection(raw_img)

       
        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
       
            self.pub_bird.publish(out_bird_msg)

            msg = Float32MultiArray()
            msg.data = sum(waypoints, [])

            self.pub_waypoints.publish(msg)
        # return waypoints
        return raw_img
        

    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)


        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        combined_sobel = cv2.addWeighted(sobel_x, 0.1, sobel_y, 0.9, 0)

        sobel_scaled = np.uint8(255*combined_sobel/np.max(combined_sobel))
        binary_output = np.zeros_like(sobel_scaled)
        binary_output[(sobel_scaled >= thresh_min) &
                      (sobel_scaled <= thresh_max)] = 1

        return binary_output


    def color_thresh(self, img, s_thresh=(0, 255), l_thresh=(0, 150)):
        """
        Convert RGB to HSL and threshold to binary image using S and L channels
        """
        hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hsl[:,:,2]
        l_channel = hsl[:,:,1]
        h_channel = hsl[:,:,0]
        
        yellow_hue_range = (10, 45)
        
        # Threshold the S channel for saturation
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        # Threshold the L channel for lightness to include shadows
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
        
        # Threshold the H channel for hue to capture yellow
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= yellow_hue_range[0]) & (h_channel <= yellow_hue_range[1])] = 1
        
        # Combine the S, L, and H thresholds
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_binary == 1) & (l_binary == 1) & (h_binary == 1)] = 1

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        # Apply sobel filter and color filter on input image
        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(img)
        # Combine the outputs
        
        # Invert the ColorOutput to create a mask that excludes unwanted colors
        ColorMask = 1 - ColorOutput

        # Combine the outputs using an AND operation
        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorMask == 1) & (SobelOutput == 1)] = 1

        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(
            binaryImage.astype('bool'), min_size=50, connectivity=2)

        binaryImage = binaryImage.astype('uint8')

        return binaryImage

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """

        img_size = (img.shape[1], img.shape[0])
        # <Rosbag transform>

        src = np.float32(
            [
                [500, 450],     # Upper left
                [780, 450],   # Upper right
                [1080, 700], # Lower right
                [0, 700],  # Lower left
            ]
        )
        dst = np.float32(
            [
                [0, 0],     # Upper left
                [1280, 0],   # Upper right
                [1280, 720], # Lower right
                [0, 720],  # Lower left
            ]
        )

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        warped_img = cv2.warpPerspective(
            img, M, img_size, flags=cv2.INTER_LINEAR)
        
        if verbose:
            # If verbose is true, visualize the source and destination points on the original and warped images
            for i in range(4):
                cv2.circle(img, tuple(src[i]), 10, (0, 0, 255), -1)
                cv2.circle(warped_img, tuple(dst[i]), 10, (0, 255, 0), -1)

        return warped_img, M, Minv

    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)


        

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']
            waypoints, ptsl, ptsr = create_waypoints(img_birdeye, ret)

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)
                try:
                    waypoints, ptsl, ptsr = create_waypoints(img_birdeye, ret)
                except:
                    waypoints = [[0,0],[640, 200]]
                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)
                    waypoints, ptsl, ptsr = create_waypoints(img_birdeye, ret)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)
                try:
                    waypoints, ptsl, ptsr = create_waypoints(img_birdeye, ret)
                except:
                    waypoints = [[0,0],[640, 200]]

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)
                    waypoints, ptsl, ptsr = create_waypoints(img_birdeye, ret)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img, a, b = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
                
                # Visualize waypoints
                # for waypoint in waypoints:
                waypoints, ptsl, ptsr = create_waypoints(img_birdeye, ret)
                x, y = waypoints[1]  
                # target = np.array([int(x), int(y)])
                        # Get the new target position
                new_target = np.array([int(x), int(y)])

                # Calculate the change from the last target
                change = np.linalg.norm(new_target - self.last_target)

                # Check if the change exceeds the threshold
                if change > self.target_change_threshold:
                    # If the change is too large, you can either ignore the update
                    # or adjust the target to a more gradual change.
                    # Here, I'm just ignoring the update.
                    target = self.last_target
                else:
                    # If the change is within the threshold, update the target
                    target = new_target
                    # Update the last target position
                    self.last_target = target
                current = np.array([640, 720])
                cv2.circle(bird_fit_img, tuple(target), 20, (0, 0, 255), -1)
                cv2.circle(bird_fit_img, tuple(current), 5, (235, 235, 52), -1)
                
                # create vector pointing from current pos to target
                vector = target - current
                fraction = 0.1 
                scaled_vector = fraction * vector
                vector_endpoint = current + scaled_vector
                
                alpha = np.arctan2( -target[1] + current[1], target[0] - current[0]) - np.pi/2
                # print("Wheel Angle: ",-np.rad2deg(alpha))

                # Draw the vector line
                cv2.line(bird_fit_img, tuple(current), tuple(vector_endpoint.astype(int)), (0, 255, 0), 2)  # Green line with thickness 2
                
                # Draw detected lane lines
                for point in ptsl:
                    x, y = point
                    cv2.circle(bird_fit_img, (int(x), int(y)), 5, (235, 235, 50), -1)

                for point in ptsr:
                    x, y = point
                    cv2.circle(bird_fit_img, (int(x), int(y)), 5, (235, 235, 50), -1)


            else:
                # print("Unable to detect lanes")
                pass

            # return combine_fit_img, bird_fit_img
            return combine_fit_img, bird_fit_img, waypoints


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)