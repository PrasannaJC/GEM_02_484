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
from controller import vehicleController
from std_msgs.msg import Float32MultiArray



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        self.sub_image = rospy.Subscriber(
            '/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # rosbag
        # self.sub_image = rospy.Subscriber('/zed2/zed_node/right_raw/image_raw_color', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher(
            "lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher(
            "lane_detection/birdseye", Image, queue_size=1)
        self.pub_waypoints = rospy.Publisher(
            'chatter', Float32MultiArray, queue_size=10)
    
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        # mask_image, bird_image = self.detection(raw_img)
        mask_image, bird_image, waypoints = self.detection(raw_img)

        # print('-----------------------------------------------')
        # print('waypoints', waypoints)

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
        # 1. Convert the image to gray scale
        # 2. Gaussian blur the image
        # 3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        # 4. Use cv2.addWeighted() to combine the results
        # 5. Convert each pixel to uint8, then apply threshold to get binary image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # cv2.imwrite('Grayblur.png', blurred)

        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        # cv2.imwrite("x.png",sobel_x)
        # cv2.imwrite("y.png",sobel_y)

        combined_sobel = cv2.addWeighted(sobel_x, 0.1, sobel_y, 0.9, 0)
        # cv2.imwrite("combined sobel.png",combined_sobel)

        sobel_scaled = np.uint8(255*combined_sobel/np.max(combined_sobel))
        binary_output = np.zeros_like(sobel_scaled)
        binary_output[(sobel_scaled >= thresh_min) &
                      (sobel_scaled <= thresh_max)] = 1

        # cv2.imwrite("gradient_thresh.png",binary_output*255)
        return binary_output

    def color_thresh(self, img, thresh=(20, 250)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass

        hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # cv2.imwrite("original.png", img)
        s_channel = hsl[:,:,2]
        h_channel = hsl[:,:,0]
        h_thresh = (0, 1)  # might need to tweak these values
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel > h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_binary == 1) & (h_binary == 1)] = 1

        # cv2.imwrite("color_thresh.png",binary_output*255)
        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        # Apply sobel filter and color filter on input image
        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(img)
        cv2.imwrite("c.png", SobelOutput)
        # Combine the outputs
        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput == 1) | (SobelOutput == 1)] = 1

        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(
            binaryImage.astype('bool'), min_size=50, connectivity=2)

        binaryImage = binaryImage.astype('uint8')
        cv2.imwrite("cbi.png", binaryImage*255)

        return binaryImage

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        # 1. Visually determine 4 source points and 4 destination points
        # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        # 3. Generate warped image in bird view using cv2.warpPerspective()

        # cv2.imshow("123", img)
        ####
        controller = vehicleController()
        currState = controller.getModelState()
        curr_x, curr_y, curr_vel, curr_yaw = controller.extract_vehicle_info(
            currState)
        img_size = (img.shape[1], img.shape[0])

        # gazebo warp
        # if curr_x > -5:
        src = np.float32(
            [
                [250, 290],     # Upper left
                [330, 290],   # Upper right
                [350, 470],  # Lower right
                [150, 470],  # Lower left
            ]
        )
        dst = np.float32(
            [
                [0, 0],     # Upper left
                [540, 0],   # Upper right
                [450, 480],  # Lower right
                [220, 480],  # Lower left
            ]
        )
        # else:
        #     src = np.float32(
        #         [
        #             [320, 275],     # Upper left
        #             [370, 275],   # Upper right
        #             [460, 470],  # Lower right
        #             [230, 470],  # Lower left
        #         ]
        #     )
        #     dst = np.float32(
        #         [
        #             [0, 0],     # Upper left
        #             [540, 0],   # Upper right
        #             [450, 480],  # Lower right
        #             [180, 480],  # Lower left
        #         ]
        #     )

        # Rosbag transform

        # src = np.float32(
        #     [
        #         [400, 390],     # Upper left
        #         [520, 390],   # Upper right
        #         [640, 480], # Lower right
        #         [200, 480],  # Lower left
        #     ]
        # )
        # dst = np.float32(
        #     [
        #         [0, 0],     # Upper left
        #         [640, 0],   # Upper right
        #         [640, 480], # Lower right
        #         [0, 480],  # Lower left
        #     ]
        # )

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        warped_img = cv2.warpPerspective(
            img, M, img_size, flags=cv2.INTER_LINEAR)
        # cv2.imwrite("warp.png", warped_img*255)
        if verbose:
            # If verbose is true, visualize the source and destination points on the original and warped images
            for i in range(4):
                cv2.circle(img, tuple(src[i]), 10, (0, 0, 255), -1)
                cv2.circle(warped_img, tuple(dst[i]), 10, (0, 255, 0), -1)

        return warped_img, M, Minv

    def detection(self, img):
        controller = vehicleController()
        currState = controller.getModelState()
        curr_x, curr_y, curr_vel, curr_yaw = controller.extract_vehicle_info(
            currState)

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        waypoints = create_waypoints(img_birdeye, curr_x)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye, curr_x)
            # waypoints = create_waypoints(img_birdeye, curr_x)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye, curr_x)
                # waypoints = create_waypoints(img_birdeye, curr_x)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit, curr_x)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
                
                # Visualize waypoints
                for waypoint in waypoints:
                    x, y = waypoint  # Assuming waypoint is a tuple (x, y)
                    # Transform (x, y) back to the perspective of the original image
                    # x_trans, y_trans = cv2.perspectiveTransform(np.array([[[x, y]]]), Minv)[0][0]
                    cv2.circle(bird_fit_img, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.circle(bird_fit_img, (320, 480), 5, (235, 235, 52), -1)
            else:
                print("Unable to detect lanes")

            # return combine_fit_img, bird_fit_img
            return combine_fit_img, bird_fit_img, waypoints


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)