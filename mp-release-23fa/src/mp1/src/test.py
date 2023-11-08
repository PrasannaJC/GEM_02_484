#!/usr/bin/env python3
import time
import math
import numpy as np
import cv2
import rospy


from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology






class lanenet_detector():
   def __init__(self):


       self.bridge = CvBridge()
       # NOTE
       # Uncomment this line for lane detection of GEM car in Gazebo
       # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
       # Uncomment this line for lane detection of videos in rosbag
       self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
       self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
       self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
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
       mask_image, bird_image = self.detection(raw_img)


       if mask_image is not None and bird_image is not None:
           # Convert an OpenCV image into a ROS image message
           out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
           out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')


           # Publish image message in ROS
           self.pub_image.publish(out_img_msg)
           self.pub_bird.publish(out_bird_msg)




   def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
       """
       Apply sobel edge detection on input image in x, y direction
       """
       #1. Convert the image to gray scale
       #2. Gaussian blur the image
       #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
       #4. Use cv2.addWeighted() to combine the results
       #5. Convert each pixel to uint8, then apply threshold to get binary image


       ## TODO


       ####
       #cv2.imwrite("a.png",img)
       # img = cv2.imread("a.png",cv2.IMREAD_COLOR)
       # cv2.imshow("abc", img)
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
       blurred = cv2.GaussianBlur(gray, (5, 5), 0)
       # cv2.imwrite("b.png",blurred)


       sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
       sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)


       combined_sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
       # cv2.imwrite("b.png",combined_sobel)


       sobel_scaled = np.uint8(255*combined_sobel/np.max(combined_sobel))
       # print(sobel_scaled)
       binary_output = np.zeros_like(sobel_scaled)
       # print(binary_output)
       binary_output[(sobel_scaled >= thresh_min) & (sobel_scaled <= thresh_max)] = 1 # Default is 1, set to 255 to get debugging visuals.
       # print(binary_output)
       # cv2.imwrite("b.png",binary_output)
       cv2.imwrite("gradient_thresh.png",binary_output)
       return binary_output




   def color_thresh(self, img, thresh=(100, 255)):
       """
       Convert RGB to HSL and threshold to binary image using S channel
       """
       #1. Convert the image from RGB to HSL
       #2. Apply threshold on S channel to get binary image
       #Hint: threshold on H to remove green grass
       ## TODO


       ####
       hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


       s_channel = hsl[:,:,2]
       # print('s_channel',s_channel)
       h_channel = hsl[:,:,0]
       h_thresh = (50, 65)  # might need to tweak these values
       h_binary = np.zeros_like(h_channel)
       h_binary[(h_channel > h_thresh[0]) & (h_channel <= h_thresh[1])] = 1 # Default is 1, set to 255 to get debugging visuals.


       s_binary = np.zeros_like(s_channel)
       s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1 # Default is 1, set to 255 to get debugging visuals.
      
       binary_output = np.zeros_like(s_channel)
       binary_output[(s_binary == 1) & (h_binary == 1)] = 1 # Default is 1, set to 255 to get debugging visuals.
       # binary_output[(s_binary == 255) & (h_binary == 255)] = 255 # Default is 1, set to 255 to get debugging visuals.


       cv2.imwrite("color_thresh.png",binary_output)
       return binary_output




   def combinedBinaryImage(self, img):
       """
       Get combined binary image from color filter and sobel filter
       """
       # Apply sobel filter and color filter on input image
       SobelOutput = self.gradient_thresh(img)
       ColorOutput = self.color_thresh(img)
       # cv2.imwrite("c.png",ColorOutput)
       # Combine the outputs
       binaryImage = np.zeros_like(SobelOutput)
       binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1 # Default is 1, set to 255 to get debugging visuals.
       # binaryImage[(ColorOutput==255)|(SobelOutput==255)] = 255 # Default is 1, set to 255 to get debugging visuals.
       # Remove noise from binary image
       # binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'), min_size=50, connectivity=2)
      
       binaryImage = binaryImage.astype('uint8')
       cv2.imwrite("cbi.png",binaryImage)       
       # print('-------------------------------------------------------------------------------------------------')


       return binaryImage






   def perspective_transform(self, img, verbose=False):
       """
       Get bird's eye view from input image
       """
       #1. Visually determine 4 source points and 4 destination points
       #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
       #3. Generate warped image in bird view using cv2.warpPerspective()


       ## TODO
       # cv2.imshow("123", img)
       ####
       img_size = (img.shape[1], img.shape[0])
       # src = np.float32(
       #     [[img_size[0] / 2 - 10, img_size[1] / 2],
       #     [ 0 , img_size[1]],
       #     [ img_size[0], img_size[1]],
       #     [ img_size[0] / 2 + 10, img_size[1] / 2]])
       # dst = np.float32(
       #     [[(img_size[0] / 4), 0],
       #     [(img_size[0] / 4), img_size[1]],
       #     [(img_size[0] * 3 / 4), img_size[1]],
       #     [(img_size[0] * 3 / 4), 0]])




       src = np.float32(
           [
               [250, 250], # Upper left
               [335, 250], # Upper right
               [620, 480], # lower right
               [0, 480],   # lower left
           ]
       )
       dst = np.float32(
           [
               [0, 0],     # Upper left
               [540, 0],   # Upper right
               [450, 480], # Lower right
               [150, 480],  # Lower left
           ]
       )
          
       # Step 2: Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
       M = cv2.getPerspectiveTransform(src, dst)
       Minv = cv2.getPerspectiveTransform(dst, src)
      
       # Step 3: Generate warped image in bird view using cv2.warpPerspective()
       # print('flag1', type(img))
       warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
       cv2.imwrite("123.png", warped_img)
       if verbose:
           # If verbose is true, visualize the source and destination points on the original and warped images
           for i in range(4):
               cv2.circle(img, tuple(src[i]), 10, (0,0,255), -1)
               cv2.circle(warped_img, tuple(dst[i]), 10, (0,255,0), -1)
      
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


       else:
           # Fit lane with previous result
           if not self.detected:
               ret = line_fit(img_birdeye)


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
               ret = tune_fit(img_birdeye, left_fit, right_fit)


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
           else:
               print("Unable to detect lanes")


           return combine_fit_img, bird_fit_img




if __name__ == '__main__':
   # init args
   rospy.init_node('lanenet_node', anonymous=True)
   lanenet_detector()
   while not rospy.core.is_shutdown():
       rospy.rostime.wallsleep(0.5)
