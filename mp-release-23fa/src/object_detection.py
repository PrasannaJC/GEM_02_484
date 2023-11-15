import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg"):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")

        # This can be replaced with the frame input
        self.sub_image = rospy.Subscriber('/zed2/zed_node/right_raw/image_raw_color', Image, self.img_callback, queue_size=1)
        self.pub_detect_image = rospy.Publisher('/object_detection/objectfeed', Image, queue_size=1)
        self.pub_detect_bool = rospy.Publisher('detect_bool', Bool, queue_size=2)

        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="dnn_model/classes.txt"):

        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        frame = self.sub_image
        mask = np.zeros(frame)
        src = np.array([
                [(500, 450)],     # Upper left
                [(780, 450)],     # Upper right
                [(1080, 700)],    # Lower right
                [(0, 700)],       # Lower left
            ])

        cv2.fillPoly(mask, src, 255)
        trapezoidMask = cv2.bitwise_and(frame, src)

        # return self.model.detect(trapezoidMask, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

        # Detect objects on frame
        (class_ids, scores, boxes) = self.model.detect(trapezoidMask,
                                                       nmsThreshold=self.nmsThreshold,
                                                       confThreshold=self.confThreshold)
        for box in boxes:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (3, 219, 252), 2)

    # def detect(self):
    #     while True:
    #         ret, frame = self.sub_image.read()
    #         if not ret:
    #             break


if __name__ == '__main__':
    # init args
    rospy.init_node('object_node', anonymous=True)
    ObjectDetection()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

