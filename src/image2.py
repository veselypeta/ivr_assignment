#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
from detect_centres import *


class image_converter:

    def __init__(self):
        rospy.init_node('image_processing_2', anonymous=True)
        self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=1)
        self.joints_pub = rospy.Publisher("joints_pos_image_2", Float64MultiArray, queue_size=10)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        self.bridge = CvBridge()

    def callback2(self, data):
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Uncomment if you want to save the image
        # cv2.imwrite('image_copy.png', cv_image)
        # im2 = cv2.imshow('window2', self.cv_image2)
        cv2.waitKey(1)

        yellow_circle = detect_yellow(self.cv_image2)
        blue_circle = detect_blue(self.cv_image2)
        green_circle = detect_green(self.cv_image2)
        red_circle = detect_red(self.cv_image2)
        joints = Float64MultiArray()
        joints.data = np.array([yellow_circle, blue_circle, green_circle, red_circle]).flatten()

        # Publish the results
        try:
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
            self.joints_pub.publish(joints)
        except CvBridgeError as e:
            print(e)


# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
