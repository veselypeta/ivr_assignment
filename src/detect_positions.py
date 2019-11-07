#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String
import numpy as np
import cv2
import sys
from cv_bridge import CvBridge, CvBridgeError


class ProcessImages:

    def __init__(self):
        rospy.init_node('detect_positions', anonymous=True)

        self.joints_sub1 = rospy.Subscriber("joints_pos_image_1", Float64MultiArray, self.callback1)
        self.joints_sub2 = rospy.Subscriber("joints_pos_image_2", Float64MultiArray, self.callback2)

        self.image_1_joints = np.zeros((4, 2))
        self.image_2_joints = np.zeros((4, 2))

    def callback2(self, data):
        self.image_2_joints = np.array(data.data).reshape((4, 2))
        print("cb2 called")

    def callback1(self, data):
        self.image_1_joints = np.array(data.data).reshape((4, 2))
        print("db1 called")


# call the class
def main(args):
    ic = ProcessImages()
    try:
        print("started the process")
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
