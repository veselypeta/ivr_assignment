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

        self.joint_positions = []
        self.angles = []

    def callback2(self, data):
        self.image_2_joints = np.array(data.data).reshape((4, 2))
        print("cb2 called")

    def callback1(self, data):
        self.image_1_joints = np.array(data.data).reshape((4, 2))
        print("db1 called")

    # assumption
    # We receive x,z from image1
    # we receive y,z from image2

    def get_joint_position(self):
        joint_positions = []
        for i in range(self.image_1_joints.shape[0]):
            xz_plane = self.image_1_joints[i]
            yz_plane = self.image_2_joints[i]
            x = xz_plane[0]
            y = yz_plane[0]
            z = (xz_plane[1] + yz_plane[1]) / 2
            joint_positions.append([x, y, z])
        self.joint_positions = np.array(joint_positions)

    # now we have 3 angles to detect
    # xy, xz, yz
    def detect_joint_angles(self):
        try:
            xs = self.joint_positions[:, 0]
            ys = self.joint_positions[:, 1]
            zs = self.joint_positions[:, 2]

            angles = []
            for i in range(len(xs) - 1):
                xy = np.arctan2(ys[i] - ys[i + 1], xs[i] - xs[i + 1])
                xz = np.arctan2(zs[i] - zs[i + 1], xs[i] - xs[i + 1]) - xy
                yz = np.arctan2(ys[i] - ys[i + 1], zs[i] - zs[i + 1]) - xy - xz
                angles.append([xy, xz, yz])

            self.angles = np.array(angles)
        except IndexError:
            print("Error happened")
            return

    def update(self):
        self.get_joint_position()
        self.detect_joint_angles()
        print(self.angles)


# call the class
def main(args):
    ic = ProcessImages()
    try:
        while not rospy.is_shutdown():
            ic.update()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
