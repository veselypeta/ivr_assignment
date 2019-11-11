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
        # self.image_1_sub = rospy.Subscriber("image_topic1", Image, self.callback_img_1)
        # self.image_1_sub = rospy.Subscriber("image_topic2", Image, self.callback_img_2)

        self.image_1_joints = np.zeros((4, 2))
        self.image_2_joints = np.zeros((4, 2))

        self.joint_positions = []
        self.angles = []
        self.bridge = CvBridge()

    def callback_img_1(self, data):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        im1 = cv2.imshow('window1', self.cv_image1)
        cv2.waitKey(1)
        # print(self.cv_image1.shape)

    def callback_img_2(self, data):
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        im1 = cv2.imshow('window2', self.cv_image2)
        cv2.waitKey(1)
        # print(self.cv_image1.shape)

    def callback2(self, data):
        self.image_2_joints = np.array(data.data).reshape((4, 2))
        # self.update()

    def callback1(self, data):
        self.image_1_joints = np.array(data.data).reshape((4, 2))
        # self.update()

    # assumption
    # We receive x,z from image2
    # we receive y,z from image1

    def get_joint_position(self):
        joint_positions = []
        for i in range(self.image_1_joints.shape[0]):
            xz_plane = self.image_2_joints[i]
            yz_plane = self.image_1_joints[i]
            # print(self.image_1_joints)
            # print(yz_plane)
            x = xz_plane[0]
            y = yz_plane[0]
            z = 800 - (xz_plane[1] + yz_plane[1]) / 2  # have z be the average from both cameras
            joint_positions.append([x, y, z])
        self.joint_positions = np.array(joint_positions)

    # now we have 3 angles to detect
    # xy, xz, yz
    def detect_joint_angles(self):
        [yellow, blue, green, red] = self.joint_positions

        j1_x = green - blue
        joint1_angle = np.arctan2(j1_x[1],
                                  j1_x[0])
        # print("joint 1 = " + str(joint1_angle))
        z_rot_mat = np.array([
            [np.cos(joint1_angle), -np.sin(joint1_angle), 0],
            [np.sin(joint1_angle), np.cos(joint1_angle), 0],
            [0, 0, 1],
        ])

        # apply rotation from joint 1
        joint2_rot = np.dot(blue, z_rot_mat)
        joint3_rot = np.dot(green, z_rot_mat)

        joint2_angle = (np.pi/2) - np.arctan2(joint3_rot[2] - joint2_rot[2], joint3_rot[0] - joint2_rot[0])

        y_rot_mat = np.array([
            [np.cos(joint2_angle), 0, -np.sin(joint2_angle)],
            [0, 1, 0],
            [np.sin(joint2_angle), 0, np.cos(joint2_angle)],
        ])

        # new rotation matrix after the first two rotations
        yz_rot = np.dot(z_rot_mat, y_rot_mat)

        joint3_rot_new = np.dot(green, yz_rot)
        end_rot = np.dot(red, yz_rot)

        diff = end_rot - joint3_rot_new
        # angle is on the yz plane
        joint3_angle = np.arctan2(diff[2], diff[1]) - (np.pi/2)
        self.angles = np.array([joint1_angle, 0, joint2_angle, joint3_angle])

    def update(self):
        self.get_joint_position()
        self.detect_joint_angles()
        # print(self.angles)

        # print(self.joint_positions)


# call the class
def main(args):
    ic = ProcessImages()
    try:
        while not rospy.is_shutdown():
            ic.update()
        # rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
