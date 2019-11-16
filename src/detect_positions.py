#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import numpy as np
import cv2
import sys
from cv_bridge import CvBridge, CvBridgeError
from detect_centres import *


class ProcessImages:

    def __init__(self):
        rospy.init_node('detect_positions', anonymous=True)

        self.joints_sub1 = rospy.Subscriber("joints_pos_image_1", Float64MultiArray, self.callback1)
        self.joints_sub2 = rospy.Subscriber("joints_pos_image_2", Float64MultiArray, self.callback2)
        self.image_1_sub = rospy.Subscriber("image_topic1", Image, self.callback_img_1)
        self.image_1_sub = rospy.Subscriber("image_topic2", Image, self.callback_img_2)

        self.image_1_joints = np.zeros((4, 2))
        self.image_2_joints = np.zeros((4, 2))

        self.joint_positions = []
        self.angles = []
        self.bridge = CvBridge()

        # default values for cv_image1 & 2
        self.cv_image1 = None
        self.cv_image2 = None
        self.target_position = np.zeros(3)

    def callback_img_1(self, data):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_img_2(self, data):
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback2(self, data):
        self.image_2_joints = np.array(data.data).reshape((4, 2))
        # self.update()

    def callback1(self, data):
        self.image_1_joints = np.array(data.data).reshape((4, 2))
        # self.update()

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

    def detect_joint_angles(self):
        [yellow, blue, green, red] = self.joint_positions

        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow

        yellow2blue = blue - yellow

        g2b = green - blue
        joint1_angle = np.arctan2(g2b[1],
                                  g2b[0])

        z_rot_mat = np.array([
            [np.cos(joint1_angle), -np.sin(joint1_angle), 0],
            [np.sin(joint1_angle), np.cos(joint1_angle), 0],
            [0, 0, 1]
        ])

        blue_1 = np.dot(blue, z_rot_mat)
        green_1 = np.dot(green, z_rot_mat)

        blue2green = green_1 - blue_1
        temp_angle = np.arctan2(blue2green[2], blue2green[0])
        joint2_angle = (np.pi / 2) - temp_angle

        arm_2 = green - blue
        arm_3 = red - green
        joint3_angle = np.arccos(np.dot(arm_3, arm_2) / (np.linalg.norm(arm_2) * np.linalg.norm(arm_3)))

        self.angles = np.array([joint1_angle, 0, joint2_angle, joint3_angle])

    def jacobian_matrix(self, angles):
        a, b, c, d = self.angles
        sin = np.sin
        cos = np.cos

        jacobian_11 = np.array(
            2*c(d)*(sin(b)*cos(c)*cos(a) - sin(c)*sin(a)) +
            3*sin(b)*cos(c)*cos(a) +
            2*sin(d)*cos(b)*cos(a) +
            3*sin(c)*sin(a)
        )

        jacobian_12 = np.array(
            2*cos(d)*sin(a)*cos(c)*cos(b) +
            3*sin(a)*cos(c)*cos(b) -
            2*sin(a)*sin(d)*sin(b)
        )

        jacobian_13 = np.array(
            2*cos(d)*(cos(a)*cos(c) - sin(a)*sin(b)*sin(c)) +
            3*cos(a)*cos(c) -
            3*sin(a)*sin(b)*sin(c)
        )

        jacobian_14 = np.array(
            2*sin(a)*cos(b)*cos(d) -
            sin(d)*(cos(a)*sin(c) + sin(a)*sin(b)*cos(c))
        )

        jacobian_21 = np.array(
            2*cos(d)*(cos(a)*sin(c) +
            sin(a)*cos(c)*sin(b)) +
            3*cos(a)*sin(c) +
            3*sin(a)*cos(c)*sin(b) +
            2*sin(a)*cos(b)*sin(d)
        )

        jacobian_22 = np.array(
            2 * cos(a) * sin(b) * sin(d) -
            2 * cos(a) * cos(c) * cos(b) * cos(d) -
            3 * cos(a) * cos(c) * cos(b)
        )

        jacobian_23 = np.array(
            2 * cos(d) * (sin(a) * cos(c) +
            sin(b) * cos(a) * sin(c)) +
            3 * sin(a) * cos(c) +
            3 * sin(b) * cos(a) * sin(c)
        )

        jacobian_24 = np.array(
            -2 * sin(d) * (sin(a) * sin(c) -
            sin(b) * cos(a) * cos(c)) -
            2 * cos(a) * cos(b) * cos(d)
        )


        jacobian_31 = np.array(
            0
        )

        jacobian_32 = np.array(
            -2*cos(b) * sin(d) -
            2 * sin(b) * cos(d) * cos(c) -
            3 * sin(b) * cos(c)
        )

        jacobian_33 = np.array(
            -2 * cos(b) * cos(d) * sin(c) -
            3 * cos(b) * sin(c)
        )

        jacobian_34 = np.array(
            -2 * sin(b) * cos(d) -
            2 * cos(b) * sin(d) * cos(c)
        )

        jac_row_1 = np.array(jacobian_11, jacobian_12, jacobian_13, jacobian_14)
        jac_row_2 = np.array(jacobian_21, jacobian_22, jacobian_23, jacobian_24)
        jac_row_3 = np.array(jacobian_31, jacobian_32, jacobian_33, jacobian_34)
        return np.array(jac_row_1, jac_row_2, jac_row_3)

    def forward_kinematics(self):
        [a1, _, a2, a3] = self.angles
        [yellow, blue, green, red] = self.joint_positions

        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow

        yellow2blue = blue - yellow
        # shift to green
        step1 = np.array([
            [np.cos(a1), -np.sin(a1), 0, 0],
            [np.sin(a1), np.cos(a1), 0, 0],
            [0, 0, 1, yellow2blue[2]],
            [0, 0, 0, 1]
        ])
        blue_rot = np.dot(np.append(blue, 1), step1)
        green_rot = np.dot(np.append(green, 1), step1)
        blue2green = green_rot - blue_rot

        step2 = np.array([
            [np.cos(a2), 0, -np.sin(a2), blue2green[0]],
            [0, 1, 0, blue2green[1]],
            [np.sin(a2), 0, np.cos(a2), blue2green[2]],
            [0, 0, 0, 1]
        ])

        green_rot2 = np.dot(green_rot, step2)
        red_rot = np.dot(np.dot(np.append(red, 1), step1), step2)
        green2red = red_rot - green_rot2

        step3 = np.array([
            [1, 0, 0, green2red[0]],
            [0, np.cos(a3), -np.sin(a3), blue2green[1]],
            [0, np.sin(a3), np.cos(a3), green2red[2]],
            [0, 0, 0, 1]
        ])

        FK = np.dot(np.dot(step1, step2), step3)
        return FK[:, 3][:3]

    def manual_FK(self):
        [a1, _, a2, a3] = self.angles

        FK = np.array([
            [3 * np.cos(a1) * np.sin(a2) + 2 * np.cos(a3) * (
                        -np.cos(a1) * np.sin(a2) - np.sin(a1) * np.cos(a2)) - 2 * np.sin(a1) * np.sin(a3)],
            [2 * np.cos(a1) * np.sin(a3) + 3 * np.sin(a1) * np.sin(a2) + 2 * np.cos(a3) * (
                        np.cos(a1) * np.cos(a2) - np.sin(a1) * np.sin(a2))],
            [3 * np.cos(a2) + 2 * np.cos(a3) + 2]
        ])
        print(FK / self.pixel_to_meter())

    def detect_target(self):
        if self.cv_image1 is not None and self.cv_image2 is not None:
            img_1_circles = detect_circles(self.cv_image1)
            img_2_circles = detect_circles(self.cv_image2)

            # TODO here we need to check what comes back -- there may be more circles -- or none
            if img_1_circles is not None and len(img_1_circles) == 1 and \
                    img_2_circles is not None and len(img_2_circles) == 1:
                c1_y, c1_z, _ = img_1_circles[0][0]
                c2_x, c2_z, _ = img_2_circles[0][0]
                self.target_position[0] = c2_x
                self.target_position[1] = c1_y
                self.target_position[2] = 800 - ((c1_z + c2_z) / 2)

    def pixel_to_meter(self):
        [yellow, blue, green, red] = self.joint_positions
        f = np.linalg.norm(yellow - blue)
        s = np.linalg.norm(blue - green)
        t = np.linalg.norm(green - red)

        try:
            pixel_f = 2 / f
            pixel_s = 3 / s
            pixel_t = 2 / t
            mean = np.mean([pixel_t, pixel_s, pixel_f])
            return mean
        except ZeroDivisionError:
            return 0

    def distance_to_target(self):
        p2m = self.pixel_to_meter()
        pixel_distance = np.linalg.norm(self.joint_positions[0] - self.target_position)
        meter_distance = pixel_distance * p2m
        return meter_distance

    def update(self):
        self.get_joint_position()
        self.detect_joint_angles()
        # self.detect_target()
        # dist = self.distance_to_target()
        fk = self.forward_kinematics()
        print(fk, self.joint_positions[3] - self.joint_positions[0])
        self.manual_FK()
        # print(self.angles)


call the class
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
