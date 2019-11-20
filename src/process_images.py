#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
import numpy as np
import cv2
import sys
from cv_bridge import CvBridge, CvBridgeError
from detect_centres import *
import message_filters
import time

from scipy.optimize import least_squares


class ImageProcessor:

    def __init__(self):
        rospy.init_node('process_images', anonymous=True)

        # ----- register subribers to a callback -----
        self.joints_sub1 = message_filters.Subscriber("joints_pos_image_1", Float64MultiArray)
        self.joints_sub2 = message_filters.Subscriber("joints_pos_image_2", Float64MultiArray)
        self.image_1_sub = message_filters.Subscriber("image_topic1", Image)
        self.image_2_sub = message_filters.Subscriber("image_topic2", Image)

        ts = message_filters.ApproximateTimeSynchronizer([self.joints_sub1, self.joints_sub2], 10, 10,
                                                         allow_headerless=True)
        ts.registerCallback(self.callback)

        ls = message_filters.ApproximateTimeSynchronizer([self.image_1_sub, self.image_2_sub], 10, 10,
                                                         allow_headerless=True)
        ls.registerCallback(self.image_callback)

        self.bridge = CvBridge()

        # data
        self.image1 = None
        self.image2 = None
        self.joints_cam1 = None
        self.joints_cam2 = None
        self.previous_target = None
        self.target_position = np.zeros(3)

        ## ----- publishers ------
        self.target_publisher = rospy.Publisher('target_topic', Float64MultiArray, queue_size=10)
        self.angles_publisher_full = rospy.Publisher('angles_topic', Float64MultiArray, queue_size=10)
        self.angles_publisher_without_2 = rospy.Publisher('angles_topic_without_2', Float64MultiArray, queue_size=10)
        self.end_effector_publisher = rospy.Publisher('end_effector_topic', Float64MultiArray, queue_size=10)
        self.jacobian_publisher_full = rospy.Publisher('jacobian_topic', Float64MultiArray, queue_size=10)
        self.jacobian_publisher_without_2 = rospy.Publisher('jacobian_topic_without_2', Float64MultiArray,
                                                            queue_size=10)

        self.plot_target = rospy.Publisher('plot_target_topic', Float64MultiArray, queue_size=10)
        self.plot_end_ef = rospy.Publisher('plot_end_ef_topic', Float64MultiArray, queue_size=10)

    def image_callback(self, img1, img2):
        try:
            self.image1 = self.bridge.imgmsg_to_cv2(img1, "bgr8")
            self.image2 = self.bridge.imgmsg_to_cv2(img2, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback(self, joints_1, joints_2):
        self.joints_cam1 = np.array(joints_1.data).reshape((4, 2))
        self.joints_cam2 = np.array(joints_2.data).reshape((4, 2))
        self.publish()

    def get_joint_position(self):
        joint_positions = []
        for i in range(self.joints_cam1.shape[0]):
            xz_plane = self.joints_cam2[i]
            yz_plane = self.joints_cam1[i]
            x = xz_plane[0]
            y = yz_plane[0]
            z = 800 - (xz_plane[1] + yz_plane[1]) / 2  # have z be the average from both cameras
            joint_positions.append([x, y, z])
        return np.array(joint_positions)

    def distance_to_end_effector(self, x):
        base, _, _, end = self.get_joint_position()
        end_effector_pos = end - base
        predicted_end_effector = self.forward_kinematics_4_angles(x)
        return predicted_end_effector - end_effector_pos

    def distance_to_end_effector_3_angles(self, x):
        base, _, _, end = self.get_joint_position()
        end_effector_pos = end - base
        predicted_end_effector = self.forward_kinematics_3_angles(x)
        return predicted_end_effector - end_effector_pos

    def get_joint_angles_full(self):
        return least_squares(self.distance_to_end_effector, [0.5, 0.5, 0.5, 0.5],
                             bounds=([0, 0, 0, 0], [np.pi, np.pi / 2, np.pi / 2, np.pi / 2]))

    def get_joint_angles_ignoring_2(self):
        return least_squares(self.distance_to_end_effector_3_angles, [0.5, 0.5, 0.5],
                             bounds=([0, 0, 0], [np.pi, np.pi / 2, np.pi / 2]))

    def forward_kinematics_4_angles(self, x):
        a1, a2, a3, a4 = x
        [yellow, blue, green, red] = self.get_joint_position()
        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow

        arm1 = np.linalg.norm(blue - yellow)
        arm3 = np.linalg.norm(red - green)
        arm2 = np.linalg.norm(green - blue)
        T1 = np.dot(self.z_rot_mat(a1), self.trans_along_z(arm1))
        T2 = self.x_rot_mat(a2)
        T3 = np.dot(self.y_rot_mat(a3), self.trans_along_z(arm2))
        T4 = np.dot(self.x_rot_mat(a4), self.trans_along_z(arm3))
        return np.dot(np.dot(np.dot(T1, T2), T3), T4)[:, 3][:3]

    def forward_kinematics_3_angles(self, x):
        a1, a2, a3 = x
        [yellow, blue, green, red] = self.get_joint_position()
        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow

        arm1 = np.linalg.norm(blue - yellow)
        arm3 = np.linalg.norm(red - green)
        arm2 = np.linalg.norm(green - blue)
        T1 = np.dot(self.z_rot_mat(a1), self.trans_along_z(arm1))
        T2 = np.dot(self.y_rot_mat(a2), self.trans_along_z(arm2))
        T3 = np.dot(self.x_rot_mat(a3), self.trans_along_z(arm3))
        return np.dot(np.dot(T1, T2), T3)[:, 3][:3]

    def trans_along_z(self, amount):
        identity = np.identity(4)
        identity[2][3] = amount
        return identity

    def x_rot_mat(self, x):
        x_rot = np.array([
            [1, 0, 0, 0],
            [0, np.cos(x), -np.sin(x), 0],
            [0, np.sin(x), np.cos(x), 0],
            [0, 0, 0, 1],
        ])
        return x_rot

    def y_rot_mat(self, y):
        y_rot = np.array([
            [np.cos(y), 0, np.sin(y), 0],
            [0, 1, 0.0, 0],
            [-np.sin(y), 0.0, np.cos(y), 0],
            [0, 0, 0, 1]
        ])
        return y_rot

    def z_rot_mat(self, z):
        z_rot = np.array([
            [np.cos(z), -np.sin(z), 0.0, 0],
            [np.sin(z), np.cos(z), 0.0, 0],
            [0.0, 0.0, 1.0, 0],
            [0, 0, 0, 1]
        ])
        return z_rot

    def detect_target(self):
        if self.image1 is not None and self.image2 is not None:
            img_1_circles = detect_circles(self.image1)
            img_2_circles = detect_circles(self.image2)

            if img_1_circles is not None and img_2_circles is not None:

                if self.previous_target is None:
                    c1_y, c1_z, _ = img_1_circles[0][0]
                    c2_x, c2_z, _ = img_2_circles[0][0]

                else:
                    cam1 = [np.linalg.norm(x[0:2] - self.previous_target[1:3]) for x in img_1_circles[0, :]]
                    min_cam_1 = np.argmin(cam1)
                    c1_y, c1_z, _ = img_1_circles[0][min_cam_1]

                    cam2 = [np.linalg.norm(x[0:2] - self.previous_target[0:3:2]) for x in img_2_circles[0, :]]
                    min_cam_2 = np.argmin(cam2)
                    c2_x, c2_z, _ = img_2_circles[0][min_cam_2]

                self.target_position[0] = c2_x
                self.target_position[1] = c1_y
                self.target_position[2] = 800 - ((c1_z + c2_z) / 2)

                self.previous_target = self.target_position

    def pixel_to_meter(self):
        [yellow, blue, green, red] = self.get_joint_position()
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

    def publish(self):
        all_positions = self.get_joint_position()
        p2m = self.pixel_to_meter()
        offset = np.array([-0.31938219,  0.48676221, -0.60965029])

        # ---- publish target position -----
        self.detect_target()
        target = Float64MultiArray()
        target.data = self.target_position - all_positions[0]
        self.target_publisher.publish(target)

        # ----- publish end effector position -----
        end_effector_pos = (all_positions[3] - all_positions[0])
        ef = Float64MultiArray()
        ef.data = end_effector_pos
        self.end_effector_publisher.publish(ef)

        # ----- publish 4 angles -----
        sol_4_angles = self.get_joint_angles_full()
        angles_full = Float64MultiArray()
        angles_full.data = sol_4_angles.x
        self.angles_publisher_full.publish(angles_full)

        # ----- publish 3 angles -----
        sol_3_angles = self.get_joint_angles_ignoring_2()
        angles_three = Float64MultiArray()
        angles_three.data = sol_3_angles.x
        self.angles_publisher_without_2.publish(angles_three)

        # ----- publish 4 angles jacobian -----
        jac4 = Float64MultiArray()
        jac4.data = sol_4_angles.jac.flatten()
        self.jacobian_publisher_full.publish(jac4)

        # ----- publish 3 angles jacobian -----
        jac3 = Float64MultiArray()
        jac3.data = sol_3_angles.jac.flatten()
        self.jacobian_publisher_without_2.publish(jac3)

        # ----- publish end-effector in meters -----
        ef_meter = Float64MultiArray()
        ef_meter.data = end_effector_pos * p2m
        self.plot_end_ef.publish(ef_meter)
        #
        # ----- publish target in meters -----
        target_meter = Float64MultiArray()
        target_meter.data = (self.target_position - all_positions[0]) * p2m
        self.plot_target.publish(target_meter)
        #
        # ----- test kinematics formula with joint angles -----
        test_angles = np.array([1, 1, 1, 1])
        print(" --------------------------------------------------------------------------------- ")
        print("Angles Used => 1 = {} rad | 2 = {} rad | 3 = {} rad | 4 = {} rad").format(*tuple(test_angles))
        print("Measured End-Effector Position   -> " + str(ef_meter.data))
        print("Calculated End-Effector Position -> " + str(self.forward_kinematics_4_angles(test_angles) * p2m))
        print(ef_meter.data - (self.forward_kinematics_4_angles(test_angles) * p2m))


def main(args):
    ic = ImageProcessor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
