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

from scipy.optimize import least_squares


class ProcessImages:

    def __init__(self):
        rospy.init_node('detect_positions', anonymous=True)

        self.joints_sub1 = message_filters.Subscriber("joints_pos_image_1", Float64MultiArray)
        self.joints_sub2 = message_filters.Subscriber("joints_pos_image_2", Float64MultiArray)
        self.image_1_sub = rospy.Subscriber("image_topic1", Image, self.callback_img_1)
        self.image_2_sub = rospy.Subscriber("image_topic2", Image, self.callback_img_2)

        ts = message_filters.ApproximateTimeSynchronizer([self.joints_sub1, self.joints_sub2], 10, 10,
                                                         allow_headerless=True)
        ts.registerCallback(self.callback)

        self.target_publisher = rospy.Publisher('target_topic', Float64MultiArray, queue_size=10)
        self.angles_publisher = rospy.Publisher('angles_topic', Float64MultiArray, queue_size=10)
        self.end_effector_publisher = rospy.Publisher('end_effector_topic', Float64MultiArray, queue_size=10)
        self.jacobian_publisher = rospy.Publisher('jacobian_topic', Float64MultiArray, queue_size=10)
        self.plot_target = rospy.Publisher('plot_target_topic', Float64MultiArray, queue_size=10)
        self.plot_end_ef = rospy.Publisher('plot_end_ef_topic', Float64MultiArray, queue_size=10)

        self.image_1_joints = np.zeros((4, 2))
        self.image_2_joints = np.zeros((4, 2))

        self.joint_positions = np.zeros((4, 3))
        self.angles = []
        self.bridge = CvBridge()

        # default values for cv_image1 & 2
        self.cv_image1 = None
        self.cv_image2 = None
        self.target_position = np.zeros(3)
        self.previous_angles = np.ones(4)
        self.previous_target = None

    def distance_to_end_effector(self, test_angles):
        end_effector_pos = self.joint_positions[3] - self.joint_positions[0]
        predicted_end_effector = self.working_forward_kinematics([test_angles[0], test_angles[1], test_angles[2], test_angles[3]])
        return predicted_end_effector - end_effector_pos

    def without2_error(self, test_angles):
        end_effector_pos = self.joint_positions[3] - self.joint_positions[0]
        predicted_end_effector = self.kinematics_without_2(test_angles)
        return predicted_end_effector - end_effector_pos

    def minimising_angles(self):
        x = self.joint_positions[3] - self.joint_positions[0]
        return least_squares(self.distance_to_end_effector, [0.5, 0.5, 0.5, 0.5], bounds=([0, 0, 0, 0], [np.pi, np.pi/2, np.pi/2, np.pi/2]))

    def minimising_angles_without2(self):
        # x = self.joint_positions[3] - self.joint_positions[0]
        return least_squares(self.without2_error, [0.5, 0.5, 0.5], bounds=([0, 0, 0], [np.pi/2, np.pi/2, np.pi/2]))

    def working_forward_kinematics(self, x):
        a1, a2, a3, a4 = x
        [yellow, blue, green, red] = self.joint_positions
        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow

        yellow2blue = blue - yellow
        blue2green = green - blue
        green2red = red - green

        arm1len = np.linalg.norm(yellow2blue)
        arm2len = np.linalg.norm(blue2green)
        arm3len = np.linalg.norm(green2red)

        c = np.cos
        s = np.sin

        yellow_rot = np.array([
            [np.cos(a1), -np.sin(a1), 0.0, 0],
            [np.sin(a1), np.cos(a1), 0.0, 0],
            [0.0, 0.0, 1.0, 0],
            [0, 0, 0, 1]
        ])

        yellow_tran = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, arm1len],
            [0, 0, 0, 1]
        ])

        T1 = np.dot(yellow_rot, yellow_tran)

        T2 = np.array([
            [1, 0, 0, 0],
            [0, c(a2), -s(a2), 0],
            [0, s(a2), c(a2), 0],
            [0, 0, 0, 1]
        ])


        y_rot = np.array([
            [np.cos(a3), 0, np.sin(a3), 0],
            [0, 1, 0.0, 0],
            [-np.sin(a3), 0.0, np.cos(a3), 0],
            [0, 0, 0, 1]
        ])

        y_tran = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, arm2len],
            [0, 0, 0, 1]
        ])

        T3 = np.dot(y_rot, y_tran)

        x_rot = np.array([
            [1, 0, 0, 0],
            [0, np.cos(a4), -np.sin(a4), 0],
            [0, np.sin(a4), np.cos(a4), 0],
            [0, 0, 0, 1],
        ])
        x_tran = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, arm3len],
            [0, 0, 0, 1]
        ])

        T4 = np.dot(x_rot, x_tran)

        return np.dot(np.dot(np.dot(T1, T2), T3), T4)[:, 3][:3]


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

    def callback(self, image1, image2):
        self.image_1_joints = np.array(image1.data).reshape((4, 2))
        self.image_2_joints = np.array(image2.data).reshape((4, 2))

        self.update()

    def get_joint_position(self):
        joint_positions = []
        for i in range(self.image_1_joints.shape[0]):
            xz_plane = self.image_2_joints[i]
            yz_plane = self.image_1_joints[i]
            x = xz_plane[0]
            y = yz_plane[0]
            if np.any(xz_plane == 0):
                x = self.joint_positions[i][0]
            if np.any(yz_plane) == 0:
                y = self.joint_positions[i][1]

            z = 800 - (xz_plane[1] + yz_plane[1]) / 2  # have z be the average from both cameras
            joint_positions.append([x, y, z])
        self.joint_positions = np.array(joint_positions)

    def forward_kinematics(self, angles):
        [a1, a2, a3, a4] = angles
        [yellow, blue, green, red] = self.joint_positions
        blue = blue - yellow
        green = green - yellow
        red = red - yellow

        arm1 = np.linalg.norm(blue - yellow)
        arm3 = np.linalg.norm(red - green)
        arm2 = np.linalg.norm(green - blue)

        M1 = np.array([
            [np.cos(a1), -np.sin(a1), 0, 0],
            [np.sin(a1), np.cos(a1), 0, 0],
            [0, 0, 1, arm2],
            [0, 0, 0, 1]
        ])

        M2 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(a2), -np.sin(a2), 0],
            [0, np.sin(a2), np.cos(a2), 0],
            [0, 0, 0, 1]
        ])

        M3 = np.array([
            [np.cos(a3), 0, np.sin(a3), np.sin(a3) * arm2],
            [0, 1, 0, 0],
            [-np.sin(a3), 0, np.cos(a2), np.cos(a3) * arm2],
            [0, 0, 0, 1]
        ])

        M4 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(a4), -np.sin(a4), -np.sin(a4) * arm3],
            [0, np.sin(a4), np.cos(a4), np.cos(a4) * arm3],
            [0, 0, 0, 1]
        ])

        O = np.array([0, 0, 0, 1])
        Y = O
        B = np.dot(M1, O)
        G = np.dot(np.matmul(np.matmul(M1, M2), M3), O)
        R = np.dot(np.matmul(np.matmul(np.matmul(M1, M2), M3), M4), O)

        return np.matmul(np.matmul(M1, M2), M3)[:3, :3]

    def jacobian_matrix(self, angles):
        a, b, c, d = angles
        sin = np.sin
        cos = np.cos

        jacobian_11 = np.array(
            2 * cos(d) * (sin(b) * cos(c) * cos(a) - sin(c) * sin(a)) +
            3 * sin(b) * cos(c) * cos(a) +
            2 * sin(d) * cos(b) * cos(a) +
            3 * sin(c) * sin(a)
        )

        jacobian_12 = np.array(
            2 * cos(d) * sin(a) * cos(c) * cos(b) +
            3 * sin(a) * cos(c) * cos(b) -
            2 * sin(a) * sin(d) * sin(b)
        )

        jacobian_13 = np.array(
            2 * cos(d) * (cos(a) * cos(c) - sin(a) * sin(b) * sin(c)) +
            3 * cos(a) * cos(c) -
            3 * sin(a) * sin(b) * sin(c)
        )

        jacobian_14 = np.array(
            2 * sin(a) * cos(b) * cos(d) -
            sin(d) * (cos(a) * sin(c) + sin(a) * sin(b) * cos(c))
        )

        jacobian_21 = np.array(
            2 * cos(d) * (cos(a) * sin(c) +
                          sin(a) * cos(c) * sin(b)) +
            3 * cos(a) * sin(c) +
            3 * sin(a) * cos(c) * sin(b) +
            2 * sin(a) * cos(b) * sin(d)
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
            -2 * cos(b) * sin(d) -
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

        jac_row_1 = np.array([jacobian_11, jacobian_12, jacobian_13, jacobian_14])
        jac_row_2 = np.array([jacobian_21, jacobian_22, jacobian_23, jacobian_24])
        jac_row_3 = np.array([jacobian_31, jacobian_32, jacobian_33, jacobian_34])
        return np.array([jac_row_1, jac_row_2, jac_row_3])

    def kinematics_without_2(self, x):
        [yellow, blue, green, red] = self.joint_positions
        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow

        yellow2blue = blue - yellow
        blue2green = green - blue
        green2red = red - green

        arm1len = np.linalg.norm(yellow2blue)
        arm2len = np.linalg.norm(blue2green)
        arm3len = np.linalg.norm(green2red)

        a1, a2, a3 = x
        c = np.cos
        s = np.sin
        k = np.array([
            arm3len*s(a1)*s(a3) + arm3len*c(a1)*s(a2)*c(a3) + arm2len*c(a1)*s(a2),
            arm3len*s(a1)*s(a2)*c(a3) - arm3len*c(a1)*s(a3) + arm2len*s(a1)*s(a2),
            arm1len + arm2len*c(a2) + arm3len*c(a2)*c(a3)
        ])
        return k

    def detect_angles_ignoring_joint_2(self):
        [yellow, blue, green, red] = self.joint_positions
        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow


        yellow2blue = blue - yellow
        blue2green = green - blue
        green2red = red - green
        a1 = np.arctan2(blue2green[1], blue2green[0])
        arm1len = np.linalg.norm(yellow2blue)
        arm2len = np.linalg.norm(blue2green)
        arm3len = np.linalg.norm(green2red)

        Y_b = np.cross(yellow2blue, blue2green) / np.linalg.norm(np.cross(yellow2blue, blue2green))
        Z_b = yellow2blue / np.linalg.norm(yellow2blue)
        X_b = np.cross(Y_b, Z_b) / np.linalg.norm(np.cross(Y_b, Z_b))

        a2 = np.arctan2(np.dot(blue2green, X_b), np.dot(blue2green, Z_b))
        z_rot = np.array([
            [np.cos(a1), -np.sin(a1), 0.0, 0],
            [np.sin(a1), np.cos(a1), 0.0, 0],
            [0.0, 0.0, 1.0, 0],
            [0, 0, 0, 1]
        ])

        z_tran = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, arm1len],
            [0, 0, 0, 1]
        ])

        z_com = np.dot(z_rot, z_tran)
        new_blue = np.dot(z_com, np.append(blue, 1))


        y_rot = np.array([
            [np.cos(a2), 0, np.sin(a2), 0],
            [0, 1, 0.0, 0],
            [-np.sin(a2), 0.0, np.cos(a2), 0],
            [0, 0, 0, 1]
        ])

        y_tran = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, arm2len],
            [0, 0, 0, 1]
        ])

        y_com = np.dot(y_rot, y_tran)
        R2 = np.dot(z_com, y_com)
        new_green = np.dot(R2, np.append(yellow, 1))
        distance = np.linalg.norm(new_green[:3] - green)

        X_g = np.cross(blue2green, green2red) / np.linalg.norm(np.cross(blue2green, green2red))
        Z_g = blue2green / np.linalg.norm(blue2green)
        Y_g = np.cross(X_g, Z_g) / np.linalg.norm(np.cross(X_g, Z_g))
        a3 = np.arctan2(np.dot(Y_g, green2red), np.dot(Z_g, green2red))

        x_rot = np.array([
            [1, 0, 0, 0],
            [0, np.cos(a3), -np.sin(a3), 0],
            [0, np.sin(a3), np.cos(a3), 0],
            [0, 0, 0, 1],
        ])
        x_tran = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, arm3len],
            [0, 0, 0, 1]
        ])

        x_com = np.dot(x_rot, x_tran)

        R3 = np.dot(R2, x_com)
        new_red = np.dot(R3, np.append(yellow, 1))
        dist = np.linalg.norm(new_red[:3] - red)

        if dist > 20:
            a3 *= -1

        return np.array([a1, a2, a3])

    def detect_joint_angles(self):
        [yellow, blue, green, red] = self.joint_positions
        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow

        arm1 = np.linalg.norm(blue - yellow)
        arm3 = np.linalg.norm(red - green)
        arm2 = np.linalg.norm(green - blue)

        Q = (red - green)
        P = (green - blue)

        Z = P / np.linalg.norm(P)
        X = np.cross(P, Q) / np.linalg.norm(np.cross(P, Q))
        Y = np.cross(Z, X)

        XYZ = [X, Y, Z]

        a1 = np.arctan2(-Y[0], Y[1])
        a2 = np.arctan2(Y[2], Y[1] / np.cos(a1))
        a3 = np.arctan2(-X[2], Z[2])
        a4 = np.arccos(np.dot(P, Q) / np.linalg.norm(P) / np.linalg.norm(Q))


        R1_3 = np.array([
            [np.cos(a1) * np.cos(a3) - np.sin(a1) * np.sin(a2) * np.sin(a3), -np.sin(a1) * np.cos(a2),
             np.cos(a1) * np.sin(a3) + np.sin(a1) * np.sin(a2) * np.cos(a3)],
            [np.sin(a1) * np.cos(a3) + np.cos(a1) * np.sin(a2) * np.sin(a3), np.cos(a1) * np.cos(a2),
             np.sin(a1) * np.sin(a3) - np.cos(a1) * np.sin(a2) * np.cos(a3)],
            [-np.cos(a2) * np.sin(a3), np.sin(a2), np.cos(a2) * np.cos(a3)]
        ])



        Z_x_error = np.absolute(np.absolute(Z[0]) - np.absolute(R1_3[0, 2]))

        XYZ_ac = self.forward_kinematics([a1, a2, a3, a4])

        threshold = 0.3

        if Z_x_error > threshold:
            X = -X
            Y = np.cross(Z, X)
            a1 = np.arctan2(-Y[0], Y[1])
            a2 = np.arctan2(Y[2], Y[1] / np.cos(a1))
            a3 = np.arctan2(-X[2], Z[2])
            a4 = -a4

        joint_angles = np.array([a1, a2, a3, a4])
        self.angles = joint_angles

    def detect_target(self):
        if self.cv_image1 is not None and self.cv_image2 is not None:
            img_1_circles = detect_circles(self.cv_image1)
            img_2_circles = detect_circles(self.cv_image2)


            if img_1_circles is not None and img_2_circles is not None:
                
                if self.previous_target is None:
                    c1_y, c1_z, _ = img_1_circles[0][0]
                    c2_x, c2_z, _ = img_2_circles[0][0]
                
                else:    
                    cam1 = [np.linalg.norm(x[0:2] - self.previous_target[1:3]) for x in img_1_circles[0,:]]
                    min_cam_1 = np.argmin(cam1)
                    c1_y, c1_z, _ = img_1_circles[0][min_cam_1]
                
                    cam2 = [np.linalg.norm(x[0:2] - self.previous_target[0:3:2]) for x in img_2_circles[0,:]]
                    min_cam_2 = np.argmin(cam2)
                    c2_x, c2_z, _ = img_2_circles[0][min_cam_2]
                
                
             
                self.target_position[0] = c2_x
                self.target_position[1] = c1_y
                self.target_position[2] = 800 - ((c1_z + c2_z) / 2)
              
                        
                self.previous_target = self.target_position
        
            

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

    def simple_jacobian(self):
        a1, a2, a3 = self.detect_angles_ignoring_joint_2()
        c = np.cos
        s = np.sin

        [yellow, blue, green, red] = self.joint_positions
        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow

        yellow2blue = blue - yellow
        blue2green = green - blue
        green2red = red - green

        x = np.linalg.norm(yellow2blue)
        y = np.linalg.norm(blue2green)
        z = np.linalg.norm(green2red)

        jac_row_1 = np.array([
            z*c(a1)*s(a3) - z*s(a1)*s(a2)*c(a3) - y*s(a1)*s(a2),
            z*c(a1)*c(a2)*c(a3) + y*c(a1)*c(a2),
            z*s(a1)*c(a3) - z*c(a1)*s(a2)*s(a3)
        ])

        jac_row_2 = np.array([
            z*c(a1)*s(a2)*c(a3) + z*s(a1)*s(a3) + y*c(a1)*s(a2),
            z*s(a1)*c(a2)*c(a3) + y*s(a1)*c(a2),
            -z*s(a1)*s(a2)*s(a3) - z*c(a1)*c(a3)
        ])

        jac_row_3 = np.array([
            0,
            -y*s(a2) - z*s(a2)*c(a3),
            -z*c(a2)*s(a3)
        ])

        full_jacobian = np.array([
            jac_row_1,
            jac_row_2,
            jac_row_3
        ])
        return full_jacobian


    def update(self):
        pass
        self.get_joint_position()
        # k = self.kinematics_without_2()
        self.detect_target()
        # print(self.detect_angles_ignoring_joint_2())
        #simple_jac = self.simple_jacobian()
        #min_angles = self.minimising_angles()

        # print(self.working_forward_kinematics([1, 1, 1, 1]), self.joint_positions[3])
        # print(min_angles.x)

        #min_angles_2 = self.minimising_angles_without2()
        # print(min_angles_2.x)


        # print(self.joint_positions[3] - self.joint_positions[0], self.working_forward_kinematics(min_angles.x))
        # self.detect_angles_ignoring_joint_2()
        # self.detect_joint_angles()
        # print(self.angles)
        # print(self.angles)
        # self.detect_target()
        # print(self.joint_positions)
        # dist = self.distance_to_target()
        # fk = self.forward_kinematics()
        # print(fk, self.joint_positions[3] - self.joint_positions[0])
        # print(self.angles)
        # print(self.target_position)

        self.publish_results()
        # print(self.angles)

    def publish_results(self):
        # pub_angles = Float64MultiArray()
        # pub_angles.data = self.angles
        # pub_end_effector = Float64MultiArray()
        # pub_end_effector.data = self.joint_positions[3]
        # pub_jacobian = Float64MultiArray()
        # pub_jacobian.data = self.jacobian_matrix(self.angles).flatten()
        # pub_target_position = Float64MultiArray()
        # pub_target_position.data = self.target_position
        min_angles_2 = self.minimising_angles_without2()
        pub_angles = Float64MultiArray()
        pub_angles.data = min_angles_2.x
        pub_end_effector = Float64MultiArray()
        pub_end_effector.data = self.joint_positions[3] #* self.pixel_to_meter()
        pub_jacobian = Float64MultiArray()
        pub_jacobian.data = min_angles_2.jac.flatten() #* self.pixel_to_meter()
        pub_target_position = Float64MultiArray()
        pub_target_position.data = self.target_position #* self.pixel_to_meter()
        pub_plot_target = Float64MultiArray()
        pub_plot_target.data = (self.target_position - self.joint_positions[0]) * self.pixel_to_meter()
        pub_plot_end_ef = Float64MultiArray()
        pub_plot_end_ef.data = (self.joint_positions[3] - self.joint_positions[0]) * self.pixel_to_meter()
        # publish
        self.target_publisher.publish(pub_target_position)
        self.jacobian_publisher.publish(pub_jacobian)
        self.end_effector_publisher.publish(pub_end_effector)
        self.angles_publisher.publish(pub_angles)
        self.plot_target.publish(pub_plot_target)
        self.plot_end_ef.publish(pub_plot_end_ef)


def main(args):
    ic = ProcessImages()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
