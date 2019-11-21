#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray, Float64
import numpy as np
import cv2
import sys
from detect_centres import *
import message_filters


class Result:

    def __init__(self):
        rospy.init_node("fk_result_calculator", anonymous=True)
        self.j1_sub = message_filters.Subscriber('/robot/joint1_position_controller/command', Float64)
        self.j2sub = message_filters.Subscriber('/robot/joint2_position_controller/command', Float64)
        self.j3_sub = message_filters.Subscriber('/robot/joint3_position_controller/command', Float64)
        self.j4_sub = message_filters.Subscriber('/robot/joint4_position_controller/command', Float64)
        self.act_end_ef = message_filters.Subscriber('plot_end_ef_topic', Float64)

        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        self.fk_pub = rospy.Publisher('fk_topic', Float64MultiArray, queue_size=10)
        self.joints = [0, 0, 0, 0]

        self.publish_data_to_joints()

        ts = message_filters.ApproximateTimeSynchronizer([self.j1_sub, self.j2sub, self.j3_sub, self.j4_sub,
                                                          self.act_end_ef], 10, 10, allow_headerless=True)
        ts.registerCallback(self.callback)

    def callback(self, j1, j2, j3, j4, ef):
        self.joints = np.array([j1.data, j2.data, j3.data, j4.data])
        self.end_ef = ef
        self.update()

    def working_forward_kinematics(self, x):
        a1, a2, a3, a4 = x
        arm1len = 2
        arm2len = 3
        arm3len = 2

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
            [0, np.cos(a2), -np.sin(a2), 0],
            [0, np.sin(a2), np.cos(a2), 0],
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


    def publish_data_to_joints(self):
        j1_ran = Float64
        j1_ran.data = np.random.random(0, np.pi)
        j2_ran = Float64()
        j2_ran.data = np.random.random(0, np.pi)
        j3_ran = Float64()
        j3_ran.data = np.random.random(0, np.pi / 2)
        j4_ran = Float64()
        j4_ran.data = np.random.random(0, np.pi / 2)

        self.robot_joint1_pub.publish(j1_ran)
        self.robot_joint2_pub.publish(j2_ran)
        self.robot_joint3_pub.publish(j3_ran)
        self.robot_joint4_pub.publish(j4_ran)

    def update(self):
        fk_publish = Float64MultiArray()
        fk_publish.data = self.working_forward_kinematics(self.joints)
        self.fk_pub.publish(fk_publish)
        print(fk_publish.data, self.end_ef)
        self.publish_data_to_joints()


def main(args):
    ic = Result()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
