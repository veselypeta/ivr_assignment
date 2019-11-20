#!/usr/bin/env python

import rospy
rom std_msgs.msg import Float64MultiArray, Float64
import numpy as np
import cv2
import sys
from detect_centres import *
import message_filters


class Result:

    def __init__(self):

        self.j1_sub = rospy.Subscriber('/robot/joint1_position_controller/command', Float64)
        self.j2sub = rospy.Subscriber('/robot/joint2_position_controller/command', Float64)
        self.j3_sub = rospy.Subscriber('/robot/joint3_position_controller/command', Float64)
        self.j4_sub = rospy.Subscriber('/robot/joint4_position_controller/command', Float64)

        self.fk_pub = rospy.Publisher('fk_topic', Float64MultiArray, queue_size=10)

        ts = message_filters.ApproximateTimeSynchronizer([self.j1_sub, self.j2sub, self.j3_sub, self.j4_sub,
                                                          ], 10, 10, allow_headerless=True)
        ts.registerCallback(self.callback)


    def callback(self, j1, j2, j3, j4):
        self.joints = np.array([j1, j2, j3, j4])
        self.update()



    def working_forward_kinematics(self, x):
        a1, a2, a3, a4 = x

        arm1len = 2
        arm2len = 3
        arm3len = 2

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


    def update(self):
        fk_publish = Float64MultiArray()
        fk_publish.data = self.working_forward_kinematics(self.joints)
        self.fk_pub.Publish(fk_publish)




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


