#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64, String
from cv_bridge import CvBridge, CvBridgeError
import time
import message_filters


class control:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('closed_control', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1", Image, queue_size=10)
        self.joints_pub = rospy.Publisher("joints_pos_image_1", Float64MultiArray, queue_size=10)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback
        # function to recieve data
        self.end_effector = np.zeros(3)
        self.trajectory = np.zeros(3)
        self.angles = np.zeros(4)
        self.jacobian = np.zeros([3,4])
        self.trajectory_sub = message_filters.Subscriber("target_topic", Float64MultiArray)
        self.end_effector_sub = message_filters.Subscriber("end_effector_topic", Float64MultiArray)
        self.angles_sub = message_filters.Subscriber("angles_topic", Float64MultiArray)
        self.jacobian_sub = message_filters.Subscriber("jacobian_topic", Float64MultiArray)
        
        self.time_trajectory = rospy.get_time()
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        self.error = np.array([0,0,0], dtype='float64')
        self.error_d = np.array([0,0,0], dtype='float64')

        ts = message_filters.ApproximateTimeSynchronizer([self.trajectory_sub, self.end_effector_sub, self.angles_sub, self.jacobian_sub], 10, 10, allow_headerless=True)
        ts.registerCallback(self.callback)
        
        
    def callback(self, trajectory, end_effector, angles, jacobian):
        # update values
        self.trajectory = np.array(trajectory.data)
        self.end_effector = np.array(end_effector.data)
        self.angles = np.array(angles.data)
        self.jacobian = np.array(jacobian.data).reshape((3,4))

        print(self.control_closed())

    def control_closed(self):
        K_p = 10 * np.identity(3)
        K_d = 0.1 * np.identity(3)
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time
        pos = self.end_effector
        pos_d = self.trajectory
        self.error_d = ((pos_d - pos) - self.error)/dt
        self.error = pos_d - pos
        q = self.angles
        J_inv = np.linalg.pinv(self.jacobian)
        dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose())))
        q_d = q + (dt * dq_d)
        return q_d

      


# call the class
def main(args):
    ic = control()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
