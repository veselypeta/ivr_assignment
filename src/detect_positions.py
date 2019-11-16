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


    def forward_kinematics(self,angles,arms):
        [a1,a2,a3,a4] = angles
        [arm1,arm2,arm3] = arms
        M1 = np.array([
            [np.cos(a1),-np.sin(a1),0,0],
            [np.sin(a1),np.cos(a1),0,0],
            [0,0,1,arm2],
            [0,0,0,1]
        ])
        
        M2 = np.array([
            [1,0,0,0],
            [0, np.cos(a2), -np.sin(a2),0],
            [0, np.sin(a2),np.cos(a2),0],
            [0,0,0,1]
        ])
        
        M3 = np.array([
            [np.cos(a3),0, np.sin(a3),np.sin(a3)*arm2],
            [0,1,0,0],
            [-np.sin(a3),0,np.cos(a2),np.cos(a3)*arm2],
            [0,0,0,1]
        ])
        
        M4 = np.array([
            [1,0,0,0],
            [0, np.cos(a4), -np.sin(a4),-np.sin(a4)*arm3],
            [0, np.sin(a4),np.cos(a4),np.cos(a4)*arm3],
            [0,0,0,1]
        ])
        
        O = np.array([0,0,0,1])
        Y = O
        B = np.dot(M1,O)
        G = np.dot(np.matmul(np.matmul(M1,M2),M3),O)
        R = np.dot(np.matmul(np.matmul(np.matmul(M1,M2),M3),M4),O)
        
        return np.matmul(np.matmul(M1,M2),M3)[:3,:3]
        
        

    def detect_joint_angles(self):
        [yellow, blue, green, red] = self.joint_positions
        a = self.pixel2meter()
        blue = blue - yellow
        green = green - yellow
        red = red - yellow
        yellow = yellow - yellow

        arm1 = np.linalg.norm(blue-yellow)
        arm3 = np.linalg.norm(red-green)
        arm2 = np.linalg.norm(green-blue)
        
        Q = (red-green) 
        P = (green-blue)
        
        Z = P/np.linalg.norm(P)
        X = np.cross(P, Q)/np.linalg.norm(np.cross(P, Q))
        Y = np.cross(Z, X)
        
        XYZ = [X, Y, Z]
        
        
        a1 = np.arctan2(-Y[0],Y[1])
        a2 = np.arctan2(Y[2],Y[1]/np.cos(a1))
        a3 = np.arctan2(-X[2],Z[2])
        a4 = np.arccos(np.dot(P, Q)/np.linalg.norm(P)/np.linalg.norm(Q))
        
         
        R1_3 = np.array([
            [np.cos(a1)*np.cos(a3)-np.sin(a1)*np.sin(a2)*np.sin(a3), -np.sin(a1)*np.cos(a2), np.cos(a1)*np.sin(a3)+np.sin(a1)*np.sin(a2)*np.cos(a3)],
            [np.sin(a1)*np.cos(a3)+np.cos(a1)*np.sin(a2)*np.sin(a3), np.cos(a1)*np.cos(a2), np.sin(a1)*np.sin(a3)-np.cos(a1)*np.sin(a2)*np.cos(a3)],
            [-np.cos(a2)*np.sin(a3), np.sin(a2), np.cos(a2)*np.cos(a3)]
            ])
        
        Z_x_error = np.absolute(np.absolute(Z[0]) - np.absolute(R1_3[0,2]))
       
        XYZ_ac = self.forward_kinematics([a1,a2,a3,a4],[arm1,arm2,arm3])

        threshold = 0.3

        if Z_x_error > threshold:
            X = -X
            Y = np.cross(Z,X)
            a1 = np.arctan2(-Y[0],Y[1])
            a2 = np.arctan2(Y[2],Y[1]/np.cos(a1))
            a3 = np.arctan2(-X[2],Z[2])
            a4 = -a4
            
      
        joint_angles = np.array([a1,a2,a3,a4])
        print(joint_angles)
        self.angles = joint_angles


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
        # fk = self.forward_kinematics()
        # print(fk, self.joint_positions[3] - self.joint_positions[0])
        #print(self.angles)


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
