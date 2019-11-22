#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray, Float64
import message_filters
from sympy import *


class control:

    # Defines publisher and subscriber
    def __init__(self):
        rospy.init_node('closed_control', anonymous=True)

        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
        
        #publish end effector position
        self.endef_x = rospy.Publisher("/endef_x", Float64, queue_size=10)
        self.endef_y = rospy.Publisher("/endef_y", Float64, queue_size=10)
        self.endef_z = rospy.Publisher("/endef_z", Float64, queue_size=10)
        
        # using target position without vision part
        self.target_x = message_filters.Subscriber("/target/x_position_controller/command", Float64)
        self.target_y = message_filters.Subscriber("/target/y_position_controller/command", Float64)
        self.target_z = message_filters.Subscriber("/target/z_position_controller/command", Float64)

        # function to recieve data
        self.end_effector = np.zeros(3)
        self.trajectory = np.zeros(3)
        self.angles = np.zeros(4)
        self.jacobian = np.zeros([3, 4], dtype='float64')
        
        # using angles, jacobian and end effector position from vision part
        # self.trajectory_sub = message_filters.Subscriber("target_topic", Float64MultiArray)
        # self.end_effector_sub = message_filters.Subscriber("end_effector_topic", Float64MultiArray)
        # self.angles_sub = message_filters.Subscriber("angles_topic", Float64MultiArray)
        # self.jacobian_sub = message_filters.Subscriber("jacobian_topic", Float64MultiArray)

        self.time_trajectory = rospy.get_time()
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.error = np.array([0, 0, 0], dtype='float64')
        self.error_d = np.array([0, 0, 0], dtype='float64')
        self.total_error = 0

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.target_x, self.target_y, self.target_z], 10, 10,
            allow_headerless=True)
        ts.registerCallback(self.callback)

    def callback(self, x, y, z):
        # ----- calculate the position of the end effector from the angles -----
        fk = self.FK()
        FK_row1 = fk[0]
        FK_row2 = fk[1]
        FK_row3 = fk[2]
        a, b, c, d = symbols('a b c d', real=True)
        a1, a2, a3, a4 = self.angles
        subst = [(a, a1), (b, a2), (c, a3), (d, a4)]
        self.end_effector = np.array([
            FK_row1.subs(subst).evalf(),
            FK_row2.subs(subst).evalf(),
            FK_row3.subs(subst).evalf()
        ])
        print("-----")
        print(self.end_effector)

        self.trajectory = np.array([x.data, y.data, z.data])
        print(self.trajectory)

        #test_angles = np.array([0.6, 1.1, 0.9, 1.0])
        #print(" --------------------------------------------------------------------------------- ")
        #print("Angles Used => 1 = {} rad | 2 = {} rad | 3 = {} rad | 4 = {} rad").format(*tuple(test_angles))
        #print("Measured End-Effector Position   -> " + str(self.end_effector))
        #a, b, c, d = symbols('a b c d', real=True)
        #subst = [(a, test_angles[0]), (b, test_angles[1]), (c, test_angles[2]), (d, test_angles[3])]
        #fk_ef_pos = self.FK().subs(subst).evalf().col(-1)
        #print("Calculated End-Effector Position -> " + str(fk_ef_pos))

        q_d = self.control_closed()
        
        joint0 = Float64()
        joint0.data = q_d[0]
        joint1 = Float64()
        joint1.data = q_d[1]
        joint2 = Float64()
        joint2.data = q_d[2]
        joint3 = Float64()
        joint3.data = q_d[3]
        
        self.robot_joint1_pub.publish(joint0)
        self.robot_joint2_pub.publish(joint1)
        self.robot_joint3_pub.publish(joint2)
        self.robot_joint4_pub.publish(joint3)
        
        ee_x = Float64()
        ee_x.data = self.end_effector[0]
        ee_y = Float64()
        ee_y.data = self.end_effector[1]
        ee_z = Float64()
        ee_z.data = self.end_effector[2]
        
        self.endef_x.publish(ee_x)
        self.endef_y.publish(ee_y)
        self.endef_z.publish(ee_z)

    def control_closed(self):
        # ----- gains -----
        K_p = 5 * np.identity(3)
        K_d = 0.6 * np.identity(3)
        K_i = 0.2 * np.identity(3)

        # ----- time step -----
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time

        # ----- ef pos and trajectory -----
        pos = self.end_effector
        pos_d = self.trajectory

        # ----- error -----
        self.error_d = ((pos_d - pos) - self.error) / dt
        self.error = pos_d - pos
        self.total_error += self.error * dt

        q = self.angles
        J_inv = np.linalg.pinv(self.empty_jacobian(q))
        dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose()) + np.dot(K_i,self.total_error)))
        q_d = q + (dt * dq_d)
        a1 = q_d[0] % np.pi
        q_d = q_d % np.pi/2
        q_d[0] = a1
        self.angles = q_d
        return q_d


    def empty_jacobian(self, x):
        a1, a2, a3, a4 = x

        a, b, c, d = symbols('a b c d', real=True)

        fk = self.FK()
        FK_row1 = fk[0]
        FK_row2 = fk[1]
        FK_row3 = fk[2]

        subst = [(a, a1), (b, a2), (c, a3), (d, a4)]
        jacob = Matrix([[float(diff(FK_row1, a).subs(subst).evalf()), float(diff(FK_row1, b).subs(subst).evalf()),
                         float(diff(FK_row1, c).subs(subst).evalf()), float(diff(FK_row1, d).subs(subst).evalf())],
                        [float(diff(FK_row2, a).subs(subst).evalf()), float(diff(FK_row2, b).subs(subst).evalf()),
                         float(diff(FK_row2, c).subs(subst).evalf()), float(diff(FK_row2, d).subs(subst).evalf())],
                        [float(diff(FK_row3, a).subs(subst).evalf()), float(diff(FK_row3, b).subs(subst).evalf()),
                         float(diff(FK_row3, c).subs(subst).evalf()), float(diff(FK_row3, d).subs(subst).evalf())]])

        return np.array(jacob).astype(np.float64)


    def FK(self):
        a, b, c, d, l = symbols('a b c d l', real=True)

        trans = Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, l],
            [0, 0, 0, 1]
        ])

        x_rot1 = Matrix([
            [1, 0, 0, 0],
            [0, cos(b), -sin(b), 0],
            [0, sin(b), cos(b), 0],
            [0, 0, 0, 1],
        ])

        x_rot2 = Matrix([
            [1, 0, 0, 0],
            [0, cos(d), -sin(d), 0],
            [0, sin(d), cos(d), 0],
            [0, 0, 0, 1],
        ])

        y_rot = Matrix([
            [cos(c), 0, sin(c), 0],
            [0, 1, 0.0, 0],
            [-sin(c), 0.0, cos(c), 0],
            [0, 0, 0, 1]
        ])

        z_rot = Matrix([
            [cos(a), -sin(a), 0.0, 0],
            [sin(a), cos(a), 0.0, 0],
            [0.0, 0.0, 1.0, 0],
            [0, 0, 0, 1]
        ])

        return (z_rot * trans.subs(l, 2) * x_rot1 * y_rot * trans.subs(l, 3) * x_rot2 * trans.subs(l, 2)).col(-1)



def main(args):
    ic = control()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main(sys.argv)
