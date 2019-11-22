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
        [0, 0, 1, arm1],
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


    Q = (red - green)
    P = (green - blue)

    Z = P / np.linalg.norm(P)
    X = np.cross(P, Q) / np.linalg.norm(np.cross(P, Q))
    Y = np.cross(Z, X)

    XYZ = [X, Y, Z]

    # solve : R_z * T_2 * R_x * R_y * T_3 = XYZ  --> state at green joint
    # will create opposite angles if a4 was negative -> because arccos will output positive even if it is negative
    # and then X is -X and Y is -Y so all angles will have opposite sign
    a1 = np.arctan2(-Y[0], Y[1])
    a2 = np.arctan2(Y[2], Y[1] / np.cos(a1))
    a3 = np.arctan2(-X[2], Z[2])
    a4 = np.arccos(np.dot(P, Q) / np.linalg.norm(P) / np.linalg.norm(Q))

    joint_angles = np.array([a1, a2, a3, a4])
    self.angles = joint_angles
