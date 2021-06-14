import numpy as np


class UR5e:
    class DH:
        d1 = 0.1625
        d4 = 0.1333
        d5 = 0.0997
        d6 = 0.0996
        a2 = -0.425
        a3 = -0.3922
        # d6 += (0.09 + 0.054)
    position = np.array([-0.422, 0.7728171, 0.2028])
    effector_transform = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.09],
            [0, 0, 0, 1],
        ]
    )

    @staticmethod
    def effector_end(joint_angles):
        transformation = UR5e.forward_kinematics(joint_angles)
        end_transformation = np.matmul(transformation, UR5e.effector_transform)
        position = end_transformation[0:3, 3]
        # print(position)
        # print("-0.422, 1.852217, 0.5260299")
        new_coords = [
            UR5e.position[0] + position[1],
            UR5e.position[1] + position[2],
            UR5e.position[2] - position[0]
        ]
        return new_coords

    @staticmethod
    def forward_kinematics(joint_angles):
        transformation = np.zeros((4, 4))
        # print(transformation)
        transformation[3, 3] = 1
        # print(transformation)
        # print(UR5e.position.shape)
        # print(UR5e.position)
        t1 = - joint_angles[0] - 90.0
        t2 = - joint_angles[1] - 90.0
        t3 = - joint_angles[2]
        t4 = - joint_angles[3] - 90.0
        t5 = - joint_angles[4]
        t6 = - joint_angles[5]

        s1 = np.sin(np.deg2rad(t1))
        s2 = np.sin(np.deg2rad(t2))
        s3 = np.sin(np.deg2rad(t3))
        s4 = np.sin(np.deg2rad(t4))
        s5 = np.sin(np.deg2rad(t5))
        s6 = np.sin(np.deg2rad(t6))
        c1 = np.cos(np.deg2rad(t1))
        c2 = np.cos(np.deg2rad(t2))
        c3 = np.cos(np.deg2rad(t3))
        c4 = np.cos(np.deg2rad(t4))
        c5 = np.cos(np.deg2rad(t5))
        c6 = np.cos(np.deg2rad(t6))

        # print(s1, c1, s2, c2, s3, c3)

        S34 = c3 * c4 - s3 * s4
        C34 = c3 * s4 + s3 * c4

        c234 = c2 * S34 - s2 * C34
        s234 = s2 * S34 + c2 * C34

        SC2346 = s6 * c234
        SS2346 = s6 * s234
        CS2346 = c6 * s234
        CC2346 = c6 * c234

        transformation[1, 0] = c5 * CC2346 - SS2346
        transformation[0, 0] = c1 * transformation[1, 0] + s1 * s5 * c6
        transformation[1, 0] = s1 * transformation[1, 0] - c1 * s5 * c6
        transformation[2, 0] = c5 * CS2346 + SC2346

        transformation[1, 1] = c5 * SC2346 + CS2346
        transformation[0, 1] = - c1 * transformation[1, 1] - s1 * s5 * s6
        transformation[1, 1] = - s1 * transformation[1, 1] + c1 * s5 * s6
        transformation[2, 1] = - c5 * SS2346 + CC2346

        transformation[0, 2] = - c1 * s5 * c234 + s1 * c5
        transformation[1, 2] = - s1 * s5 * c234 - c1 * c5
        transformation[2, 2] = - s5 * s234

        transformation[1, 3] = UR5e.DH.d5 * s234 - s5 * UR5e.DH.d6 * c234
        transformation[0, 3] = (c1 * transformation[1, 3] + s1 * c5 * UR5e.DH.d6
                                + UR5e.DH.a3 * c1 * (c2 * c3 - s2 * s3)
                                + UR5e.DH.a2 * c1 * c2 + UR5e.DH.d4 * s1)
        transformation[1, 3] = (s1 * transformation[1, 3] - c1 * c5 * UR5e.DH.d6
                                + UR5e.DH.a3 * s1 * (c2 * c3 - s2 * s3)
                                + UR5e.DH.a2 * s1 * c2 - UR5e.DH.d4 * c1)

        transformation[2, 3] = (UR5e.DH.d1 + UR5e.DH.a2 * s2 + UR5e.DH.a3 * (s2 * c3 + c2 * s3)
                               - UR5e.DH.d5 * c234 + UR5e.DH.d6 * transformation[2, 2])

        return transformation

