#!/usr/bin/env python

# Node to move the end effector of the robot using the keyboard


import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
import argparse
from control_utils.eef_velocity_controller import SawyerEEFVelocityController
import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION
from control_utils import transform_utils as T
import numpy as np




class ManualPositioning(object):
    def __init__(self, ):
        rospy.on_shutdown(self.clean_shutdown)

        self._rate = 500.0
        self._right_arm = intera_interface.limb.Limb("right")
        self._right_joint_names = self._right_arm.joint_names()

        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        self.controller = SawyerEEFVelocityController()

        self._right_arm.move_to_neutral(speed = 0.1)

        # Getting the base rotation in eef for coordinate frame transformations
        eef_pose_in_base = self._right_arm.endpoint_pose()
        self.eef_quat_in_base = np.array([eef_pose_in_base['orientation'].x,
                                          eef_pose_in_base['orientation'].y,
                                          eef_pose_in_base['orientation'].z,
                                          eef_pose_in_base['orientation'].w, ])

        self.eef_rot_in_base = T.quat2mat(self.eef_quat_in_base)
        self.base_rot_in_eef = self.eef_rot_in_base.T

    def get_right_hand_quat(self):
        eef_pose = self._right_arm.endpoint_pose()
        return np.array([eef_pose['orientation'].x,
                         eef_pose['orientation'].y,
                         eef_pose['orientation'].z,
                         eef_pose['orientation'].w, ])

    def get_right_hand_pos(self):
        eef_pose = self._right_arm.endpoint_pose()
        return np.array([eef_pose['position'].x,
                         eef_pose['position'].y,
                         eef_pose['position'].z, ])

    def get_control(self, action, max_action=0.001, safety_max=0.1):
        action = max_action * action
        current_joint_angles = [self._right_arm.joint_angles()['right_j{}'.format(i)] for i in range(7)]

        # get x,y,z velocity in base frame
        action_in_base = self.base_rot_in_eef.dot(action)
        action_in_base = np.clip(action_in_base, -safety_max, safety_max)

        # Correct for any change in orientation using a proportional controller
        current_right_hand_quat = self.get_right_hand_quat()
        reference_right_hand_quat = self.eef_quat_in_base
        orn_diff = T.quat_multiply(reference_right_hand_quat, T.quat_inverse(current_right_hand_quat))

        orn_diff_mat = T.quat2mat(orn_diff)
        orn_diff_twice = orn_diff_mat.dot(orn_diff_mat)

        # Construct pose matrix
        pose_matrix = np.zeros((4, 4))
        pose_matrix[:3, :3] = orn_diff_twice
        pose_matrix[:3, 3] = action_in_base
        pose_matrix[3, 3] = 1

        # pose_matrix = T.pose2mat((action_in_base, orn_diff))
        joint_velocities = self.controller.compute_joint_velocities_for_endpoint_velocity(pose_matrix,
                                                                                          current_joint_angles)
        j_v = {}
        for i in range(7):
            key = 'right_j{}'.format(i)
            j_v[key] = joint_velocities[i]
        return j_v

    def move_eef(self):
        action_size = 0.05
        rate = rospy.Rate(100)
        action = np.array([0.0, 0.0, 0.0])

        print('Move the end-effector  and press d for done')
        print('''\r **Ready. Do one of the following.
                        8 : go forwards in x
                        2 : go backwards in x
                        6 : go forwards in y
                        4 : go backwards in y
                        + : go forwards in z
                        - : go backwards in z
                        h : double speed
                        l : half speed
                        d : finish moving
                        esc : quit
                        ''')
        while not rospy.is_shutdown():
            c = intera_external_devices.getch()

            if c:
                if (c in ['8']):
                    action = np.array([1.0, 0.0, 0.0])
                elif (c in ['2']):
                    action = np.array([-1.0, 0.0, 0.0])
                elif (c in ['6']):
                    action = np.array([0.0, 1.0, 0.0])
                elif (c in ['4']):
                    action = np.array([0.0, -1.0, 0.0])
                elif (c in ['+']):
                    action = np.array([0.0, 0.0, 1.0])
                elif (c in ['-']):
                    action = np.array([0.0, 0.0, -1.0])
                elif (c in ['5']):
                    action = np.array([0.0, 0.0, 0.0])
                elif (c in ['h']):
                    action = np.array([0.0, 0.0, 0.0])
                    action_size = 2 * action_size
                elif (c in ['l']):
                    action = np.array([0.0, 0.0, 0.0])
                    action_size = action_size / 2
                elif c in ['\x1b', '\x03', 'f']:
                    action = np.array([0.0, 0.0, 0.0])
                    rospy.signal_shutdown("Shutting Down...")
                elif c in ['d']:
                    rospy.loginfo('Done Moving')
                    rospy.set_param('manual_arm_move', False)
                    print(self._right_arm.endpoint_pose())
                    return

            joint_vels = self.get_control(action, max_action=action_size)

            # Send contgrol to robot
            self._right_arm.set_joint_velocities(joint_vels)
            rate.sleep()
        return

    def clean_shutdown(self):
        print("\nExiting example...")
        # return to normal
        self._reset_control_modes()
        return True

    def _reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._right_arm.exit_control_mode()
            rate.sleep()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--move", type=bool,
        default=False,
        help="Path where the trajectories produced are stored")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('manual_arm_move', anonymous=True)

    # Initialise the manual arm move from launch files
    if (rospy.has_param('~manual_arm_move') and rospy.get_param('~manual_arm_move')):
        rospy.set_param('manual_arm_move', True)

    # Initialise the manual arm move from the rosrun command
    if(args.move):
        rospy.set_param('manual_arm_move', True)


    control = ManualPositioning()

    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        if (rospy.has_param('manual_arm_move') and rospy.get_param('manual_arm_move')):
            control.move_eef()

        rate.sleep()


if __name__ == '__main__':
    main()
