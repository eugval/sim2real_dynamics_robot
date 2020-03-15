#!/usr/bin/env python
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
import math
from control_utils.eef_velocity_controller import SawyerEEFVelocityController
import intera_external_devices
import intera_interface
import os
from intera_interface import CHECK_VERSION
from control_utils import transform_utils as T
import numpy as np
import pickle
import time
import argparse

# Script for executing a set of pre-defined open loop policies.

_POLICIES = [{'policy_type':'joint', 'period':0.4, 'amplitude':0.2, 'line_policy':None , 'steps': 50 },
             {'policy_type':'joint', 'period':0.3, 'amplitude':0.2,'line_policy':None, 'steps': 50 },
             {'policy_type':'joint', 'period':0.5, 'amplitude':0.2,'line_policy':None, 'steps': 50 },
             {'policy_type':'eef_circle', 'period':0.2, 'amplitude':0.05,'line_policy':None, 'steps': 50 },
             {'policy_type':'eef_line',  'period':None, 'amplitude':None,'line_policy': np.array([0.0, 0.0, 0.05]), 'steps': 36 },
             {'policy_type':'eef_line',  'period':None, 'amplitude':None,'line_policy': np.array([0.05, 0.05, 0.05]), 'steps': 36 },
             {'policy_type':'eef_line',  'period':None, 'amplitude':None,'line_policy': np.array([-0.05, 0.05, 0.05]), 'steps': 36 },
             {'policy_type':'eef_line',  'period':None, 'amplitude':None,'line_policy': np.array([0.05, 0.0, 0.0]), 'steps': 36 },
             {'policy_type':'eef_line',  'period':None, 'amplitude':None,'line_policy': np.array([0.0, 0.05, 0.0]), 'steps': 36 },
             {'policy_type':'eef_line',  'period':None, 'amplitude':None,'line_policy': np.array([0.0, -0.05, 0.0]), 'steps': 24 },
             {'policy_type':'eef_line',  'period':None, 'amplitude':None,'line_policy': np.array([-0.05, 0.0, 0.0]), 'steps': 36 },
             {'policy_type':'eef_line',  'period':None, 'amplitude':None,'line_policy': np.array([0.05, 0.05, 0.0]), 'steps': 36 },
             {'policy_type':'eef_line',  'period':None, 'amplitude':None,'line_policy': np.array([-0.05, -0.05, 0.0]), 'steps': 18 },
             ]


class Control(object):
    def __init__(self):
        # Make sure there is a clean shutdown
        rospy.on_shutdown(self.clean_shutdown)
        # Setting up Sawyer
        self._right_arm = intera_interface.limb.Limb("right")
        self._right_joint_names = self._right_arm.joint_names()

        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        self.set_neutral()

        #  end-effector initial pose
        self.eef_pose = self._right_arm.endpoint_pose()
        self.eef_quat = np.array([self.eef_pose['orientation'].x,
                                  self.eef_pose['orientation'].y,
                                  self.eef_pose['orientation'].z,
                                  self.eef_pose['orientation'].w, ])

        self.base_rot_in_eef = T.quat2mat(self.eef_quat).T

        # control parameters
        self._rate = 500.0  # Hz

        # Eef velocity control
        self.controller = SawyerEEFVelocityController()

        #Policy parameters
        self.policy_type = 'joint'
        self.line_policy = np.array([0.,0.,0.])
        self.period_factor = 0.15
        self.amplitude_factor = 0.1

    def set_policy_type(self,policy_type):
        assert policy_type == 'joint' or policy_type == 'eef_circle' or policy_type == "eef_line"

        self.policy_type = policy_type


    def set_period_amplitude(self, period, amplitude):
        if(period is not None):
            self.period_factor = period
        if(amplitude is not None):
            self.amplitude_factor = amplitude


    def set_line_policy(self, line_policy):
        if(line_policy is not None):
            self.line_policy= line_policy

    def cos_wave(self, elapsed):
        w = self.period_factor * elapsed
        return self.amplitude_factor * math.cos(w * 2 * math.pi)

    def sin_wave(self, elapsed):
        w = self.period_factor * elapsed
        return self.amplitude_factor * math.sin(w * 2 * math.pi)

    def get_policy(self, time, theshold_time = -1.):
        if(self.policy_type == 'joint' ):
            return np.array([self.sin_wave(time) for _ in range(7)])

        elif (self.policy_type == 'eef_circle'):
            return np.array([self.sin_wave(time),self.cos_wave(time),0.0])

        elif(self.policy_type == 'eef_line'):

            if(time<theshold_time):
                return self.line_policy
            else:
                return np.array([0.,0.,0.])



    def _reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._right_arm.exit_control_mode()
            rate.sleep()
        return True

    def get_right_hand_quat(self):
        eef_pose = self._right_arm.endpoint_pose()
        return np.array([eef_pose['orientation'].x,
                         eef_pose['orientation'].y,
                         eef_pose['orientation'].z,
                         eef_pose['orientation'].w, ])

    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        self._right_arm.move_to_neutral(speed = 0.1)

    def clean_shutdown(self):
        print("\nExiting example...")
        # return to normal
        self._reset_control_modes()
        self.set_neutral()
        return True



    def get_control(self, action, safety_max=0.1):
        current_joint_angles = [self._right_arm.joint_angles()['right_j{}'.format(i)] for i in range(7)]

        # get x,y,z velocity in base frame
        action_in_base = self.base_rot_in_eef.dot(action)
        action_in_base = np.clip(action_in_base,-safety_max,safety_max)

        # Correct for any change in orientation using a proportional controller
        current_right_hand_quat = self.get_right_hand_quat()
        reference_right_hand_quat = self.eef_quat
        orn_diff = T.quat_multiply(reference_right_hand_quat, T.quat_inverse(current_right_hand_quat))

        orn_diff_mat = T.quat2mat(orn_diff)
        orn_diff_twice = orn_diff_mat.dot(orn_diff_mat)

        # Construct pose matrix
        pose_matrix = np.zeros((4,4))
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

    def callback(self, data):
        self.control = data.data

        rospy.loginfo(type(self.control))
        rospy.loginfo(self.control)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--log_path", type= str,
        default='../logs/internal_dynamics_calibration/trajectories.pckl',
        help="Path where the trajectories produced are stored")
    args = parser.parse_args(rospy.myargv()[1:])
    log_path = args.log_path

    rospy.init_node('calibrate_internal_dynamics', anonymous=True)

    # Controller
    controller = Control()

    done = False
    print('Ready. Press any key to start.')
    while not done and not rospy.is_shutdown():
        c = intera_external_devices.getch()
        if c:
            done = True


    # Set control rate
    rate = rospy.Rate(10)

    trajectories_gathered =[]

    # Time and iteration count
    for policy_id, policy_params in enumerate(_POLICIES):
        if(rospy.is_shutdown()):
            break

        trajectories_gathered.append([])

        controller.set_neutral()
        time.sleep(2)

        controller.set_policy_type(policy_params['policy_type'])
        controller.set_period_amplitude(policy_params['period'], policy_params['amplitude'])
        controller.set_line_policy(policy_params['line_policy'])

        steps = policy_params['steps']
        start = rospy.Time.now()
        for i in range(steps):
            if (rospy.is_shutdown()):
                break
            elapsed = rospy.Time.now() - start

            # Get observation
            eef_pose_in_base = controller._right_arm.endpoint_pose()
            eef_pos_in_base =  np.array([eef_pose_in_base['position'].x,
                                             eef_pose_in_base['position'].y,
                                             eef_pose_in_base['position'].z,
                                             ])

            eef_quat_in_base = np.array([eef_pose_in_base['orientation'].x,
                                              eef_pose_in_base['orientation'].y,
                                              eef_pose_in_base['orientation'].z,
                                              eef_pose_in_base['orientation'].w, ])

            eef_rot_in_base = T.quat2mat(eef_quat_in_base)

            eef_vel_in_eef = controller._right_arm.endpoint_velocity()
            eef_vel_in_eef = np.array([eef_vel_in_eef['linear'].x,
                                       eef_vel_in_eef['linear'].y,
                                       eef_vel_in_eef['linear'].z,
                                       ])
            eef_vel_in_base = eef_rot_in_base.dot(eef_vel_in_eef)

            obs_joint_pos = controller._right_arm.joint_angles()
            obs_joint_pos = np.array([obs_joint_pos['right_j0'], obs_joint_pos['right_j1'],
                                      obs_joint_pos['right_j2'], obs_joint_pos['right_j3'],
                                      obs_joint_pos['right_j4'], obs_joint_pos['right_j5'],
                                      obs_joint_pos['right_j6']])
            obs_joint_vels = controller._right_arm.joint_velocities()
            obs_joint_vels = np.array([obs_joint_vels['right_j0'], obs_joint_vels['right_j1'],
                                       obs_joint_vels['right_j2'],obs_joint_vels['right_j3'],
                                       obs_joint_vels['right_j4'], obs_joint_vels['right_j5'],
                                       obs_joint_vels['right_j6'],])

            # Find action
            action = controller.get_policy(elapsed.to_sec(), theshold_time=steps *(5./ 60.))

            # Get control
            if (controller.policy_type=='joint'):
                joint_vels = dict(
                    [('right_j{}'.format(idx), action[idx]) for idx in range(7)])
            else:
                joint_vels = controller.get_control(action)
            # Send contgrol to robot
            controller._right_arm.set_joint_velocities(joint_vels)

            # Record
            trajectories_gathered[-1].append([eef_pos_in_base, eef_vel_in_base,
                                              obs_joint_pos,
                                              obs_joint_vels])

            rate.sleep()


    #pickle results
    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            log_path)
    pickle.dump(trajectories_gathered, open(os.path.abspath(log_path),'wb'))


    rospy.signal_shutdown('Done')


if __name__ == '__main__':
    main()
