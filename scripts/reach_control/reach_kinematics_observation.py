import sys

sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy

import numpy as np
from control_utils import transform_utils as T
from control_utils.eef_velocity_controller import SawyerEEFVelocityController





class Observation(object):
    def __init__(self, right_arm, reset_goal_pos = False, predefined_goal = None ):
        '''
        Creates observations for the Reaching task
        :param right_arm: The intera link object to control the robot.
        :param reset_goal_pos: Whether to reset any pre-existing goal position
        :param predefined_goal:  Use one of the hard coded goal positions to avoid placing the end effector at the goal by hand.
        '''
        self._right_arm = right_arm
        self._joint_names = self._right_arm.joint_names()

        self.previous_joint_vels_array = np.zeros((7,))

        self._right_arm.move_to_neutral(speed=0.1)

        # Getting the base rotation in eef for coordinate frame transformations
        eef_pose_in_base = self._right_arm.endpoint_pose()
        self.eef_quat_in_base = np.array([eef_pose_in_base['orientation'].x,
                                     eef_pose_in_base['orientation'].y,
                                     eef_pose_in_base['orientation'].z,
                                     eef_pose_in_base['orientation'].w, ])

        self.eef_rot_in_base = T.quat2mat( self.eef_quat_in_base)
        self.base_rot_in_eef = self.eef_rot_in_base.T

        self.controller = SawyerEEFVelocityController()

        # Predifined, hard-coded, goal positions for fast switching between pre-defined goals.
        #  These vectors correspond to the different positions of eefg in the base frame.
        self.predefined_goals = {'goal_1': np.array([ 0.44889998342216186,0.15767033384085796, -0.07223470331644867]),
                                'goal_2': np.array([0.4064366403627939,0.2151227875993398,-0.07152062349980176]),
                                'goal_3': np.array([0.5353834307301527,0.0993579385905306,  -0.072644653035373])}

        if(predefined_goal is None):
            self.goal_pos_in_base=self.record_goal_pos(reset_goal_pos)
        elif predefined_goal in self.predefined_goals:
            self.set_predefined_goal(predefined_goal)
        else:
            raise NotImplementedError()

    def reset(self, specific_goal = None):
        if(specific_goal  is None):
            self.goal_pos_in_base = self.record_goal_pos(True)
        elif (specific_goal in self.predefined_goals):
            self.set_predefined_goal(specific_goal)
        else:
            raise NotImplementedError()

    def set_predefined_goal(self,goal):
        self.goal_pos_in_base = self.predefined_goals[goal]
        rospy.set_param('goal_position',  self.predefined_goals[goal].tolist())

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

    def record_goal_pos(self, reset_goal_pos=False):
        '''Record a goal position manually by placing the end-effector to the desired goal position.'''
        rospy.loginfo('Recording reaching goal position')
        if ( not reset_goal_pos and rospy.has_param('goal_position')):
            return np.array(rospy.get_param('goal_position'))
        else:
            rospy.set_param('manual_arm_move', True)

            rate = rospy.Rate(20)
            while True:
                if(not rospy.get_param('manual_arm_move')):
                    rospy.loginfo('Done')
                    return self.done_moving()
                rate.sleep()

    def done_moving(self):
        eef_pose = self._right_arm.endpoint_pose()
        eefg_pos_in_base = np.array([eef_pose['position'].x,
                                     eef_pose['position'].y,
                                     eef_pose['position'].z,
                                     ])

        rospy.set_param('goal_position', eefg_pos_in_base.tolist())

        return eefg_pos_in_base


    def change_goal_pos(self, dx_in_base, dy_in_base):
        self.goal_pos_in_base[0] += dx_in_base
        self.goal_pos_in_base[1] += dy_in_base

    def get_characterisation_observation(self):
        '''Get a breakdown of the different components of the observation, useful for logging'''
        eef_pose_in_base = self._right_arm.endpoint_pose()
        eef_vel_in_eef = self._right_arm.endpoint_velocity()

        eef_vel_in_eef = np.array([eef_vel_in_eef['linear'].x,
                                   eef_vel_in_eef['linear'].y,
                                   eef_vel_in_eef['linear'].z,
                                   ])

        eef_vel_in_base = self.eef_rot_in_base.dot(eef_vel_in_eef)

        return {'eef_pos_in_base': np.array([eef_pose_in_base['position'].x,
                                             eef_pose_in_base['position'].y,
                                             eef_pose_in_base['position'].z,
                                             ]),
                'eef_vel_in_base': eef_vel_in_base,
                'goal_pos_in_base': self.goal_pos_in_base,
                }

    def get_observation(self):
        '''Get the observation vector input to the policy.'''
        eef_pose_in_base = self._right_arm.endpoint_pose()
        eef_vel_in_eef = self._right_arm.endpoint_velocity()

        eef_pos_in_base = np.array([eef_pose_in_base['position'].x,
                                    eef_pose_in_base['position'].y,
                                    eef_pose_in_base['position'].z,
                                    ])

        eef_to_goal_in_base = self.goal_pos_in_base - eef_pos_in_base

        eef_to_goal_in_eef = self.base_rot_in_eef.dot(eef_to_goal_in_base)

        eef_vel_in_eef = np.array([eef_vel_in_eef['linear'].x,
                                   eef_vel_in_eef['linear'].y,
                                   eef_vel_in_eef['linear'].z,
                                   ])

        return np.concatenate([eef_to_goal_in_eef, eef_vel_in_eef])



