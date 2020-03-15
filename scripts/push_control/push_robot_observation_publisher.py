#!/usr/bin/env python

import sys

sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy

import numpy as np
import intera_external_devices

from control_utils import transform_utils as T
import intera_interface
from intera_interface import CHECK_VERSION
from sim2real_dynamics_sawyer.msg import pushing_robot_observations
from control_utils.eef_velocity_controller import SawyerEEFVelocityController
from sim2real_dynamics_sawyer.msg import pushing_object_observations, pushing_reset


class RobotObservationGenerator(object):
    def __init__(self, goal_pos_in_eefg=np.array([0., 0., 0.12]), publication_rate = 30, predefined_goal = 1):
        '''
        Creates observations for the robot state.
        eefg corresponds to the frame of the end effector when the gripper tip is on top of the goal.
        eef corresponds to the frame of the end effector at any given moment.
        The end effector positions are given in the eefg frame.
        :param goal_pos_in_eefg: Accounts for the offset between the tip of the gripper and the end effector (wrist).
        :param publication_rate: The rate of publication of robot observations
        :param predefined_goal: Use one of the hard coded goal positions to avoid placing the end effector at the goal by hand.
        '''
        rospy.on_shutdown(self.clean_shutdown)

        self.operational = False

        # Predifined, hard-coded goal positions for fast switching between pre-defined goals.
        #  These vectors correspond to the different positions of eefg in the base frame.
        self.predefined_goals = {'goal_1': np.array([   0.4427022010704967,0.31099539625258815,  -0.0715500640533782]),
                                 'goal_2': np.array([0.5430102763875416,0.2695079683642847, -0.07240877305735312]),
                                 'goal_3': np.array([ 0.33322457544607487,  0.3301411776091693,  -0.0714909431185011])}

        self._rate = 500.0
        self._right_arm = intera_interface.limb.Limb("right")
        self._right_joint_names = self._right_arm.joint_names()

        rospy.loginfo("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        rospy.loginfo("Enabling robot... ")
        self._rs.enable()

        self.controller = SawyerEEFVelocityController()

        self._right_arm.move_to_neutral(speed=0.1)

        #Subscribers
        self.reset_subscriber = rospy.Subscriber('pushing/reset', pushing_reset, self.reset_callback, queue_size=1)

        # Get the orientation of eef and eefg
        # The orientation of the end effector is constant, so eefg and eef have the same orientation.
        eef_pose_in_base = self._right_arm.endpoint_pose()
        self.eef_quat_in_base = np.array([eef_pose_in_base['orientation'].x,
                                          eef_pose_in_base['orientation'].y,
                                          eef_pose_in_base['orientation'].z,
                                          eef_pose_in_base['orientation'].w, ])

        self.eef_rot_in_base = T.quat2mat(self.eef_quat_in_base)
        self.base_rot_in_eef = self.eef_rot_in_base.T


        self.publication_rate = publication_rate
        self.goal_pos_in_eefg = goal_pos_in_eefg


        self._finish_initialisation(predefined_goal = predefined_goal)

    def reset_callback(self, res):
        ''' Calls reset when the reset flag is published in the reset topic.'''
        if (res.reset):
            if (res.use_predefined):
                self.reset(res.goal_idx)
            else:
                self.reset()

    def reset(self, goal_idx=None):
        ''' Reset by changing the eefg position (changes the location of the goal). If a goal index is
        present in the reset message, the corresponding pre-defined goal is used to set eefg. Otherwise, the goal
        is defined manually by placing the gripper tip at the goal.'''
        self.operational = False
        rospy.loginfo('Resetting push robot observation publisher')
        rospy.set_param('pushing/robot_observations_initialised', False)
        if (goal_idx is None):
            rospy.delete_param('pushing/eefg_position')
            self._finish_initialisation()
        else:
            self._finish_initialisation(goal_idx)

    def set_predefined_goal(self, goal_idx):
        '''Store the eefg position for some pre-defined goal position. The available pre-defined eefg positions
        are hard-coded in the initialiser.'''
        goal = 'goal_{}'.format(goal_idx)

        rospy.set_param('pushing/eefg_position', self.predefined_goals[goal].tolist())

        return  self.predefined_goals[goal]

    def _finish_initialisation(self, predefined_goal = None):
        '''Record the eefg position in base (the orientation is constant and equal to the orientation of eef).
        If one of the predefined goals is used, just look up the position form the hard-coded positions. Otherwise,
        place the tip of the gripper at the desired goal, and record the corresponding eefg position.'''
        if(predefined_goal is None):
            self.eefg_pos_in_base = self.record_eefg_pos()
        else:
            self.eefg_pos_in_base = self.set_predefined_goal(predefined_goal)

        self.base_posemat_in_eefg = self._get_base_posemat_in_eefg(self.eefg_pos_in_base, self.eef_quat_in_base)
        self._right_arm.move_to_neutral(speed=0.15)
        self.operational= True
        rospy.set_param('pushing/robot_observations_initialised', True)

    def clean_shutdown(self):
        rospy.loginfo("\nExiting example...")
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


    def record_eefg_pos(self):
        '''Record the position of eefg manually by placing the tip of the gripper at the goal location using the keyboard.'''
        if (rospy.has_param('pushing/eefg_position')):
            return np.array(rospy.get_param('pushing/eefg_position'))
        else:
            rospy.loginfo('Place the enf effector at the goal position and press d...')
            rospy.set_param('manual_arm_move', True)

            rate = rospy.Rate(20)
            while True:
                if(not rospy.get_param('manual_arm_move')):
                    return self.done_moving()
                rate.sleep()

    def done_moving(self):
        ''' When the tip of the gripper is at the goal location, set the eefg position.'''
        rospy.loginfo('Recording goal pos')
        eef_pose = self._right_arm.endpoint_pose()
        eefg_pos_in_base = np.array([eef_pose['position'].x,
                                     eef_pose['position'].y,
                                     eef_pose['position'].z,
                                     ])
        rospy.set_param('pushing/eefg_position', eefg_pos_in_base.tolist())

        return eefg_pos_in_base


    def _get_base_posemat_in_eefg(self, eefg_pos_in_base, eefg_quat_in_base):
        eefg_posemat_in_base = T.pose2mat((eefg_pos_in_base,eefg_quat_in_base))
        return T.pose_inv(eefg_posemat_in_base)

    def _from_pos_in_base_to_pos_in_eefg(self, pos_in_base_h):
        temp = self.base_posemat_in_eefg.dot(pos_in_base_h)
        return (temp / temp[3])[:3]

    def get_robot_observation(self):
        ''' Get the end effector position and velocity, and the goal position in the eefg frame'''
        eef_pose_in_base = self._right_arm.endpoint_pose()
        eef_vel_in_eefg = self._right_arm.endpoint_velocity()

        eef_vel_in_eefg = np.array([eef_vel_in_eefg['linear'].x,
                                    eef_vel_in_eefg['linear'].y,
                                    eef_vel_in_eefg['linear'].z
                                    ])

        eef_pos_in_base_h = np.array([eef_pose_in_base['position'].x,
                                      eef_pose_in_base['position'].y,
                                      eef_pose_in_base['position'].z,
                                      1.,
                                      ])


        eef_pos_in_eefg = self._from_pos_in_base_to_pos_in_eefg(eef_pos_in_base_h)


        return {'eef_pos_in_eefg': eef_pos_in_eefg,
                'eef_vel_in_eefg': eef_vel_in_eefg,
                'goal_pos_in_eefg': self.goal_pos_in_eefg,
                }



def main():
    rospy.init_node('robot_observation_node', anonymous=True)

    obs_generator = RobotObservationGenerator()


    observation_publisher = rospy.Publisher('robot_observations', pushing_robot_observations,
                                              queue_size=30)

    rate = rospy.Rate(obs_generator.publication_rate)

    while not rospy.is_shutdown():
        if(obs_generator.operational):
            o = obs_generator.get_robot_observation()

            obs_msg = pushing_robot_observations()
            obs_msg.eef_pos_in_eefg.x = o['eef_pos_in_eefg'][0]
            obs_msg.eef_pos_in_eefg.y = o['eef_pos_in_eefg'][1]
            obs_msg.eef_pos_in_eefg.z = o['eef_pos_in_eefg'][2]
            obs_msg.eef_vel_in_eefg.x = o['eef_vel_in_eefg'][0]
            obs_msg.eef_vel_in_eefg.y = o['eef_vel_in_eefg'][1]
            obs_msg.eef_vel_in_eefg.z = o['eef_vel_in_eefg'][2]
            obs_msg.goal_pos_in_eefg.x = o['goal_pos_in_eefg'][0]
            obs_msg.goal_pos_in_eefg.y = o['goal_pos_in_eefg'][1]
            obs_msg.goal_pos_in_eefg.z = o['goal_pos_in_eefg'][2]

            observation_publisher.publish(obs_msg)

        rate.sleep()




if __name__ == '__main__':
    main()
