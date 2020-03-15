#!/usr/bin/env python
import sys

sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy

from control_utils.eef_velocity_controller import SawyerEEFVelocityController
import intera_interface
import os
import intera_external_devices
from intera_interface import CHECK_VERSION
from control_utils import transform_utils as T
from control_utils.logger import Logger
import numpy as np
import argparse
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Bool

from sim2real_dynamics_sawyer.msg import pushing_characterisation_observations, Floats64
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_motion_msgs.msg import TrajectoryOptions
from rospy.numpy_msg import numpy_msg
import copy


class Control(object):
    def __init__(self):
        # Make sure there is a clean shutdown
        rospy.on_shutdown(self.clean_shutdown)

        # control parameters
        self._rate = 500.0  # Hz

        # Setting up the observation subscriber
        self.characterisation_observation = None
        self.observation = None

        self.characterisation_observation_subscriber = rospy.Subscriber('characterisation_observations',
                                                                        pushing_characterisation_observations,
                                                                        self.characterisation_observation_callback,
                                                                        queue_size=1)
        self.observation_subscriber = rospy.Subscriber('observations', numpy_msg(Floats64),
                                                       self.observation_callback,
                                                       queue_size=1)


        while True:
            if (self.characterisation_observation is not None and self.observation is not None):
                break


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
        eef_pose = self._right_arm.endpoint_pose()
        self.start_eef_pos = np.array([eef_pose['position'].x,
                                       eef_pose['position'].y,
                                       eef_pose['position'].z,
                                       ])
        self.start_eef_quat = np.array([eef_pose['orientation'].x,
                                        eef_pose['orientation'].y,
                                        eef_pose['orientation'].z,
                                        eef_pose['orientation'].w, ])

        self.base_rot_in_eef = T.quat2mat(self.start_eef_quat).T

        if (not rospy.has_param('pushing/start_position')):
            self.start_eef_pos, self.start_eef_quat = self.calibrate_initial_position()
        else:
            self.start_eef_pos = np.array(rospy.get_param('pushing/start_position'))
            self.start_eef_quat = np.array(rospy.get_param('pushing/start_orientation'))

        # Eef velocity control
        self.controller = SawyerEEFVelocityController()

        # Sin-cos control
        self.period_factor = 0.15
        self.amplitude_factor = 0.1

    def calibrate_initial_position(self):
        rospy.loginfo('Place the end-effector in the desired starting position and press  d...')

        if (rospy.has_param('pushing/start_position')):
            return np.array(rospy.get_param('pushing/start_position'))
        else:
            rospy.set_param('manual_arm_move', True)

            rate = rospy.Rate(20)
            while True:
                if (not rospy.get_param('manual_arm_move')):
                    rospy.loginfo('Done')
                    return self.done_moving()
                rate.sleep()


    def done_moving(self):
        eef_pose = self._right_arm.endpoint_pose()
        start_eef_pos = np.array([eef_pose['position'].x,
                                       eef_pose['position'].y,
                                       eef_pose['position'].z,
                                       ])
        start_eef_quat = np.array([eef_pose['orientation'].x,
                                   eef_pose['orientation'].y,
                                   eef_pose['orientation'].z,
                                   eef_pose['orientation'].w, ])

        rospy.set_param('pushing/start_position', start_eef_pos.tolist())
        rospy.set_param('pushing/start_orientation', start_eef_quat.tolist())

        return start_eef_pos, start_eef_quat

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

    def get_right_hand_pos(self):
        eef_pose = self._right_arm.endpoint_pose()
        return np.array([eef_pose['position'].x,
                         eef_pose['position'].y,
                         eef_pose['position'].z, ])

    def go_to_pose(self, position, orientation):
        try:
            traj_options = TrajectoryOptions()
            traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
            traj = MotionTrajectory(trajectory_options=traj_options, limb=self._right_arm)

            wpt_opts = MotionWaypointOptions(max_linear_speed=0.6,
                                             max_linear_accel=0.6,
                                             max_rotational_speed=1.57,
                                             max_rotational_accel=1.57,
                                             max_joint_speed_ratio=1.0)
            waypoint = MotionWaypoint(options=wpt_opts.to_msg(), limb=self._right_arm)

            pose = Pose()
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = position[2]
            pose.orientation.x = orientation[0]
            pose.orientation.y = orientation[1]
            pose.orientation.z = orientation[2]
            pose.orientation.w = orientation[0]
            poseStamped = PoseStamped()
            poseStamped.pose = pose
            joint_angles = self._right_arm.joint_ordered_angles()
            waypoint.set_cartesian_pose(poseStamped, "right_hand", joint_angles)

            rospy.loginfo('Sending waypoint: \n%s', waypoint.to_string())

            traj.append_waypoint(waypoint.to_msg())

            result = traj.send_trajectory(timeout=10)
            if result is None:
                rospy.logerr('Trajectory FAILED to send')
                return

            if result.result:
                rospy.loginfo('Motion controller successfully finished the trajectory!')
            else:
                rospy.logerr('Motion controller failed to complete the trajectory with error %s',
                             result.errorId)

        except rospy.ROSInterruptException:
            rospy.logerr('Keyboard interrupt detected from the user. Exiting before trajectory completion.')

    def go_to_start(self):
        self.go_to_pose(self.start_eef_pos, self.start_eef_quat)

    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        self._right_arm.move_to_neutral(speed = 0.15)

    def clean_shutdown(self):
        print("\nExiting example...")
        # return to normal
        self._reset_control_modes()
        self.set_neutral()
        return True

    def get_control(self, action, safety_max = 0.1):
        current_joint_angles = [self._right_arm.joint_angles()['right_j{}'.format(i)] for i in range(7)]

        z_vel = self.get_right_hand_pos()[2] - self.start_eef_pos[2]
        action = np.concatenate([action, [2 * z_vel]])

        # get x,y,z velocity in base frame
        action_in_base = self.base_rot_in_eef.dot(action)
        action_in_base = np.clip(action_in_base,-safety_max,safety_max)

        # Correct for any change in orientation using a proportional controller
        current_right_hand_quat = self.get_right_hand_quat()
        reference_right_hand_quat = self.start_eef_quat
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

    def characterisation_observation_callback(self, data):
        self.characterisation_observation = data

    def observation_callback(self, data):
        self.observation = data.data

    def get_observations(self):
        return copy.deepcopy(self.observation), copy.deepcopy(self.characterisation_observation)

    def reset(self):
        rospy.loginfo('Resetting push robot observation publisher')
        self.observation = None
        self.characterisation_observation = None

def _convert_message_to_array(controller, c_observation):
    goal_pos_in_base = controller.base_rot_in_eef.dot(np.array([c_observation.goal_pos_in_eefg.x,
                                                                c_observation.goal_pos_in_eefg.y,
                                                                c_observation.goal_pos_in_eefg.z,
                                                                ]))

    eef_pos_in_base = controller.base_rot_in_eef.dot(np.array([c_observation.eef_pos_in_eefg.x,
                                                               c_observation.eef_pos_in_eefg.y,
                                                               c_observation.eef_pos_in_eefg.z, ]))

    object_pos_in_base = controller.base_rot_in_eef.dot(np.array([c_observation.object_pos_in_eefg.x,
                                                                  c_observation.object_pos_in_eefg.y,
                                                                  c_observation.object_pos_in_eefg.z, ]))

    eef_vel_in_base = controller.base_rot_in_eef.dot(np.array([c_observation.eef_vel_in_eefg.x,
                                                               c_observation.eef_vel_in_eefg.y,
                                                               c_observation.eef_vel_in_eefg.z, ]))

    object_vel_in_base = controller.base_rot_in_eef.dot(np.array([c_observation.object_vel_in_eefg.x,
                                                                  c_observation.object_vel_in_eefg.y,
                                                                  c_observation.object_vel_in_eefg.z,
                                                                  ]))

    z_angle = c_observation.z_angle

    return goal_pos_in_base, eef_pos_in_base, object_pos_in_base, \
           eef_vel_in_base, object_vel_in_base, z_angle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--log_dir", type=str,
        default='../logs/test/manual_push',
        help="Path to the parent directory where the produced trajectories are stored")
    args = parser.parse_args(rospy.myargv()[1:])
    log_dir = args.log_path



    rospy.init_node('push_control', anonymous=True)

    default_value = 0.1
    a = default_value
    repeats = 0
    goal_no = 1

    # Reset publisher for the topic resetting the observations (used in order to change goals)
    reset_publisher = rospy.Publisher('pushing/reset', Bool, queue_size=30)

    print('Setting up the robot...')
    # Controller
    controller = Control()

    while not rospy.is_shutdown():

        # Only continue with the script if the observations are ready
        done = False
        while not done and not rospy.is_shutdown():
            observation, c_observation = controller.get_observations()
            if (observation is not None and c_observation is not None):
                sys.stdout.write('\r **Observations ready**.')
                sys.stdout.flush()
                done = True
            else:
                sys.stdout.write('\rWaiting for observation initialisation...')
                sys.stdout.flush()

        controller.go_to_start()

        # Ask the user when to start the control loop
        done = False
        while not done and not rospy.is_shutdown():
            sys.stdout.write('\r **Ready. Press any key to start, f to quit**.')
            sys.stdout.flush()
            c = intera_external_devices.getch()
            if c in ['\x1b', '\x03', 'f']:
                rospy.signal_shutdown("Shutting Down...")
                return
            elif c:
                done = True

        log_path = '{}/goal{}/trajectory_log_{}.csv'.format(log_dir, goal_no,repeats)
        log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                log_path)
        log_list = ["step", "time",
                    "cmd_eef_vx", "cmd_eef_vy",
                    "goal_x", "goal_y", "goal_z",
                    "eef_x", "eef_y", "eef_z",
                    "eef_vx", "eef_vy", "eef_vz",
                    "object_x", "object_y", "object_z",
                    "object_vx", "object_vy", "object_vz",
                    "z_angle",
                    "obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "obs_5", "obs_6", "obs_7", "obs_8", "obs_9",
                    ]

        logger = Logger(log_list, log_path)
        # Set control rate
        rate = rospy.Rate(10)

        # Time and iteration count
        i = 0
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            # Record the time and the current observations
            elapsed = rospy.Time.now() - start
            observation, c_observation = controller.get_observations()

            # Set the action
            action = [a, 0.0]
            joint_vels = controller.get_control(action)
            controller._right_arm.set_joint_velocities(joint_vels)

            # Log observations
            elapsed_sec = elapsed.to_sec()
            action_in_base = controller.base_rot_in_eef.dot(np.concatenate([action, [0.0]]))[:2]
            goal_pos_in_base, eef_pos_in_base, object_pos_in_base, \
            eef_vel_in_base, object_vel_in_base, z_angle = _convert_message_to_array(controller, c_observation)

            logger.log(i, elapsed_sec,
                       action_in_base[0], action_in_base[1],
                       goal_pos_in_base[0], goal_pos_in_base[1], goal_pos_in_base[2],
                       eef_pos_in_base[0], eef_pos_in_base[1], eef_pos_in_base[2],
                       eef_vel_in_base[0], eef_vel_in_base[1], eef_vel_in_base[2],
                       object_pos_in_base[0], object_pos_in_base[1], object_pos_in_base[2],
                       object_vel_in_base[0], object_vel_in_base[1], object_vel_in_base[2],
                       z_angle,
                       observation[0], observation[1], observation[2],
                       observation[3], observation[4], observation[5],
                       observation[6], observation[7], observation[8], observation[9]
                       )

            ## Check that trajectory not stopped
            c = intera_external_devices.getch()
            if c in ['\x1b', '\x03']:
                rospy.signal_shutdown("Shutting Down...")

            # Change the action if above 2.5 sec
            if (elapsed_sec >2.5):
                a = 0.0

            if(elapsed_sec > 3.0):
                a = default_value
                break

            i += 1
            rate.sleep()

        controller.set_neutral()
        done = False
        print('Episode finished, press f or esc to quit or r to restart, q to change goal position')

        while not done and not rospy.is_shutdown():
            c = intera_external_devices.getch()
            if c in ['\x1b', '\x03', 'f']:
                done = True
                rospy.signal_shutdown("Shutting Down...")
            elif c in ['r']:
                repeats += 1
                done = True
            elif c in ['q']:
                # publish reset
                reset_message = Bool()
                reset_message.data = True
                reset_publisher.publish(reset_message)
                # controller reset
                repeats = 0
                goal_no += 1
                controller.reset()
                done = True


if __name__ == '__main__':
    main()
