#!/usr/bin/env python
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
import intera_interface
import os
import intera_external_devices
from intera_interface import CHECK_VERSION
from control_utils.logger import Logger
import numpy as np
import argparse
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped, Pose
from slide_observations import Observation
from copy import deepcopy

class Control(object):
    def __init__(self, right_arm):
        # Make sure there is a clean shutdown
        rospy.on_shutdown(self.clean_shutdown)
        self.neutral_joint_positions = {'right_j6': 3.3168583984375,
                                        'right_j5': 0.566333984375,
                                        'right_j4': -0.0032294921875,
                                        'right_j3': 2.17734765625,
                                        'right_j2': -0.002685546875,
                                        'right_j1': -1.1773017578125,
                                        'right_j0': 0.0018466796875}

        # control parameters
        self._rate = 500.0  # Hz

        self._right_arm = right_arm

        self._right_joint_names = self._right_arm.joint_names()

        #  end-effector initial pose
        self.set_slide_neutral()

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
        self.set_slide_neutral()

    def set_slide_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        joint_positions = deepcopy(self.neutral_joint_positions)

        joint_positions['right_j5'] = joint_positions['right_j5'] - np.pi / 2.
        self._right_arm.move_to_joint_positions(joint_positions)

    def clean_shutdown(self):
        print("\nExiting example...")
        # return to normal
        self._reset_control_modes()
        self.set_slide_neutral()
        return True

    def get_control(self, action, safety_max = 0.1):
        action = np.clip(action, -safety_max, safety_max)

        if (np.isnan(action).any()):
            rospy.signal_shutdown('NAN in action')
            sys.exit(1)

        return {
            'right_j5':action[0],
            'right_j6':action[1],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--log_dir", type=str,
        default='../logs/manual_slide',
        help="Path to the parent directory where the produced trajectories are stored")
    args = parser.parse_args(rospy.myargv()[1:])
    log_dir = args.log_path


    rospy.init_node('reach_characterisation', anonymous=True)
    default_value = 0.1
    a = default_value
    repeats = 0

    # Setting up Sawyer
    print('Setting up the robot...')
    Right_arm = intera_interface.limb.Limb("right")

    print("Getting robot state... ")
    _rs = intera_interface.RobotEnable(CHECK_VERSION)
    _init_state = _rs.state().enabled
    print("Enabling robot... ")
    _rs.enable()
    print("Robot ready.")
    Right_arm.set_joint_position_speed(speed=0.1)

    # Setup controller and observation maker
    observation_generator = Observation(Right_arm)
    controller = Control(Right_arm)

    while not rospy.is_shutdown():
        done = False
        while not done and not rospy.is_shutdown():
            if (observation_generator.operational is not None ):
                sys.stdout.write('\r **Observations ready**.')
                sys.stdout.flush()
                done = True
            else:
                sys.stdout.write('\rWaiting for observation initialisation...')
                sys.stdout.flush()

        controller.go_to_start()

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


        log_path = '{}/goal1/trajectory_log_{}.csv'.format(log_dir,repeats)
        log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                log_path)
        log_list = ["step", "time",
                    "cmd_j5", "cmd_j6",
                    "obj_x", "obj_y", "obj_z",
                    "sin_z", "cos_z",
                    "obj_vx", "obj_vy", "obj_vz",
                    "z_angle", "obj_orn_x", "obj_orn_y", "obj_orn_z", "obj_orn_w",
                    "rot_1", "rot_2", "rot_3", "rot_4", "rot_5", "rot_6", "rot_7", "rot_8", "rot_9",
                    "fallen_object",
                    "a_j5", "a_j6",
                    "v_j5", "v_j6",
                    ]

        logger = Logger(log_list, log_path)
        # Set control rate
        rate = rospy.Rate(10)

        # Time and iteration count
        i = 0
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            # Grab observations
            elapsed = rospy.Time.now() - start
            observation = observation_generator.get_c_observation()

            # Set the action
            action = [a, a]
            joint_vels = controller.get_control(action)
            controller._right_arm.set_joint_velocities(joint_vels)

            ## Format the observations for logging and log
            elapsed_sec = elapsed.to_sec()

            logger.log(i, elapsed_sec,
                       action[0], action[1],
                       observation[0], observation[1],observation[2],
                       observation[3], observation[4], observation[5], observation[6],
                       observation[7],observation[8],observation[9],
                       observation[10],observation[11],
                       observation[12],observation[13],
                       observation[14], observation[15], observation[16],
                       observation[17], observation[18], observation[19],
                       observation[20], observation[21], observation[22],
                       observation[23], observation[24], observation[25],
                       observation[26]
                       )
            if (elapsed_sec > 2.):
                break

            i += 1
            rate.sleep()

        done = False

        def _finish_message():
            print('''Episode finished, press:
                      f or esc to quit 
                      r to restart
                      g to change the goal position''')
        _finish_message()
        while not done and not rospy.is_shutdown():
            c = intera_external_devices.getch()
            if c:
                if c in ['\x1b', '\x03', 'f']:
                    done = True
                    rospy.signal_shutdown("Shutting Down...")
                    controller.set_slide_neutral()
                elif c in ['r']:
                    controller.set_slide_neutral()
                    repeats+=1
                    done = True
                else:
                    _finish_message()


if __name__ == '__main__':
    main()
