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
from sim2real_dynamics_sawyer.msg import pushing_characterisation_observations, Floats64, pushing_reset

from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_motion_msgs.msg import TrajectoryOptions
from control_utils.rl_utils import load, load_model, stack_data
import torch
from rospy.numpy_msg import numpy_msg
import copy
from multiprocessing import Lock
import sys





class Control(object):
    def __init__(self, use_recorded_starting_position = True):
        # Make sure there is a clean shutdown
        rospy.on_shutdown(self.clean_shutdown)

        # Hard-coded starting position of the end - effector for pushing. Used to save time in multiple experiments from the same
        # starting position
        self.recorded_starting_position = np.array([0.4497117313460486, -0.010764198311385465, -0.0651])
        # control parameters
        self._rate = 500.0  # Hz

        # Setting up the observation subscriber
        self.characterisation_observation = None
        self.observation = None

        self.obs_lock = Lock()
        self.c_obs_lock = Lock()

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
            self.start_eef_pos, self.start_eef_quat = self.calibrate_initial_position(use_recorded_starting_position)

        else:
            self.start_eef_pos = np.array(rospy.get_param('pushing/start_position'))
            self.start_eef_quat = np.array(rospy.get_param('pushing/start_orientation'))

        # Eef velocity control
        self.controller = SawyerEEFVelocityController()

    def calibrate_initial_position(self, use_recorded_starting_position = False):
        ''' Set the starting position of the end-effector.'''
        rospy.loginfo('Place the end-effector in the desired starting position and press  d...')

        if(use_recorded_starting_position):
            eef_pose = self._right_arm.endpoint_pose()
            start_eef_quat = np.array([eef_pose['orientation'].x,
                                       eef_pose['orientation'].y,
                                       eef_pose['orientation'].z,
                                       eef_pose['orientation'].w, ])

            rospy.set_param('pushing/start_position', self.recorded_starting_position.tolist())
            rospy.set_param('pushing/start_orientation', start_eef_quat.tolist())
            return self.recorded_starting_position, start_eef_quat

        elif (rospy.has_param('pushing/start_position')):
            return np.array(rospy.get_param('pushing/start_position')), np.array(rospy.get_param('pushing/start_orientation'))
        else:
            rospy.set_param('manual_arm_move', True)

            rate = rospy.Rate(20)
            while True:
                if (not rospy.get_param('manual_arm_move')):
                    rospy.loginfo('Done')
                    return self.done_moving()
                rate.sleep()


    def done_moving(self):
        ''' Record the current pose of the end effector and end the manual moving.'''
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
        '''Send the end effector to a specified pose'''
        try:
            traj_options = TrajectoryOptions()
            traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
            traj = MotionTrajectory(trajectory_options=traj_options, limb=self._right_arm)

            wpt_opts = MotionWaypointOptions(max_linear_speed=0.4,
                                             max_linear_accel=0.4,
                                             max_rotational_speed=1.4,
                                             max_rotational_accel=1.4,
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
        ''' Send the end effector to the recorded starting position'''
        self.go_to_pose(self.start_eef_pos, self.start_eef_quat)

    def set_neutral(self):
        '''Send the end effector to its neutral pose'''
        print("Moving to neutral pose...")
        self._right_arm.move_to_neutral(speed = 0.15)

    def clean_shutdown(self):
        print("\nExiting example...")
        # return to normal
        self._reset_control_modes()
        self.set_neutral()
        return True

    def get_control(self, action, max_action=0.1, safety_max = 0.15):
        '''Convert the action of joint velocity commands.
        Use inverse kinematics to go from end effector cartesian velocities to joint velocities.'''
        action = max_action * action
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

        joint_velocities = self.controller.compute_joint_velocities_for_endpoint_velocity(pose_matrix,
                                                                                          current_joint_angles)

        if(np.isnan(joint_velocities).any()):
            sys.exit(1)

        j_v = {}
        for i in range(7):
            key = 'right_j{}'.format(i)
            j_v[key] = joint_velocities[i]
        return j_v

    def characterisation_observation_callback(self, data):
        self.c_obs_lock.acquire()
        try:
            self.characterisation_observation = data
        finally:
            self.c_obs_lock.release()

    def observation_callback(self, data):
        self.obs_lock.acquire()
        try:
            self.observation = data.data
        finally:
            self.obs_lock.release()

    def get_observations(self):
        self.obs_lock.acquire()
        self.c_obs_lock.acquire()
        try:
            return copy.deepcopy(self.observation), copy.deepcopy(self.characterisation_observation)
        finally:
            self.obs_lock.release()
            self.c_obs_lock.release()

    def reset(self):
        rospy.loginfo('Resetting push robot observation publisher')
        self.observation = None
        self.characterisation_observation = None

    def get_joints_vel(self):
        joints_vel = np.array([
            self._right_arm.joint_velocity('right_j0'),
            self._right_arm.joint_velocity('right_j1'),
            self._right_arm.joint_velocity('right_j2'),
            self._right_arm.joint_velocity('right_j3'),
            self._right_arm.joint_velocity('right_j4'),
            self._right_arm.joint_velocity('right_j5'),
            self._right_arm.joint_velocity('right_j6'),

        ])
        return joints_vel

    def get_joints_angle(self):
        joints_angle = np.array([
            self._right_arm.joint_velocity('right_j0'),
            self._right_arm.joint_velocity('right_j1'),
            self._right_arm.joint_velocity('right_j2'),
            self._right_arm.joint_velocity('right_j3'),
            self._right_arm.joint_velocity('right_j4'),
            self._right_arm.joint_velocity('right_j5'),
            self._right_arm.joint_velocity('right_j6'),
        ])
        return joints_angle

    def joint_array_from_joint_dict(self, joint_dict):
        joints_array = np.array([
            joint_dict['right_j0'],
            joint_dict['right_j1'],
            joint_dict['right_j2'],
            joint_dict['right_j3'],
            joint_dict['right_j4'],
            joint_dict['right_j5'],
            joint_dict['right_j6'],
        ])
        return joints_array.squeeze()

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
        "-d", "--log_dir", type=str,
        default='../logs',
        help="Path where the trajectories produced are stored")
    parser.add_argument(
        "-l", "--load_dir", type=str,
        default='../../assets/rl',
        help="Path where the policy networks are loaded")
    parser.add_argument(
        "-m", "--method_idx", type=int,
        default=0,
        help="The index of the method to use")
    parser.add_argument(
        "-r", "--randomisation_idx", type=int,
        default=0,
        help="The index of the randomisation regime to use")

    args = parser.parse_args(rospy.myargv()[1:])
    log_dir = args.log_dir
    load_dir = args.load_dir
    method_idx = args.method_idx
    randomisation_idx = args.randomisation_idx

    rospy.init_node('push_control', anonymous=True)

    #Params
    repeats = 0
    goal_no = 1
    horizon = 80

    # Reset publisher
    reset_publisher = rospy.Publisher('pushing/reset', pushing_reset,
                                      queue_size=30)

    action_dim=2
    internal_state_dim=14
    internal_action_dim=7
    params_dim=55
    env_name='SawyerPush'


    rospy.loginfo('Setting up the robot...')
    # Controller
    controller = Control()

    while not rospy.is_shutdown():
        state_dim = 10

        ##### SETTING UP THE POLICY #########
        method = ['Single', 'LSTM', 'EPI', 'UPOSI'][method_idx]
        if(method == 'Single'):
            alg_idx = 1
        elif(method == 'LSTM'):
            alg_idx = 2
        elif(method == 'UPOSI'):
            alg_idx = 3
            osi_l = 5

            CAT_INTERNAL = True
            if CAT_INTERNAL:
                osi_input_dim = osi_l*(state_dim+action_dim+internal_state_dim+internal_action_dim)
            else:
                osi_input_dim = osi_l * (state_dim + action_dim )

            state_dim+=params_dim
        elif(method == 'EPI'):
            alg_idx = 4
            embed_dim = 10
            traj_l = 10
            NO_RESET = True
            embed_input_dim = traj_l*(state_dim+action_dim)
            ori_state_dim = state_dim
            state_dim += embed_dim
        else:
            continue

        alg_name = ['sac', 'td3', 'lstm_td3', 'uposi_td3', 'epi_td3'][alg_idx]
        randomisation_type = ['push_no-randomisation', 'push_full-randomisation', \
                              'push_force-randomisation', 'push_force-&-noise-randomisation'][randomisation_idx]
        number_random_params = [0, 23, 2, 9][randomisation_idx]
        folder_path = load_dir + method + '/' + alg_name + '/model/'
        path = folder_path + env_name + str(
            number_random_params) + '_' + alg_name
        policy = load(path=path, alg=alg_name, state_dim=state_dim,
                      action_dim=action_dim)
        if method == 'UPOSI':
            osi_model = load_model(model_name='osi', path=path, input_dim = osi_input_dim, output_dim=params_dim )
        elif method == 'EPI':
            embed_model = load_model(model_name='embedding', path=path, input_dim = embed_input_dim, output_dim = embed_dim )
            embed_model.cuda()
            epi_policy_path = folder_path + env_name + str(number_random_params) + '_' + 'epi_ppo_epi_policy'
            epi_policy = load(path=epi_policy_path, alg='ppo', state_dim=ori_state_dim, action_dim=action_dim )

        if (alg_name == 'lstm_td3'):
            # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            hidden_out = (torch.zeros([1, 1, 512], dtype=torch.float).cuda(), \
                          torch.zeros([1, 1, 512],
                                      dtype=torch.float).cuda())
            last_action = np.array([0, 0])
        ######################################
        save_name = '{}_{}_{}_{}'.format(method, alg_name, randomisation_type, number_random_params)

        # Wait for the observation topics to be publishing
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

        done = False
        skip = False
        # Ask confirmation of the user to start the trajectory, have the option to quit.
        while not done and not rospy.is_shutdown():
            sys.stdout.write('\r **Ready. Press any key to start, f to quit, q to change goal**.')
            sys.stdout.flush()
            c = intera_external_devices.getch()
            if c in ['\x1b', '\x03', 'f']:
                rospy.signal_shutdown("Shutting Down...")
                return

            elif c in ['q']:
                repeats = 0
                goal_no += 1
                # publish reset
                reset_message = pushing_reset()
                reset_message.reset = True
                reset_message.goal_idx = goal_no
                reset_message.use_predefined = True
                reset_publisher.publish(reset_message)
                # controller reset

                controller.reset()
                skip = True
                done = True

            elif c:
                done = True

        # Initialise the logger for the trajectory
        log_path = '{}/{}/goal_{}/trajectory_log_{}.csv'.format(log_dir,save_name,goal_no,repeats)
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

        if(not skip):
            logger = Logger(log_list, log_path)
        # Set control rate
        rate = rospy.Rate(10)

        # Time and iteration count
        i = 0
        start = rospy.Time.now()
        if (alg_name == 'uposi_td3') or (alg_name == 'epi_td3'):
            epi_traj = []

        while not rospy.is_shutdown() and not skip:
            # Grab time and policy observations
            elapsed = rospy.Time.now() - start
            observation, c_observation = controller.get_observations()

            ##### Pre action choosing processing ######
            if (alg_name == 'uposi_td3'):
                if len(epi_traj)>=osi_l:
                    osi_input = stack_data(epi_traj, osi_l)
                    pre_params = osi_model(osi_input).detach().numpy()
                else:
                    zero_osi_input = np.zeros(osi_input_dim)
                    pre_params = osi_model(zero_osi_input).detach().numpy()   

                params_state = np.concatenate((pre_params, observation))
            elif (alg_name == 'epi_td3'):
                # epi rollout first for each episode;
                if len(epi_traj) < traj_l:
                    a = epi_policy.get_action(observation)
                    action = np.clip(a, -epi_policy.action_range, epi_policy.action_range)
                    s_a_r = np.concatenate((observation, action, [0]))  # state, action, reward; no reward in reality
                    epi_traj.append(s_a_r)
                if (len(epi_traj) >= traj_l):
                    state_action_in_traj = np.array(epi_traj)[:, :-1]  # remove the rewards
                    embedding = embed_model(state_action_in_traj.reshape(-1))
                    embedding = embedding.detach().cpu().numpy()
                    if NO_RESET:
                        observation = observation  # last observation
                    else:
                        # reset the env here and get new observation
                        rospy.signal_shutdown('Cannot reset the environment in the real world')
                        sys.exit(1)
                    observation=np.concatenate((observation, embedding))
            ###########

            ####### CHOOSING THE ACTION ########
            if (alg_name == 'lstm_td3'):
                hidden_in = hidden_out
                action, hidden_out=policy.get_action(observation, last_action, hidden_in, noise_scale=0.0)
                last_action = copy.deepcopy(action)
            elif (alg_name == 'uposi_td3'):
                # using internal state or not
                if CAT_INTERNAL:
                    internal_state = np.concatenate((controller.get_joints_angle(), np.array(controller.get_joints_vel())))
                    full_state = np.concatenate([observation, internal_state])
                else:
                    full_state = observation
                action = policy.get_action(params_state, noise_scale=0.0)
            elif (alg_name == 'epi_td3') and len(epi_traj) < traj_l:  # action already given by EPI policy
                pass
            else:
                action = policy.get_action(observation, noise_scale=0.0)
            #####################################

            if(np.isnan(action).any()):
                rospy.signal_shutdown('NAN in action')
                sys.exit(1)

            # joint_vels is the joint velocity COMMAND - the current joint velocities can be queried through controller.joint_vels
            joint_vels = controller.get_control(action)
            controller._right_arm.set_joint_velocities(joint_vels)

            if (alg_name == 'uposi_td3'):
                if CAT_INTERNAL:
                    target_joint_action =  controller.joint_array_from_joint_dict(joint_vels)
                    full_action = np.concatenate([action, target_joint_action])
                else:
                    full_action = action
                epi_traj.append(np.concatenate((full_state, full_action)))

            # Log observations
            elapsed_sec = elapsed.to_sec()
            action_in_base = controller.base_rot_in_eef.dot(np.concatenate([action, [0.0]]))[:2]
            goal_pos_in_base, eef_pos_in_base, object_pos_in_base, \
            eef_vel_in_base, object_vel_in_base, z_angle = _convert_message_to_array(controller, c_observation)

            print(goal_pos_in_base)
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

            i += 1
            if (i >= horizon):
                break

            rate.sleep()

        controller.set_neutral()

        if(skip):
            done = True
        else:
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
                repeats = 0
                goal_no += 1
                # publish reset
                reset_message = pushing_reset()
                reset_message.reset = True
                reset_message.goal_idx = goal_no
                reset_message.use_predefined = True
                reset_publisher.publish(reset_message)
                # controller reset

                controller.reset()
                done = True


if __name__ == '__main__':
    main()
