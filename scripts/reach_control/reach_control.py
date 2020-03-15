#!/usr/bin/env python
import sys

sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy

from control_utils.eef_velocity_controller import SawyerEEFVelocityController
import torch
import intera_interface
import os
import intera_external_devices
from intera_interface import CHECK_VERSION
from control_utils import transform_utils as T
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
from reach_kinematics_observation import Observation

from control_utils.rl_utils import load, load_model, stack_data


class Control(object):
    def __init__(self, right_arm):
        # Make sure there is a clean shutdown
        rospy.on_shutdown(self.clean_shutdown)

        # control parameters
        self._rate = 500.0  # Hz

        self._right_arm = right_arm

        self._right_joint_names = self._right_arm.joint_names()

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

        self.start_eef_pos = self.calibrate_initial_position()

        # Eef velocity control
        self.controller = SawyerEEFVelocityController()

        # Sin-cos control
        self.period_factor = 0.15
        self.amplitude_factor = 0.1


    def calibrate_initial_position(self):
        ''' Set the starting position of the end-effector.'''
        rospy.loginfo('Calibrating reaching starting position....')
        if (rospy.has_param('start_reaching_position')):
            return np.array(rospy.get_param('start_reaching_position'))
        else:
            rospy.set_param('manual_arm_move', True)

            rate = rospy.Rate(20)
            while True:
                if(not rospy.get_param('manual_arm_move')):
                    rospy.loginfo('Done')
                    return self.done_moving()
                rate.sleep()

    def done_moving(self):
        ''' Record the current pose of the end effector and end the manual moving.'''
        rospy.loginfo('Recording start pos')
        eef_pose = self._right_arm.endpoint_pose()
        start_eef_pos = np.array([eef_pose['position'].x,
                                  eef_pose['position'].y,
                                  eef_pose['position'].z,
                                  ])

        rospy.set_param('start_reaching_position', start_eef_pos.tolist())

        return start_eef_pos

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

    def get_control(self, action, max_action = 0.1, safety_max = 0.15):
        '''Convert the action of joint velocity commands.
                Use inverse kinematics to go from end effector cartesian velocities to joint velocities.'''
        action = max_action * action
        current_joint_angles = [self._right_arm.joint_angles()['right_j{}'.format(i)] for i in range(7)]

        # get x,y,z velocity in base frame
        action_in_base = self.base_rot_in_eef.dot(action)
        action_in_base = np.clip(action_in_base, -safety_max, safety_max)

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

    rospy.init_node('reach_control', anonymous=True)

    # Parameters
    horizon = 50
    repeats = 0
    goal_no = 1

    action_dim=3
    internal_state_dim=14
    internal_action_dim=7
    params_dim=49    
    env_name='SawyerReach'


    # Setting up Sawyer
    print('Setting up the robot...')
    Right_arm = intera_interface.limb.Limb("right")

    print("Getting robot state... ")
    _rs = intera_interface.RobotEnable(CHECK_VERSION)
    _init_state = _rs.state().enabled
    print("Enabling robot... ")
    _rs.enable()
    print("Robot ready.")

    # Setup controller and observation maker
    observation_generator = Observation(Right_arm, reset_goal_pos= True, predefined_goal='goal_{}'.format(goal_no))
    controller = Control(Right_arm)

    while not rospy.is_shutdown():
        state_dim = 6

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
                osi_input_dim = osi_l * (state_dim + action_dim)

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
        randomisation_type = ['reach_no-randomisation', 'reach_full-randomisation', \
                              'reach_force-randomisation', 'reach_force-&-noise-randomisation'][randomisation_idx]
        number_random_params = [0, 14, 1, 3][randomisation_idx]
        folder_path = load_dir +'/' +method + '/' + alg_name + '/model/'
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
            # Initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            hidden_out = (torch.zeros([1, 1, 512], dtype=torch.float).cuda(), \
                          torch.zeros([1, 1, 512],
                                      dtype=torch.float).cuda())
            last_action = np.array([0, 0, 0])
        ######################################

        log_save_name = '{}_{}_{}_{}'.format(method, alg_name, randomisation_type, number_random_params)


        done = False
        while not done and not rospy.is_shutdown():
            if (observation_generator.goal_pos_in_base is not None ):
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

        #Initialise the trajectory logger
        log_path = '{}/reaching/{}/goal_{}/trajectory_log_{}.csv'.format(log_dir,log_save_name,goal_no,repeats)
        log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                log_path)
        log_list = ["step", "time",
                    "cmd_eef_vx", "cmd_eef_vy", "cmd_eef_vz",
                    "eef_x", "eef_y", "eef_z",
                    "eef_vx", "eef_vy", "eef_vz",
                    "goal_x","goal_y","goal_z",
                    "obs_0", "obs_1", "obs_2",
                    "obs_3", "obs_4", "obs_5"
                    ]

        logger = Logger(log_list, log_path)
        # Set control rate
        rate = rospy.Rate(10)

        # Time and iteration count
        i = 0
        start = rospy.Time.now()
        if (alg_name == 'uposi_td3') or (alg_name == 'epi_td3'):
            epi_traj = []

        while not rospy.is_shutdown():
            # Grab observations
            elapsed = rospy.Time.now() - start
            observation = observation_generator.get_observation()
            c_observation = observation_generator.get_characterisation_observation()

            ##### Pre-action choosing operations
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
                if(len(epi_traj)>= traj_l):
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
            ###############

            #### CHOOSING THE ACTION #####
            if (alg_name == 'lstm_td3'):
                hidden_in = hidden_out
                action, hidden_out=policy.get_action(observation, last_action, hidden_in, noise_scale=0.0)
                last_action = action
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

            ###############################
            if (np.isnan(action).any()):
                rospy.signal_shutdown('NAN in action')
                sys.exit(1)
                
            # joint_vels is the joint velocity COMMAND - the current joint velocities can be queried through controller.joint_vels
            joint_vels = controller.get_control(action)
            controller._right_arm.set_joint_velocities(joint_vels)

            if (alg_name == 'uposi_td3'):
                if CAT_INTERNAL:
                    target_joint_action = controller.joint_array_from_joint_dict(joint_vels)
                    print(action.shape)#
                    print(target_joint_action.shape)
                    full_action = np.concatenate([action, target_joint_action])
                else:
                    full_action = action
                epi_traj.append(np.concatenate((full_state, full_action)))
                
            ## Format the observations for logging and log
            elapsed_sec = elapsed.to_sec()
            eef_pos_in_base = c_observation['eef_pos_in_base']
            eef_vel_in_base = c_observation['eef_vel_in_base']
            goal_pos_in_base = c_observation['goal_pos_in_base']
            action_in_base = controller.base_rot_in_eef.dot(action)

            logger.log(i, elapsed_sec,
                       action_in_base[0], action_in_base[1], action_in_base[2],
                       eef_pos_in_base[0], eef_pos_in_base[1], eef_pos_in_base[2],
                       eef_vel_in_base[0], eef_vel_in_base[1], eef_vel_in_base[2],
                       goal_pos_in_base[0], goal_pos_in_base[1], goal_pos_in_base[2],
                       observation[0], observation[1], observation[2],
                       observation[3], observation[4], observation[5]
                       )

            i += 1
            if (i >= horizon):
                break

            rate.sleep()

        controller.set_neutral()
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
                elif c in ['g']:
                    repeats = 0
                    goal_no +=1
                    observation_generator.reset(specific_goal ='goal_{}'.format(goal_no))
                    done = True
                elif c in ['r']:
                    repeats+=1
                    done = True
                else:
                    _finish_message()


if __name__ == '__main__':
    main()
