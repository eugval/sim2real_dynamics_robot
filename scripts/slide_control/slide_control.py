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
from control_utils.rl_utils import load, load_model, stack_data
from slide_observations import Observation
import torch
from copy import deepcopy

class Control(object):
    def __init__(self, right_arm):
        # Make sure there is a clean shutdown
        rospy.on_shutdown(self.clean_shutdown)

        # The neutral position of the joints
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

    def go_to_start(self):
        '''Go to the neutral sliding position'''
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

    def get_control(self, action, max_action = 0.3, safety_max = 0.4):
        action = max_action*action
        action = np.clip(action, -safety_max, safety_max)

        if (np.isnan(action).any()):
            rospy.signal_shutdown('NAN in action')
            sys.exit(1)

        return {
            'right_j5':action[0],
            'right_j6':action[1],
        }

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

    rospy.init_node('slide_control', anonymous=True)
    repeats = 0
    horizon = 60


    action_dim = 2
    params_dim = 20
    env_name = 'SawyerSlide'



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
        state_dim = 12
        ##### SETTING UP THE POLICY #########
        method = ['Single', 'LSTM', 'EPI', 'UPOSI'][method_idx]
        if(method == 'Single'):
            alg_idx = 1
        elif(method == 'LSTM'):
            alg_idx = 2
        elif(method == 'UPOSI'):
            alg_idx = 3
            osi_l = 5
            CAT_INTERNAL = False
            osi_input_dim = osi_l*(state_dim+action_dim)
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
        randomisation_type = ['slide_no-randomisation', 'slide_full-randomisation',
                              'slide_force-randomisation', 'slide_force-&-noise-randomisation'][randomisation_idx]
        number_random_params = [0, 22, 2, 8][randomisation_idx]
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

        # Initialise the logger
        log_path = '{}/sliding/video_runs/{}/goal_1/trajectory_log_{}.csv'.format(log_dir,save_name,repeats)
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
        loop_times = 0.0
        start = rospy.Time.now()
        if (alg_name == 'uposi_td3') or (alg_name == 'epi_td3'):
            epi_traj = []
        while not rospy.is_shutdown():
            loop_beginning = rospy.Time.now().to_sec()
            # Grab time and policy observations
            elapsed = rospy.Time.now() - start
            observation, c_observation = observation_generator.get_all_obs()

            ##### Pre action choosing operations
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
                ###############

            ####### CHOOSING THE ACTION ########
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
            #####################################

            if (np.isnan(action).any()):
                rospy.signal_shutdown('NAN in action')
                sys.exit(1)

            joint_vels = controller.get_control(action)
            controller._right_arm.set_joint_velocities(joint_vels)

            if (alg_name == 'uposi_td3'):
                if CAT_INTERNAL:
                    target_joint_action = np.array(joint_vels)
                    full_action = np.concatenate([action, target_joint_action])
                else:
                    full_action = action
                epi_traj.append(np.concatenate((full_state, full_action)))

            # Log observations
            elapsed_sec = elapsed.to_sec()
            logger.log(i, elapsed_sec,
                       action[0], action[1],
                       c_observation[0], c_observation[1], c_observation[2],
                       c_observation[3], c_observation[4],
                       c_observation[5], c_observation[6], c_observation[7],
                       c_observation[8], c_observation[9], c_observation[10],c_observation[11],c_observation[12],
                       c_observation[13], c_observation[14],  c_observation[15],
                       c_observation[16], c_observation[17], c_observation[18],
                       c_observation[19], c_observation[20], c_observation[21],
                       c_observation[22],
                       c_observation[23], c_observation[24],
                       c_observation[25],  c_observation[26]
                       )

            i += 1
            loop_times += rospy.Time.now().to_sec()-loop_beginning

            if (c_observation[22] > 0.5):
                controller.set_slide_neutral()
                print('lost')
                print(loop_times/i)
                break

            if (i >= horizon):
                print(loop_times/i)
                break

            rate.sleep()

        done = False

        def _finish_message():
            print('''Episode finished, press:
                      f or esc to quit 
                      r to restart''')
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
