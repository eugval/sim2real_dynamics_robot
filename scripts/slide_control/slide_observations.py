import sys

sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
import numpy as np
from multiprocessing import Lock
from copy import deepcopy
from sim2real_dynamics_sawyer.msg import pushing_object_observations, sliding_object_observations, Floats64


class Observation(object):
    def __init__(self, right_arm):
        ''' Creates the observations for the sliding tasks, by combining the published object observations and
        the robot state'''
        self.operational = False

        #arm attributes
        self._right_arm = right_arm
        self._joint_names = self._right_arm.joint_names()

        #Object observation attributes
        self.observation_data = None
        self.obs_lock = Lock()

        self.observation_subscriber = rospy.Subscriber('sliding/object_observations',sliding_object_observations,
                                                       self.object_observation_callback,
                                                       queue_size=1)

        while True:
            if(self.observation_data is not None):
                break

        self.operational = True

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



    def object_observation_callback(self, data):
        self.obs_lock.acquire()
        try:
            self.observation_data = data
        finally:
            self.obs_lock.release()

    def grab_raw_object_obs_data(self):
        self.obs_lock.acquire()
        try:
            return deepcopy(self.observation_data)
        finally:
            self.obs_lock.release()

    def _extract_xyzw_array_from_msg(self, msg_obj, w=False):
        if(w):
            return np.array([msg_obj.x,
                             msg_obj.y,
                             msg_obj.z,
                             msg_obj.w])
        else:
            return np.array([msg_obj.x,
                             msg_obj.y,
                             msg_obj.z])


    def convert_message_data(self, object_data):
        object_pos_in_goal = self._extract_xyzw_array_from_msg(object_data.object_pose_in_goal.position)
        object_orn_in_goal = self._extract_xyzw_array_from_msg(object_data.object_pose_in_goal.orientation, w=True)
        object_vel_in_goal = self._extract_xyzw_array_from_msg(object_data.object_vel_in_goal)
        z_angle = object_data.z_angle
        sin_z = object_data.sin_z
        cos_z = object_data.cos_z
        rot = np.array([object_data.rot_1,object_data.rot_2,object_data.rot_3,object_data.rot_4,object_data.rot_5,
                        object_data.rot_6,object_data.rot_7,object_data.rot_8,object_data.rot_9,])

        fallen_object = object_data.fallen_object

        return object_pos_in_goal,object_orn_in_goal,object_vel_in_goal, z_angle, sin_z, cos_z, rot, fallen_object

    def grab_object_obs_data(self):
        raw_data = self.grab_raw_object_obs_data()
        object_pos_in_goal,object_orn_in_goal,object_vel_in_goal, z_angle, sin_z, cos_z, rot, fallen_object = self.convert_message_data(raw_data)

        object_obs = np.concatenate([object_pos_in_goal, np.array([sin_z]), np.array([cos_z]),  object_vel_in_goal,
                                     np.array([z_angle]),object_orn_in_goal, rot, np.array([float(fallen_object)])])

        return object_obs


    def get_joints_vel(self):
        joints_vel = np.array([
            self._right_arm.joint_velocity('right_j5'),
            self._right_arm.joint_velocity('right_j6'),
        ])
        return joints_vel

    def get_joints_angle(self):
        joints_angle = np.array([
            self._right_arm.joint_angle('right_j5'),
            self._right_arm.joint_angle('right_j6'),
        ])
        return joints_angle


    def get_observation(self):
        '''Get the policy observation for the sliding task'''
        obs = self.grab_object_obs_data()
        joints_vel = self.get_joints_vel()
        joints_angle = self.get_joints_angle()

        trimmed_obs = obs[:8]

        return np.concatenate([trimmed_obs, joints_angle,joints_vel])


    def get_c_observation(self):
        '''Get a more comprehensive observation for logging purposes'''
        obs = self.grab_object_obs_data()
        joints_vel = self.get_joints_vel()
        joints_angle = self.get_joints_angle()

        return np.concatenate([obs,joints_angle,joints_vel])


    def get_all_obs(self):
        obs = self.grab_object_obs_data()
        joints_vel = self.get_joints_vel()
        joints_angle = self.get_joints_angle()

        trimmed_obs = obs[:8]

        return np.concatenate([trimmed_obs, joints_angle, joints_vel]) , np.concatenate([obs,joints_angle,joints_vel])