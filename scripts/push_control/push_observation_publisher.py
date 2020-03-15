#!/usr/bin/env python
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
import numpy as np
from multiprocessing import Lock
from copy import deepcopy
from rospy.numpy_msg import numpy_msg
from sim2real_dynamics_sawyer.msg import pushing_object_observations, pushing_robot_observations, \
    pushing_characterisation_observations, Floats64, pushing_reset
import signal

class ObservationGenerator(object):
    def __init__(self, publication_rate=30):
        '''
        Used to put together the object observations of two cameras and the robot observations, and publish the final pushing
        task observations
        :param publication_rate: The publication rate of the pushing task observations
        '''
        signal.signal(signal.SIGINT, self.clean_shutdown)
        self.operational=False

        self.publication_rate = publication_rate
        self.cam1_object_observation = None
        self.cam2_object_observation = None
        self.robot_observation = None

        self.cam1_object_observation_lock = Lock()
        self.cam2_object_observation_lock = Lock()
        self.robot_observation_lock = Lock()

        self.reset_subscriber = rospy.Subscriber('pushing/reset', pushing_reset, self.reset_callback, queue_size=1)

        self.cam1_object_observation_subscriber = rospy.Subscriber('cam1/object_observations',
                                                                   pushing_object_observations,
                                                                   self.cam1_object_observation_callback,
                                                                   queue_size=1)  #

        self.cam2_object_observation_subscriber = rospy.Subscriber('cam2/object_observations',
                                                                   pushing_object_observations,
                                                                   self.cam2_object_observation_callback,
                                                                   queue_size=1)  #
        self.robot_observation_subscriber = rospy.Subscriber('robot_observations', pushing_robot_observations,
                                                             self.robot_observation_callback)

        self.reset()


    def _finish_initialisation(self):
        ''' Verifies that all the nodes that the pushing observations rely on are publishing, and signals the initialisation of this node'''
        while True:
            if (self.cam1_object_observation is not None and
                    self.cam2_object_observation is not None and
                    self.robot_observation is not None):
                self.operational = True
                rospy.set_param('pushing/observations_initialised', True)
                break

    def reset_callback(self,res):
        if(res.reset):
            self.reset()

    def reset(self):
        '''On reset, clean all the observations received by the object and robot observation nodes'''
        rospy.loginfo('Resetting push observation publisher')

        self.operational = False
        rospy.set_param('pushing/observations_initialised', False)
        self.cam1_object_observation = None
        self.cam2_object_observation = None
        self.robot_observation = None
        self._finish_initialisation()

    def clean_shutdown(self,sig,frame):
        self.cam1_object_observation_lock.release()
        self.cam2_object_observation_lock.release()
        self.robot_observation_lock.release()
        sys.exit(0)


    def cam1_object_observation_callback(self, data):
        '''Acquire and store the object observations from cam1 in a thread safe manner'''
        self.cam1_object_observation_lock.acquire()
        try:
            self.cam1_object_observation = data
        finally:
            self.cam1_object_observation_lock.release()

    def cam2_object_observation_callback(self, data):
        '''Acquire and store the object observations from cam2 in a thread safe manner'''

        self.cam2_object_observation_lock.acquire()
        try:
            self.cam2_object_observation = data
        finally:
            self.cam2_object_observation_lock.release()

    def robot_observation_callback(self, data):
        '''Acquire and store the robot observations in a thread safe manner'''
        self.robot_observation_lock.acquire()
        try:
            self.robot_observation = data
        finally:
            self.robot_observation_lock.release()

    def _grab_subscribed_data(self):
        '''Grab all the currently stored data from the low level observation publishers in a thread safe manner '''
        self.robot_observation_lock.acquire()
        self.cam1_object_observation_lock.acquire()
        self.cam2_object_observation_lock.acquire()

        try:
            cam1_object_data = deepcopy(self.cam1_object_observation)
            cam2_object_data = deepcopy(self.cam2_object_observation)
            robot_data = deepcopy(self.robot_observation)
        finally:
            self.robot_observation_lock.release()
            self.cam1_object_observation_lock.release()
            self.cam2_object_observation_lock.release()

        return cam1_object_data, cam2_object_data, robot_data

    def _extract_xyz_array_from_msg(self, msg_obj):
        '''Extract a position array from a position message'''
        return np.array([msg_obj.x,
                         msg_obj.y,
                         msg_obj.z])

    def _elementwise_mean(self, array1, array2):
        '''Take the elementwise mean of two numpy arrays'''
        return (array1 + array2) / 2.

    def _convert_message_data(self, cam1_object_data, cam2_object_data, robot_data):
        '''Take message data and convert them into numpy arrays, and process redundant data'''

        # If the object is lost in only one of the cameras, return the observations from the other camera, in all other cases
        # keep the average observation between the two cameras
        if (cam1_object_data.lost == cam2_object_data.lost):
            cam1_object_pos_in_eefg = self._extract_xyz_array_from_msg(cam1_object_data.object_pos_in_eefg)
            cam1_object_vel_in_eefg = self._extract_xyz_array_from_msg(cam1_object_data.object_vel_in_eefg)
            cam1_z_angle = cam1_object_data.z_angle

            cam2_object_pos_in_eefg = self._extract_xyz_array_from_msg(cam2_object_data.object_pos_in_eefg)
            cam2_object_vel_in_eefg = self._extract_xyz_array_from_msg(cam2_object_data.object_vel_in_eefg)
            cam2_z_angle = cam2_object_data.z_angle

            object_pos_in_eefg = self._elementwise_mean(cam1_object_pos_in_eefg, cam2_object_pos_in_eefg)
            object_vel_in_eefg = self._elementwise_mean(cam1_object_vel_in_eefg, cam2_object_vel_in_eefg)
            z_angle = self._elementwise_mean(cam1_z_angle, cam2_z_angle)
        elif (cam1_object_data.lost):

            object_pos_in_eefg = self._extract_xyz_array_from_msg(cam2_object_data.object_pos_in_eefg)
            object_vel_in_eefg = self._extract_xyz_array_from_msg(cam2_object_data.object_vel_in_eefg)
            z_angle = cam2_object_data.z_angle

        elif (cam2_object_data.lost):
            object_pos_in_eefg = self._extract_xyz_array_from_msg(cam1_object_data.object_pos_in_eefg)
            object_vel_in_eefg = self._extract_xyz_array_from_msg(cam1_object_data.object_vel_in_eefg)
            z_angle = cam1_object_data.z_angle
        else:
            raise ValueError(
                'Either both cameras are lost/found or one only is found -> issue with cameras publications')

        eef_pos_in_eefg = self._extract_xyz_array_from_msg(robot_data.eef_pos_in_eefg)
        eef_vel_in_eefg = self._extract_xyz_array_from_msg(robot_data.eef_vel_in_eefg)
        goal_pos_in_eefg = self._extract_xyz_array_from_msg(robot_data.goal_pos_in_eefg)

        return object_pos_in_eefg, object_vel_in_eefg, z_angle, \
               eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg

    def _get_observation(self, object_pos_in_eefg, object_vel_in_eefg, z_angle,
                         eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg):
        '''Construct the pushing observation vector from its individual components'''

        object_to_goal_in_eefg = goal_pos_in_eefg - object_pos_in_eefg
        eef_to_object_in_eefg = object_pos_in_eefg - eef_pos_in_eefg

        return np.concatenate([eef_to_object_in_eefg[:2], object_to_goal_in_eefg[:2],
                               eef_vel_in_eefg[:2], object_vel_in_eefg[:2],
                               [np.sin(8*z_angle), np.cos(8*z_angle)]]).astype(np.float64)

    def _get_characterisation_observation(self, object_pos_in_eefg, object_vel_in_eefg, z_angle,
                                          eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg):
        ''' Summarise the individual observations into a dictionary '''
        return {'object_pos_in_eefg': object_pos_in_eefg,
                'goal_pos_in_eefg': goal_pos_in_eefg,
                'eef_pos_in_eefg': eef_pos_in_eefg,
                'object_vel_in_eefg': object_vel_in_eefg,
                'eef_vel_in_eefg': eef_vel_in_eefg,
                'z_angle': z_angle, }

    def get_observation(self):
        '''Grab the data from the different observation topics, convert them to numpy arrays and use them to make a pushing
        observation array'''
        cam1_object_data, cam2_object_data, robot_data = self._grab_subscribed_data()
        object_pos_in_eefg, object_vel_in_eefg, z_angle, \
        eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg = self._convert_message_data(cam1_object_data,
                                                                                        cam2_object_data,
                                                                                        robot_data)
        return self._get_observation(object_pos_in_eefg, object_vel_in_eefg, z_angle,
                                     eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg)

    def get_characterisation_observation(self):
        '''Grab the data from the different observation topics, convert them to numpy arrays and return them in
        a python dictionary'''
        cam1_object_data, cam2_object_data, robot_data = self._grab_subscribed_data()
        object_pos_in_eefg, object_vel_in_eefg, z_angle, \
        eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg = self._convert_message_data(cam1_object_data,
                                                                                        cam2_object_data,
                                                                                        robot_data)



        return self._get_characterisation_observation(object_pos_in_eefg, object_vel_in_eefg, z_angle,
                                                      eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg)

    def get_all_observations(self):
        ''' Return both the pushing observation vector and the more descriptive individual observations components'''
        cam1_object_data, cam2_object_data, robot_data = self._grab_subscribed_data()
        object_pos_in_eefg, object_vel_in_eefg, z_angle, \
        eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg = self._convert_message_data(cam1_object_data,
                                                                                        cam2_object_data,
                                                                                        robot_data)
        return self._get_observation(object_pos_in_eefg, object_vel_in_eefg, z_angle,
                                     eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg), \
               self._get_characterisation_observation(object_pos_in_eefg, object_vel_in_eefg, z_angle,
                                                      eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg)

    def _get_characterisation_observation_msg(self, c_obs):
        ''' Make a message with the diffent pushing observation components, useful for logging trajectories'''
        obs_msg = pushing_characterisation_observations()

        obs_msg.object_pos_in_eefg.x = c_obs['object_pos_in_eefg'][0]
        obs_msg.object_pos_in_eefg.y = c_obs['object_pos_in_eefg'][1]
        obs_msg.object_pos_in_eefg.z = c_obs['object_pos_in_eefg'][2]

        obs_msg.object_vel_in_eefg.x = c_obs['object_vel_in_eefg'][0]
        obs_msg.object_vel_in_eefg.y = c_obs['object_vel_in_eefg'][1]
        obs_msg.object_vel_in_eefg.z = c_obs['object_vel_in_eefg'][2]

        obs_msg.z_angle = c_obs['z_angle']

        obs_msg.goal_pos_in_eefg.x = c_obs['goal_pos_in_eefg'][0]
        obs_msg.goal_pos_in_eefg.y = c_obs['goal_pos_in_eefg'][1]
        obs_msg.goal_pos_in_eefg.z = c_obs['goal_pos_in_eefg'][2]

        obs_msg.eef_pos_in_eefg.x = c_obs['eef_pos_in_eefg'][0]
        obs_msg.eef_pos_in_eefg.y = c_obs['eef_pos_in_eefg'][1]
        obs_msg.eef_pos_in_eefg.z = c_obs['eef_pos_in_eefg'][2]

        obs_msg.eef_vel_in_eefg.x = c_obs['eef_vel_in_eefg'][0]
        obs_msg.eef_vel_in_eefg.y = c_obs['eef_vel_in_eefg'][1]
        obs_msg.eef_vel_in_eefg.z = c_obs['eef_vel_in_eefg'][2]

        return obs_msg

    def get_observation_msg(self):
        ''' Return the observation message, in this case it is a numpy array as we use the numpy_msg package'''
        return self.get_observation()

    def get_characterisation_observation_msg(self):
        '''Return a message with the different observation components, useful for trajectory logging.'''
        c_obs = self.get_characterisation_observation()

        return self._get_characterisation_observation_msg(c_obs)

    def get_all_observations_msg(self):
        '''Return both the pushing observation message , and the observation components message.'''
        cam1_object_data, cam2_object_data, robot_data = self._grab_subscribed_data()
        object_pos_in_eefg, object_vel_in_eefg, z_angle, \
        eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg = self._convert_message_data(cam1_object_data,
                                                                                        cam2_object_data,
                                                                                        robot_data)

        c_obs = self._get_characterisation_observation(object_pos_in_eefg, object_vel_in_eefg, z_angle,
                                                       eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg)



        return self._get_observation(object_pos_in_eefg, object_vel_in_eefg, z_angle,
                                     eef_pos_in_eefg, eef_vel_in_eefg, goal_pos_in_eefg), \
               self._get_characterisation_observation_msg(c_obs)


def main():
    rospy.init_node('observation_node', anonymous=True)

    obs_generator = ObservationGenerator()

    observation_publisher = rospy.Publisher('observations', numpy_msg(Floats64), queue_size=1)
    c_observation_publisher = rospy.Publisher('characterisation_observations', pushing_characterisation_observations,
                                              queue_size=30)

    rate = rospy.Rate(obs_generator.publication_rate)

    while not rospy.is_shutdown():
        if(obs_generator.operational):
            obs_msg, c_obs_msg = obs_generator.get_all_observations_msg()

            observation_publisher.publish(obs_msg)
            c_observation_publisher.publish(c_obs_msg)

        rate.sleep()


if __name__ == '__main__':
    main()
