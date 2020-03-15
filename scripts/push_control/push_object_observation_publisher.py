#!/usr/bin/env python
import sys

sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from apriltag_ros.msg import AprilTagDetectionArray
from rospy_message_converter import message_converter
import numpy as np
from collections import deque
from control_utils import transform_utils as T
from multiprocessing import Lock
from copy import deepcopy
from sim2real_dynamics_sawyer.msg import pushing_object_observations, Floats64,pushing_reset
import signal


class ObjectObservationGenerator(object):
    def __init__(self, goal_id=2, object_id=1,
                 eefg_posemat_in_goal=np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.12],[0., 0., 0., 1.]]),
                 calibration_samples=200, publication_rate = 30,
                 camera = 'cam1', tag_detector='april_ros'):
        '''
        :param goal_id: The ID of the april tag used to represent the goal
        :param object_id: The ID of the april tag used to track the object
        :param eefg_posemat_in_goal: The end effector position when the gripper is placed at the goal position, in the goal frame
        :param calibration_samples: The number of samples to use to extract the goal position, which is kept constant after calibration
        :param publication_rate: The publication rate of the object observations
        :param camera: The camera name, useful when using multiple cameras
        :param tag_detector: which package is used for tag detection
        '''

        signal.signal(signal.SIGINT, self.clean_shutdown)

        self.operational = False
        self.publication_rate = publication_rate
        self.camera=camera
        # IDs
        self.object_id = str(object_id)
        self.goal_id = str(goal_id)

        # Goal calibration attributes
        self.calibration_samples = calibration_samples
        self.goal_positions_in_cam = deque()
        self.goal_orientations_in_cam = deque()

        # Reference frame transforms
        self.eefg_posemat_in_goal = eefg_posemat_in_goal
        self.goal_posemat_in_eefg = T.pose_inv(self.eefg_posemat_in_goal)

        # Initialised later
        self.cam_rot_in_goal = None
        self.cam_posemat_in_eefg = None

        # Online at fast rate
        self.time_cam = 0.0
        self.object_pose_in_cam = (np.zeros(3, ), np.zeros(4, ))

        # Online at delayed rate
        self.prev_time_cam = -1 / publication_rate  # frame rate of publisher
        self.prev_object_pos_in_cam = np.zeros(3, )
        self.prev_vel_in_cam = np.zeros(3,)


        # Subscribers
        self.reset_subscriber = rospy.Subscriber('pushing/reset', pushing_reset, self.reset_callback,queue_size=1)

        if(tag_detector == 'april_ros'):
            self.AR_marker_subscriber = rospy.Subscriber('{}/tag_detections'.format(camera), AprilTagDetectionArray,
                                                         self.cam_callback_april_ros_marker)
        else:
            raise RuntimeError('Unknown tag detector')

        # Locks
        self.cam_lock = Lock()
        self.calibration_lock = Lock()
        self.reset()


    def reset_callback(self,res):
        ''' When a reset signal is published to the pushing/reset topic, reset the object observations'''
        if(res.reset):
            self.reset()

    def reset(self):
        '''
        Reset the object observation publisher by re-calibrating the goal position.
        '''
        rospy.loginfo('Resetting push object observation publisher')
        self.operational = False
        self.goal_positions_in_cam = deque()
        self.goal_orientations_in_cam = deque()
        if(rospy.has_param('{}/pushing/goal_position'.format(self.camera))):
            rospy.delete_param('{}/pushing/goal_position'.format(self.camera))
            rospy.delete_param('{}/pushing/goal_orientation'.format(self.camera))
        self._finish_initialisation()


    def clean_shutdown(self,sig,frame):
        self.cam_lock.release()
        self.calibration_lock.release()
        sys.exit(0)

    def _finish_initialisation(self):
        '''
        Find the goal position at initialisation and then keep it constant until reset of the observations.
        '''
        rospy.loginfo('Initialising observations ...')

        while True:
            # If a goal position is already in the ROS parameters, there is nothing to be done
            if(rospy.has_param('{}/pushing/goal_position'.format(self.camera))):
                break

            # Until enough individual goal positions have been recorded, do nothing
            if len(self.goal_positions_in_cam) > self.calibration_samples:
                break

        self.calibration_lock.acquire()
        try:
            # Get the mean position and orientation of the goal in cam1 and cam2
            goal_positions_in_cam = np.stack(self.goal_positions_in_cam, axis=0)
            goal_position_in_cam = np.mean(goal_positions_in_cam, axis=0)

            goal_orientations_in_cam = np.stack(self.goal_orientations_in_cam, axis=0)
            goal_orientation_in_cam = np.mean(goal_orientations_in_cam, axis=0)

            rospy.set_param('{}/pushing/goal_position'.format(self.camera), goal_position_in_cam.tolist())
            rospy.set_param('{}/pushing/goal_orientation'.format(self.camera), goal_orientation_in_cam.tolist())

            goal_pose_in_cam = (goal_position_in_cam, goal_orientation_in_cam)
            goal_rot_in_cam = T.quat2mat(goal_orientation_in_cam)

            self.cam_rot_in_goal = goal_rot_in_cam.T
            goal_posemat_in_cam = T.pose2mat(goal_pose_in_cam)
            cam_posemat_in_goal = T.pose_inv(goal_posemat_in_cam)
            self.cam_posemat_in_eefg = T.pose_in_A_to_pose_in_B(cam_posemat_in_goal, self.goal_posemat_in_eefg)

            rospy.loginfo('Done')
            self.operational = True

        finally:
            self.calibration_lock.release()

    def _get_pose_from_dict(self, pose_dict):
        '''
        Use a pose dictionary with x,y,z, position and x,y,z,w quaternion orientation, and return a pose tuple
        (position, orientation) of numpy arrays.
        '''
        position = np.array([pose_dict['position']['x'], pose_dict['position']['y'], pose_dict['position']['z']])
        orientation = np.array(
            [pose_dict['orientation']['x'], pose_dict['orientation']['y'], pose_dict['orientation']['z'],
             pose_dict['orientation']['w']])
        return (position, orientation)


    def calibrate(self, detected_markers, marker_poses):
        '''If the goal in in the detected markes, then store that goal position and orientation in the calibration lists'''
        if (self.goal_id in detected_markers):
            self.calibration_lock.acquire()
            try:
                self.goal_positions_in_cam.append(marker_poses[self.goal_id][0])
                self.goal_orientations_in_cam.append(marker_poses[self.goal_id][1])
            finally:
                self.calibration_lock.release()

    def process_markers(self, detected_markers, marker_poses,time):
        '''If the calibration of the goal is not finished yet, then just use the detected markers for calibration.
        Otherwise, store  the current object pose and the current time for publishing'''
        if (not self.operational):
            self.calibrate(detected_markers, marker_poses)
        else:
            if (self.object_id in detected_markers):
                self.cam_lock.acquire()
                try:
                    self.time_cam = time
                    self.object_pose_in_cam = marker_poses[self.object_id]
                finally:
                    self.cam_lock.release()



    def cam_callback_april_ros_marker(self, data):
        '''
        Callback to the april tag detector subscriber.
        Extract the timestamps from the headers, convert the message into a dictionary, and extract the pose of each detected marker.
        '''
        header = data.header
        markers = data.detections

        secs = header.stamp.secs
        nsecs = header.stamp.nsecs
        time = secs + nsecs * 1e-9

        marker_poses = {}
        for i, marker in enumerate(markers):
            converted_marker = message_converter.convert_ros_message_to_dictionary(marker)

            marker_poses[str(converted_marker['id'][0])] = self._get_pose_from_dict(converted_marker['pose']['pose']['pose'])

        detected_markers = marker_poses.keys()

        self.process_markers(detected_markers, marker_poses, time)


    def grab_data_raw(self):
        '''
        Grabs the raw data obtained from the ar tag subscriptions from each camera in a thread safe manner.
        '''

        self.cam_lock.acquire()
        try:
            object_pos_in_cam = deepcopy(self.object_pose_in_cam[0])
            time_cam = deepcopy(self.time_cam)
            object_orn_in_cam = deepcopy(self.object_pose_in_cam[1])

        finally:
            self.cam_lock.release()

        return (object_pos_in_cam, object_orn_in_cam), time_cam

    def grab_data(self):
        '''Returns the object observations in the end-effector goal frame (appart from z-angle, which is around the table normal (or goal frame)'''

        # Grab raw data
        object_pose_in_cam, time_cam = self.grab_data_raw()

        # Compute velocities
        object_vel_in_cam, lost = self.compute_velocities(object_pose_in_cam[0], time_cam)

        if (lost):
            # If the tracking lost the object, recover slightly by using the previous velocity in order to predict the current position
            object_pose_in_cam = list(object_pose_in_cam)
            object_pose_in_cam[0] = object_pose_in_cam[0] + object_vel_in_cam / self.publication_rate

        # Convert data to the correct frame
        object_posemat_in_eefg = T.pose_in_A_to_pose_in_B(T.pose2mat(object_pose_in_cam), self.cam_posemat_in_eefg)
        object_pos_in_eefg = object_posemat_in_eefg[:3, 3]
        object_vel_in_eefg = self.cam_posemat_in_eefg[:3, :3].dot(object_vel_in_cam)

        object_rot_in_camera = T.quat2mat(object_pose_in_cam[1])
        object_rot_in_goal = self.cam_rot_in_goal.dot(object_rot_in_camera)
        object_euler_in_goal = T.mat2euler(object_rot_in_goal)
        z_angle = np.array([object_euler_in_goal[2]])

        return object_pos_in_eefg, object_vel_in_eefg, z_angle, lost

    def compute_velocities(self, current_pos, time_cam):
        ''' Compute the velocities using finite differences. Detect if the tracking is lost, by seeing if the position
        between two detections is identical. '''
        prev_pos = self.prev_object_pos_in_cam
        prev_time = self.prev_time_cam
        dt = time_cam - prev_time
        dx = current_pos - prev_pos

        if (dt <= 0 or  (dx == 0).all()):
            # If the object is lost, assume it kept its previous velocity
            vel = self.prev_vel_in_cam
            lost = True
        else:
            vel = dx / dt
            lost = False

        self.prev_object_pos_in_cam = current_pos
        self.prev_time_cam = time_cam
        self.prev_vel_in_cam = vel

        return vel, lost

    def _get_object_observation(self, data):
        ''' Put the object observations into a dictionary'''
        object_pos_in_eefg, object_vel_in_eefg, z_angle, lost = data

        # Correct for the fact that the tag is not in the middle of the object
        object_pos_in_eefg = object_pos_in_eefg + np.array([0., 0., 0.015])
        return {'object_pos_in_eefg': object_pos_in_eefg,
                'object_vel_in_eefg': object_vel_in_eefg,
                'z_angle': z_angle[0],
                'lost': lost}

    def get_object_observation(self):
        data = self.grab_data()
        return self._get_object_observation(data)

def main():
    rospy.init_node('object_observation_node', anonymous=True)
    tag_detector = rospy.get_param('~tag_detector')

    obs_generator = ObjectObservationGenerator(camera=rospy.get_param('~camera'), tag_detector=tag_detector)

    observation_publisher = rospy.Publisher('{}/object_observations'.format(obs_generator.camera),
                                            pushing_object_observations,
                                            queue_size=30)

    rate = rospy.Rate(obs_generator.publication_rate)

    while not rospy.is_shutdown():
        if(obs_generator.operational):
            o = obs_generator.get_object_observation()

            obs_msg = pushing_object_observations()
            obs_msg.object_pos_in_eefg.x = o['object_pos_in_eefg'][0]
            obs_msg.object_pos_in_eefg.y = o['object_pos_in_eefg'][1]
            obs_msg.object_pos_in_eefg.z = o['object_pos_in_eefg'][2]
            obs_msg.object_vel_in_eefg.x = o['object_vel_in_eefg'][0]
            obs_msg.object_vel_in_eefg.y = o['object_vel_in_eefg'][1]
            obs_msg.object_vel_in_eefg.z = o['object_vel_in_eefg'][2]
            obs_msg.z_angle = o['z_angle']
            obs_msg.lost = o['lost']

            observation_publisher.publish(obs_msg)

        rate.sleep()

if __name__ == '__main__':
    main()
