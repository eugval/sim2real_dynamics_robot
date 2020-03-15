#!/usr/bin/env python
import sys

sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from rospy_message_converter import message_converter
import numpy as np
from collections import deque
from control_utils import transform_utils as T
from multiprocessing import Lock
from copy import deepcopy
from sim2real_dynamics_sawyer.msg import pushing_object_observations, sliding_object_observations, Floats64
import intera_interface
from intera_interface import CHECK_VERSION

import signal



class ObjectObservationGenerator(object):
    def __init__(self, goal_id=2, object_id=1,
                 eef_posemat_in_goal=np.array([[-1.00000000e+00, 6.30034379e-19, 9.15101892e-17, -8.50000000e-02],
                                               [7.50102353e-17, -4.18750849e-17, 1.00000000e+00, -2.25000000e-01],
                                               [-1.60209754e-17, 1.00000000e+00, -3.97274917e-17, -0.0015],
                                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                                1.00000000e+00]]),

                 object_centre_posemat_in_object_tag=np.array(
                     [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -0.015], [0., 0., 0., 1.]]),
                 calibration_samples=100, publication_rate=30,
                 camera='cam1', tag_detector='april_ros'):
        '''
        :param goal_id: The ID of the april tag used as the goal
        :param object_id: The ID of the april tag used as the object
        :param eef_posemat_in_goal: The pose matrix of the end effector in the goal frame: always fixed.
        :param object_centre_posemat_in_object_tag: Captures the offset between the april tag and the centre of the object
        :param calibration_samples: Number of samples to use to calibrate the camera position in the base of the robot.
        :param publication_rate: Publication rate of sliding object observations
        :param camera: The camera name used to extract the poses
        :param tag_detector: The tag detector used to extract poses
        '''

        self.operational = False
        signal.signal(signal.SIGINT, self.clean_shutdown)
        rospy.on_shutdown(self.clean_shutdown)

        # Setting up the robot
        self._rate = 500.0
        self._right_arm = intera_interface.Limb("right")
        self._right_joint_names = self._right_arm.joint_names()

        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        self._right_arm.set_joint_position_speed(speed=0.1)
        # # move the joint 5 by pi/2 - Initial position for the sliding task
        self.neutral_joint_positions = {'right_j6': 3.3168583984375,
                                        'right_j5': 0.566333984375,
                                        'right_j4': -0.0032294921875,
                                        'right_j3': 2.17734765625,
                                        'right_j2': -0.002685546875,
                                        'right_j1': -1.1773017578125,
                                        'right_j0': 0.0018466796875}

        joint_positions = deepcopy(self.neutral_joint_positions)

        joint_positions['right_j5']=  joint_positions['right_j5'] - np.pi/2.
        self._right_arm.move_to_joint_positions(joint_positions)

        # Operational attributes
        self.publication_rate = publication_rate
        self.camera = camera

        # IDs
        self.object_id = str(object_id)
        self.goal_id = str(goal_id)

        # Goal calibration attributes
        self.calibration_samples = calibration_samples
        self.goal_positions_in_cam = deque()
        self.goal_orientations_in_cam = deque()

        # Reference frame transforms
        self.eef_posemat_in_goal = eef_posemat_in_goal
        self.object_centre_posemat_in_object_tag = object_centre_posemat_in_object_tag
        self.goal_posemat_in_eef = T.pose_inv(eef_posemat_in_goal)
        self.base_posemat_in_eefi = T.pose_inv(self._get_eef_posemat_in_base())

        # Initialised later
        self.base_posemat_in_cam = None

        # Online at fast rate
        self.time_cam = 0.0
        self.object_pose_in_cam = (np.zeros(3, ), np.zeros(4, ))
        self.goal_pose_in_cam = (np.zeros(3, ), np.zeros(4, ))

        # Online at delayed rate
        self.prev_time_cam = -1. / publication_rate  # frame rate of publisher
        self.prev_object_pos_in_cam = np.zeros(3, )
        self.prev_goal_pos_in_cam = np.zeros(3, )
        self.prev_vel_in_cam = np.zeros(3, )

        #Other attributes
        self.lost_times = 0

        # Subscribers
        if (tag_detector == 'april_ros'):
            self.AR_marker_subscriber = rospy.Subscriber('{}/tag_detections'.format(camera), AprilTagDetectionArray,
                                                         self.cam_callback_april_ros_marker)
        else:
            raise RuntimeError('Unknown tag detector')

        # Locks
        self.cam_lock = Lock()
        self.calibration_lock = Lock()

        self.reset()

    def _get_eef_posemat_in_base(self):
        eef_pose_in_base = self._right_arm.endpoint_pose()
        eef_quat_in_base = np.array([eef_pose_in_base['orientation'].x,
                                     eef_pose_in_base['orientation'].y,
                                     eef_pose_in_base['orientation'].z,
                                     eef_pose_in_base['orientation'].w, ])
        eef_pos_in_base = np.array([eef_pose_in_base['position'].x,
                                    eef_pose_in_base['position'].y,
                                    eef_pose_in_base['position'].z, ])
        return T.pose2mat((eef_pos_in_base, eef_quat_in_base))


    def reset(self):
        '''Resetting the sliding environment will recalibrate the pose of the camera with respect to the base of the robot.'''
        rospy.loginfo('Resetting slide object observation publisher')
        self.operational = False
        if (rospy.has_param('sliding/base_posemat_in_cam')):
            rospy.delete_param('sliding/base_posemat_in_cam')
        self.goal_positions_in_cam = deque()
        self.goal_orientations_in_cam = deque()

        self._finish_initialisation()

    def clean_shutdown(self, ):
        self.cam_lock.release()
        self.calibration_lock.release()
        self._reset_control_modes()
        sys.exit(0)

    def _reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._right_arm.exit_control_mode()
            rate.sleep()

    def _finish_initialisation(self):
        ''' Obtain the pose of the camera in the robot's base frame by using the calibration samples gathered'''
        rospy.loginfo('Initialising observations ...')

        while True:
            if (rospy.has_param('sliding/base_posemat_in_cam')):
                break

            if len(self.goal_positions_in_cam) > self.calibration_samples:
                break

        self.calibration_lock.acquire()
        try:
            if (rospy.has_param('sliding/base_posemat_in_cam')):
                self.base_posemat_in_cam = np.array(rospy.get_param('sliding/base_posemat_in_cam'))
            else:
                # Get the mean position and orientation of the goal in cam1 and cam2
                goal_positions_in_cam = np.stack(self.goal_positions_in_cam, axis=0)
                goal_position_in_cam = np.mean(goal_positions_in_cam, axis=0)

                goal_orientations_in_cam = np.stack(self.goal_orientations_in_cam, axis=0)
                goal_orientation_in_cam = np.mean(goal_orientations_in_cam, axis=0)

                goal_pose_in_cam = (goal_position_in_cam, goal_orientation_in_cam)

                goal_posemat_in_cam = T.pose2mat(goal_pose_in_cam)

                eefi_posemat_in_cam = goal_posemat_in_cam.dot(self.eef_posemat_in_goal)
                self.base_posemat_in_cam = eefi_posemat_in_cam.dot(self.base_posemat_in_eefi)
                rospy.set_param('sliding/base_posemat_in_cam', self.base_posemat_in_cam.tolist())

            rospy.loginfo('Done')
            self.operational = True

        finally:
            self.calibration_lock.release()

    def _get_pose_from_dict(self, pose_dict):
        position = np.array([pose_dict['position']['x'], pose_dict['position']['y'], pose_dict['position']['z']])
        orientation = np.array(
            [pose_dict['orientation']['x'], pose_dict['orientation']['y'], pose_dict['orientation']['z'],
             pose_dict['orientation']['w']])
        return (position, orientation)

    def process_markers(self, detected_markers, marker_poses, time):
        '''If the calibration of the camera pose in the base of the robot is not finished yet,
         then just use the detected markers for calibration.
             Otherwise, store  the current object pose and the current time for publishing'''
        if (not self.operational):
            self.calibrate(detected_markers, marker_poses)
        else:
            self.cam_lock.acquire()
            try:
                if (self.object_id in detected_markers):
                    self.time_cam = time
                    self.object_pose_in_cam = marker_poses[self.object_id]
                if (self.goal_id in detected_markers):
                    self.goal_pose_in_cam = marker_poses[self.goal_id]
            finally:
                self.cam_lock.release()

    def calibrate(self, detected_markers, marker_poses):
        '''If the goal in in the detected markes, then store that goal position and orientation in the calibration lists.
        Those can then be used to find the pose of the camera in the base frame of the robot'''
        if (self.goal_id in detected_markers):
            self.calibration_lock.acquire()
            try:
                self.goal_positions_in_cam.append(marker_poses[self.goal_id][0])
                self.goal_orientations_in_cam.append(marker_poses[self.goal_id][1])
            finally:
                self.calibration_lock.release()

    def cam_callback_april_ros_marker(self, data):
        '''
          Callback to the april tag detector subscriber.
        Extract the timestamps from the headers, convert the message into a dictionary, and extract the pose of each detected marker.
        :param data: message data
        '''
        header = data.header
        markers = data.detections

        secs = header.stamp.secs
        nsecs = header.stamp.nsecs
        time = secs + nsecs * 1e-9

        marker_poses = {}

        for i, marker in enumerate(markers):
            converted_marker = message_converter.convert_ros_message_to_dictionary(marker)

            marker_poses[str(converted_marker['id'][0])] = self._get_pose_from_dict(
                converted_marker['pose']['pose']['pose'])

        detected_markers = marker_poses.keys()

        self.process_markers(detected_markers, marker_poses, time)



    def grab_data_raw(self):
        '''
        Grabs the raw data obtained from the ar tag subscriptions from each camera in a thread safe manner
        :return:
        '''

        self.cam_lock.acquire()
        try:
            object_pose_in_cam = deepcopy(self.object_pose_in_cam)
            goal_pose_in_cam = deepcopy(self.goal_pose_in_cam)
            time_cam = deepcopy(self.time_cam)
        finally:
            self.cam_lock.release()

        return object_pose_in_cam, goal_pose_in_cam, time_cam

    def grab_data(self):
        '''Returns the object observations in the end-effector goal frame'''

        # Grab raw data
        object_pose_in_cam, goal_pose_in_cam, time_cam = self.grab_data_raw()

        # Compute object velocities
        object_vel_in_cam, object_lost = self.compute_velocities(object_pose_in_cam[0], time_cam)


        if (object_lost):
            self.lost_times +=1
            object_pose_in_cam = list(object_pose_in_cam)
            object_pose_in_cam[0] = object_pose_in_cam[0] + self.lost_times*object_vel_in_cam / self.publication_rate
        else:
            self.lost_times = 0

        # Get goal posemat
        goal_posemat_in_cam, goal_lost = self.compute_goal_posemat(goal_pose_in_cam)

        # Convert data to the correct frame
        cam_posemat_in_goal = T.pose_inv(goal_posemat_in_cam)
        object_posemat_in_cam = T.pose2mat(object_pose_in_cam)

        object_posemat_in_goal = cam_posemat_in_goal.dot(object_posemat_in_cam)
        object_vel_in_goal = cam_posemat_in_goal[:3, :3].dot(object_vel_in_cam)

        return object_posemat_in_goal, object_vel_in_goal, object_lost, goal_lost, self.lost_times > 2

    def compute_goal_posemat(self, current_goal_pose):
        current_goal_pos = current_goal_pose[0]
        if ((current_goal_pos == self.prev_goal_pos_in_cam).any()):
            base_posemat_in_cam = self.base_posemat_in_cam
            eef_posemat_in_base = self._get_eef_posemat_in_base()
            goal_posemat_in_eef = self.goal_posemat_in_eef
            goal_posemat_in_cam = base_posemat_in_cam.dot(eef_posemat_in_base).dot(goal_posemat_in_eef)
            return goal_posemat_in_cam, True
        else:
            self.prev_goal_pos_in_cam = current_goal_pos
            return T.pose2mat(current_goal_pose), False

    def compute_velocities(self, current_pos, time_cam):
        ''' Compute the velocities using finite differences. Detect if the tracking is lost, by seeing if the position
              between two detections is identical. '''
        prev_pos = self.prev_object_pos_in_cam
        prev_time = self.prev_time_cam
        dt = time_cam - prev_time
        dx = current_pos - prev_pos

        if (dt <= 0 or (dx == 0).all()):
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
        object_posemat_in_goal, object_vel_in_goal, object_lost, goal_lost, fallen_object = data

        # Correct for the fact that the tag is not in the middle of the object
        object_posemat_in_goal = object_posemat_in_goal.dot(self.object_centre_posemat_in_object_tag)

        z_angle = T.mat2euler(object_posemat_in_goal[:3, :3])[2]
        sin_cos = np.array([np.sin(8 * z_angle), np.cos(8 * z_angle)])

        flattened_rot_matrix = deepcopy(object_posemat_in_goal[:3, :3]).reshape((9,))
        return {
            'object_pos_in_goal': object_posemat_in_goal[:3, 3],
            'object_orn_in_goal': T.mat2quat(object_posemat_in_goal[:3, :3]),
            'object_vel_in_goal': object_vel_in_goal,
            'object_lost': object_lost,
            'goal_lost': goal_lost,
            'z_angle':z_angle,
            'sin_z' : sin_cos[0],
            'cos_z' : sin_cos[1],
            'rot_1' :flattened_rot_matrix[0],
            'rot_2' :flattened_rot_matrix[1],
            'rot_3' :flattened_rot_matrix[2],
            'rot_4' :flattened_rot_matrix[3],
            'rot_5' :flattened_rot_matrix[4],
            'rot_6' :flattened_rot_matrix[5],
            'rot_7' :flattened_rot_matrix[6],
            'rot_8' :flattened_rot_matrix[7],
            'rot_9' :flattened_rot_matrix[8],
            'fallen_object' : fallen_object
        }

    def get_object_observation(self):
        data = self.grab_data()
        return self._get_object_observation(data)


def main():
    rospy.init_node('object_observation_node', anonymous=True)
    tag_detector = rospy.get_param('~tag_detector')

    obs_generator = ObjectObservationGenerator(camera=rospy.get_param('~camera'), tag_detector=tag_detector)

    observation_publisher = rospy.Publisher('sliding/object_observations',
                                            sliding_object_observations,
                                            queue_size=30)

    rate = rospy.Rate(obs_generator.publication_rate)

    while not rospy.is_shutdown():
        if (obs_generator.operational):
            o = obs_generator.get_object_observation()

            obs_msg = sliding_object_observations()
            obs_msg.object_pose_in_goal.position.x = o['object_pos_in_goal'][0]
            obs_msg.object_pose_in_goal.position.y = o['object_pos_in_goal'][1]
            obs_msg.object_pose_in_goal.position.z = o['object_pos_in_goal'][2]
            obs_msg.object_pose_in_goal.orientation.x = o['object_orn_in_goal'][0]
            obs_msg.object_pose_in_goal.orientation.y = o['object_orn_in_goal'][1]
            obs_msg.object_pose_in_goal.orientation.z = o['object_orn_in_goal'][2]
            obs_msg.object_pose_in_goal.orientation.w = o['object_orn_in_goal'][3]
            obs_msg.object_vel_in_goal.x = o['object_vel_in_goal'][0]
            obs_msg.object_vel_in_goal.y = o['object_vel_in_goal'][1]
            obs_msg.object_vel_in_goal.z = o['object_vel_in_goal'][2]
            obs_msg.object_lost = o['object_lost']
            obs_msg.goal_lost = o['goal_lost']
            obs_msg.z_angle = o['z_angle']
            obs_msg.sin_z = o['sin_z']
            obs_msg.cos_z = o['cos_z']
            obs_msg.rot_1 = o['rot_1']
            obs_msg.rot_2 = o['rot_2']
            obs_msg.rot_3 = o['rot_3']
            obs_msg.rot_4 = o['rot_4']
            obs_msg.rot_5 = o['rot_5']
            obs_msg.rot_6 = o['rot_6']
            obs_msg.rot_7 = o['rot_7']
            obs_msg.rot_8 = o['rot_8']
            obs_msg.rot_9 = o['rot_9']
            obs_msg.fallen_object = o['fallen_object']

            observation_publisher.publish(obs_msg)

        rate.sleep()


if __name__ == '__main__':
    main()


