import gtsam
from DataSets.extractData import ROSData
from DataSets.extractGt import GroundTruthEstimates
from DataTypes.uwb_position import UWB_Ancors_Descriptor
from Sensors.CameraSensor.featureDetector import feature_detector_factory
from Sensors.IMU import IMU
from Sensors.GNSS import GNSS
from settings import DATASET_NUMBER
from gtsam.symbol_shorthand import X, L, V, B
import numpy as np
from scipy.spatial.transform import Rotation as R
from Sensors.CameraSensor.camera import PinholeCamera
from Sensors.CameraSensor.epipolarGeometry import EpipolarGeometry
from Sensors.CameraSensor.featureDetector import *
from Sensors.CameraSensor.featureMatcher import *
from Sensors.CameraSensor2.visualOdometry import VisualOdometry
from Plotting.plot_gtsam import plot_horizontal_trajectory, plot_position, plot_angels, plot_bias, plot_vel


import matplotlib.pyplot as plt


class CameraUwbImuFusion:

    def __init__(self) -> None:
        self.dataset = ROSData(DATASET_NUMBER)
        isam_params: gtsam.ISAM2Params = gtsam.ISAM2Params()
        isam_params.setFactorization("QR")
        isam_params.setRelinearizeSkip(1)
        self.isam: gtsam.ISAM2 = gtsam.ISAM2(isam_params)
        self.uwb_positions: UWB_Ancors_Descriptor = UWB_Ancors_Descriptor(
            DATASET_NUMBER)
        self.ground_truth: GroundTruthEstimates = GroundTruthEstimates(
            DATASET_NUMBER, pre_initialization=True)
        self.imu_params: IMU = IMU()
        self.gnss_params: GNSS = GNSS()

        # Tracked variables for IMU and UWB
        self.pose_variables: list = []
        self.velocity_variables: list = []
        self.imu_bias_variables: list = []
        self.landmarks_variables: dict = {}
        self.uwb_counter: set = set()
        self.time_stamps: list = []

        # Setting up gtsam values
        self.graph_values: gtsam.Values = gtsam.Values()
        self.factor_graph: gtsam.NonlinearFactorGraph = gtsam.NonlinearFactorGraph()
        self.initialize_graph()
        # sns.set()

        # Visual odometry part
        camera = PinholeCamera()

    def initialize_graph(self):

        # Defining the state
        X1 = X(0)
        V1 = V(0)
        B1 = B(0)
        self.pose_variables.append(X1)
        self.velocity_variables.append(V1)
        self.imu_bias_variables.append(B1)

        # Set priors
        self.prior_noise_x = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-15, 1e-15, 1e-5, 1e-3, 1e-3, 1e-10]))
        self.prior_noise_v = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.0001, 0.0001, 0.0001]))
        self.prior_noise_b = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([5e-12, 5e-12, 5e-12, 5e-7, 5e-7, 5e-7]))

        # Pose for pre init
        R_init = R.from_euler(
            "xyz", self.ground_truth.initial_pose()[:3], degrees=False)
        T_init = self.ground_truth.initial_pose()[3:]
        T_init[2] = -0.7
        self.current_pose = gtsam.Pose3(gtsam.Rot3(R_init.as_matrix()), T_init)

        self.current_velocity = self.ground_truth.initial_velocity()
        self.current_bias = gtsam.imuBias.ConstantBias(
            np.zeros((3,)), np.array([0, 0, 0, -15e-3, -5e-3, -7e-3]))
        self.navstate = gtsam.NavState(self.current_pose.rotation(), self.current_pose.translation(
        ), self.current_pose.rotation().matrix().T @ self.current_velocity)

        self.factor_graph.add(gtsam.PriorFactorPose3(
            X1, self.current_pose, self.prior_noise_x))
        self.factor_graph.add(gtsam.PriorFactorVector(
            V1, self.current_velocity, self.prior_noise_v))
        self.factor_graph.add(gtsam.PriorFactorConstantBias(
            B1, self.current_bias, self.prior_noise_b))

        self.graph_values.insert(X1, self.current_pose)
        self.graph_values.insert(V1, self.current_velocity)
        self.graph_values.insert(B1, self.current_bias)

        # Initialize vo
        self.visual_odometry = VisualOdometry(R_init.as_matrix(), T_init)

    def run(self):
        imu_measurements = []
        iteration_number = 0
        iteration_number_cam = 0

        for measurement in self.dataset.generate_measurements():

            if measurement.measurement_type.value == "Camera":
                self.visual_odometry.track(measurement.image)
                self.time_stamps.append(measurement.time.to_time())

                iteration_number_cam += 1
                print(iteration_number_cam)
                print("Start time of dataset", measurement.time)
            iteration_number += 1
            scale = 1

            if iteration_number_cam > 1000:
                break

        plt.plot(range(len(self.visual_odometry.yaw)),
                 np.array(self.visual_odometry.yaw))
        plt.title("Yaw measurements")
        plt.figure(2)
        plot_position(np.array([np.array(self.visual_odometry.North)[:, 0], np.array(self.visual_odometry.East)[
                      :, 0], np.array(self.visual_odometry.Down)[:, 0]]).T, self.ground_truth, self.time_stamps, convert_NED=True)
        plt.figure(3)
        plot_angels(np.array([np.array(self.visual_odometry.roll), np.array(self.visual_odometry.pitch), np.array(
            self.visual_odometry.yaw)]).T, self.ground_truth, self.time_stamps)
        plt.figure(4)

        plt.plot(np.array(self.visual_odometry.North),
                 np.array(self.visual_odometry.East))
        #plt.title("Horisontal plot")
        plt.show()


fusion = CameraUwbImuFusion()
fusion.run()
