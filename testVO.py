import gtsam
from DataSets.extractData import ROSData
from DataSets.extractGt import GroundTruthEstimates
from settings import DATASET_NUMBER
import numpy as np
from scipy.spatial.transform import Rotation as R
from Sensors.CameraSensor.featureDetector import *
from Sensors.CameraSensor.featureMatcher import *
from Sensors.CameraSensor2.visualOdometry import VisualOdometry
from Plotting.plot_gtsam import plot_position, plot_angels
import matplotlib.pyplot as plt


class CameraUwbImuFusion:

    def __init__(self) -> None:
        self.dataset = ROSData(DATASET_NUMBER)
        self.ground_truth: GroundTruthEstimates = GroundTruthEstimates(DATASET_NUMBER, pre_initialization=False)

        # Tracked variables for IMU and UWB
        self.time_stamps: list = []

        # Setting up gtsam values
        self.graph_values: gtsam.Values = gtsam.Values()
        self.factor_graph: gtsam.NonlinearFactorGraph = gtsam.NonlinearFactorGraph()
        self.initialize_graph()
        # sns.set()

    def initialize_graph(self):

        # Pose for pre init
        R_init = R.from_euler("xyz", self.ground_truth.initial_pose()[:3], degrees=False)
        T_init = self.ground_truth.initial_pose()[3:]
        T_init[2] = -0.7

        # Initialize vo
        self.visual_odometry = VisualOdometry(R_init.as_matrix(), T_init)

    def run(self):
        iteration_number = 0
        iteration_number_cam = 0

        for measurement in self.dataset.generate_measurements():

            if measurement.measurement_type.value == "Camera":
                self.visual_odometry.track(measurement.image)
                self.time_stamps.append(measurement.time.to_time())
                iteration_number_cam += 1
                print(iteration_number_cam)

            iteration_number += 1
            scale = 1

            if iteration_number_cam > 200:
                break

        plt.plot(range(len(self.visual_odometry.yaw)), np.array(self.visual_odometry.yaw))
        plt.title("Yaw measurements")
        plt.figure(2)
        plot_position(np.array([np.array(self.visual_odometry.North)[:, 0], np.array(self.visual_odometry.East)[:, 0],
                      np.array(self.visual_odometry.Down)[:, 0]]).T, self.ground_truth, self.time_stamps, convert_NED=True)
        plt.figure(3)
        plot_angels(np.array([np.array(self.visual_odometry.roll), np.array(self.visual_odometry.pitch), np.array(
            self.visual_odometry.yaw)]).T, self.ground_truth, self.time_stamps)

        plt.show()


fusion = CameraUwbImuFusion()
fusion.run()
