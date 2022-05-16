import gtsam
from DataSets.extractData import ROSData
from DataSets.extractGt import GroundTruthEstimates
from settings import DATASET_NUMBER
import numpy as np
from scipy.spatial.transform import Rotation as R
from Sensors.CameraSensor.visualOdometry import VisualOdometry
from Plotting.plot_gtsam import plot_position, plot_angels, plot_threedof
import matplotlib.pyplot as plt
import seaborn as sns


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
        sns.set()

    def initialize_graph(self):

        # Pose for pre init
        R_init = R.from_euler("xyz", self.ground_truth.initial_pose(voBruteForce=True)[:3], degrees=False)
        T_init = self.ground_truth.initial_pose(voBruteForce=True)[3:]
        T_init[2] = -0.7

        self.initial_state = np.eye(4)
        self.initial_state[:3, :3] = R_init.as_matrix()
        self.initial_state[:3, 3] = T_init.T

        # Initialize vo
        self.visual_odometry = VisualOdometry()
        self.visual_odometry.update_scale(0.25)

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

            if iteration_number_cam > 2000:
                break

        states = self.visual_odometry.states

        for i in range(len(states)):
            states[i] = self.initial_state @ states[i]

        north = np.array([states[i][0, 3] for i in range(len(states))])
        east = np.array([states[i][1, 3] for i in range(len(states))])
        down = np.array([states[i][2, 3] for i in range(len(states))])

        roll = np.array([R.from_matrix(states[i][:3, :3]).as_euler("xyz")[0] for i in range(len(states))])
        pitch = np.array([R.from_matrix(states[i][:3, :3]).as_euler("xyz")[1] for i in range(len(states))])
        yaw = np.array([R.from_matrix(states[i][:3, :3]).as_euler("xyz")[2] for i in range(len(states))])

        # plt.figure(2)
        #plot_position(np.array([north, east, down]).T, self.ground_truth, self.time_stamps, convert_NED=False)
        # plt.figure(3)
        #plot_angels(np.array([roll, pitch, yaw]).T, self.ground_truth, self.time_stamps)
        plot_threedof(np.array([north, east, down]).T, np.array([roll, pitch, yaw]).T, self.ground_truth, self.time_stamps)
        plt.show()


fusion = CameraUwbImuFusion()
fusion.run()
