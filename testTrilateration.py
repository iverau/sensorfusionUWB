from math import degrees
from DataSets.extractData import RosDataTrilateration
from DataSets.extractGt import GroundTruthEstimates
from settings import DATASET_NUMBER
from Sensors.IMU import IMU
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L, V, B
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from Utils.gtsam_pose_utils import gtsam_pose_from_result, gtsam_pose_to_numpy, gtsam_landmark_from_results
from Plotting.plot_gtsam import plot_horizontal_trajectory


class TrilaterationEstimates:

    # Requires dataset_sensor_config to be IMU_TRI

    def __init__(self) -> None:
        self.dataset = RosDataTrilateration(4)
        isam_params = gtsam.ISAM2Params()
        isam_params.setFactorization("CHOLESKY")
        isam_params.setRelinearizeSkip(10)
        self.isam = gtsam.ISAM2(isam_params)
        self.ground_truth = GroundTruthEstimates(DATASET_NUMBER)
        self.imu_params = IMU()

        # Tracked variables for IMU and UWB
        self.pose_variables = []
        self.velocity_variables = []
        self.imu_bias_variables = []
        self.landmarks_variables = {}
        self.uwb_counter = 0

        # Setting up gtsam values
        self.graph_values = gtsam.Values()
        self.factor_graph = gtsam.NonlinearFactorGraph()
        self.initialize_graph()
        print(self.ground_truth.initial_pose())


    def initialize_graph(self):

        # Defining the state
        X1 = X(0)
        V1 = V(0)
        B1 = B(0)
        self.pose_variables.append(X1)
        self.velocity_variables.append(V1)
        self.imu_bias_variables.append(B1)

        # Set prior noises
        prior_noise_x = gtsam.noiseModel.Isotropic.Precisions([0.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5])
        prior_noise_v = gtsam.noiseModel.Isotropic.Sigma(3, 1000.0)
        prior_noise_b = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 5e-05, 5e-05, 5e-05]))

        # Calculate the initial pose
        R_init = R.from_euler("xyz", self.ground_truth.initial_pose()[:3], degrees=False).as_matrix()
        T_init = self.ground_truth.initial_pose()[3:]

        # Set the prior states 
        #TODO: Fix the naming convention here
        self.current_pose = gtsam.Pose3(gtsam.Rot3(R_init), T_init)
        self.current_velocity = self.ground_truth.initial_velocity()
        self.current_bias = gtsam.imuBias.ConstantBias(np.zeros((3,)), np.zeros((3,))) 

        print(self.current_pose)

        # Add the priors to the graph and value list
        self.factor_graph.add(gtsam.PriorFactorPose3(X1, self.current_pose, prior_noise_x))
        self.factor_graph.add(gtsam.PriorFactorVector(V1, self.current_velocity, prior_noise_v))
        self.factor_graph.add(gtsam.PriorFactorConstantBias(B1, self.current_bias, prior_noise_b))
        self.graph_values.insert(X1, self.current_pose)
        self.graph_values.insert(V1, self.current_velocity)
        self.graph_values.insert(B1, self.current_bias)

    def reset_pose_graph_variables(self):
        self.graph_values = gtsam.Values()
        self.factor_graph = gtsam.NonlinearFactorGraph()
        self.uwb_counter = 0

    def pre_integrate_imu_measurement(self, imu_measurements):
        # Calculate the preintegrated imu measurement parameters
        currentBias = gtsam.imuBias.ConstantBias(np.zeros((3,)), np.zeros((3,)))
        summarized_measurement = gtsam.PreintegratedImuMeasurements(self.imu_params.preintegration_param, currentBias)
        deltaT = 1 / self.dataset.dataset_settings.imu_frequency

        # Calculate the summarized measurement
        for measurement in imu_measurements:
            summarized_measurement.integrateMeasurement(measurement.linear_vel, measurement.angular_vel, deltaT)

        return summarized_measurement

    def add_imu_factor(self, integrated_measurement, imu_measurements):
        # Create new state variables
        self.pose_variables.append(X(len(self.pose_variables)))
        self.velocity_variables.append(V(len(self.velocity_variables)))
        self.imu_bias_variables.append(B(len(self.imu_bias_variables)))

        # Add the new factors to the graph
        self.factor_graph.add(gtsam.ImuFactor(
            self.pose_variables[-2],
            self.velocity_variables[-2],
            self.pose_variables[-1],
            self.velocity_variables[-1],
            self.imu_bias_variables[-2],
            integrated_measurement
        ))
        
        # Add bias constraints
        self.factor_graph.add(
            gtsam.BetweenFactorConstantBias(
                self.imu_bias_variables[-2],
                self.imu_bias_variables[-1],
                gtsam.imuBias.ConstantBias(np.zeros((3, 1)), np.zeros((3, 1))),
                gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(len(imu_measurements)) * self.imu_params.sigmaBetweenBias)
            )
        )
    
    def add_UWB_to_graph(self, graph, uwb_measurement):
        pose = gtsam.Pose3(self.current_pose.rotation(), uwb_measurement.position)
        graph.add(gtsam.PriorFactorPose3(self.pose_variables[-1], pose, uwb_measurement.noise_model))
        
        return pose

    def run(self):
        imu_measurements = []
        for measurement in self.dataset.generate_trilateration_combo_measurements():
            # TODO: Få lagt inn rett transformasjoner 


            # If the measurment is an UWB measurement, calculate the preintegrated measurement and add the UWB measurment
            if measurement.measurement_type.value == "UWB_Tri":
                # Add the imu measurment if present
                if imu_measurements:
                    integrated_measurement = self.pre_integrate_imu_measurement(imu_measurements)
                    self.add_imu_factor(integrated_measurement, imu_measurements) 

                    # Reset the IMU measurement list
                    imu_measurements = []
                
                # Add the UWB pose
                uwb_pose = self.add_UWB_to_graph(self.factor_graph, measurement)
                self.uwb_counter += 1
                self.graph_values.insert(self.pose_variables[-1], uwb_pose)
                self.graph_values.insert(self.velocity_variables[-1], self.current_velocity)
                self.graph_values.insert(self.imu_bias_variables[-1], self.current_bias)


            # Store the IMU factors unntil a new UWB measurement is recieved
            elif measurement.measurement_type.value == "IMU":
                imu_measurements.append(measurement)
                
        
            # Update ISAM with graph and initial_values
            if self.uwb_counter == 2:
                self.isam.update(self.factor_graph, self.graph_values)
                result = self.isam.calculateEstimate()

                self.reset_pose_graph_variables()
                self.current_pose = result.atPose3(self.pose_variables[-1])
                self.current_velocity = result.atVector(self.velocity_variables[-1])
                self.current_bias = result.atConstantBias(self.imu_bias_variables[-1])

                # TODO: Fix this bug
                if len(self.imu_bias_variables)  == 569:
                    break

                
        positions, eulers = gtsam_pose_from_result(result)
        print("\n-- Plot pose")
        print("Poses ",gtsam_landmark_from_results(result, self.landmarks_variables))

        plt.figure(1)
        plot_horizontal_trajectory(positions, [-100, 20], [-160, -65], self.landmarks_variables)
        plt.show()


test = TrilaterationEstimates()
data = test.run()