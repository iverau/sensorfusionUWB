import gtsam
from DataSets.extractData import ROSData
from DataSets.extractGt import GroundTruthEstimates
from gtsam.symbol_shorthand import X, L, V, B
import numpy as np
from settings import DATASET_NUMBER
from DataTypes.uwb_position import UWB_Ancors_Descriptor

from scipy.spatial.transform import Rotation as R
from Sensors.IMU import IMU

import matplotlib.pyplot as plt
from Utils.gtsam_pose_utils import gtsam_pose_from_result, gtsam_pose_to_numpy, gtsam_landmark_from_results
from Plotting.plot_gtsam import plot_horizontal_trajectory, plot_position, plot_angels
import seaborn as sns

class GtSAMTest:


    def __init__(self) -> None:
        self.dataset: ROSData = ROSData(DATASET_NUMBER)
        isam_params: gtsam.ISAM2Params = gtsam.ISAM2Params()
        isam_params.setFactorization("Cholesky")
        isam_params.setRelinearizeSkip(10)
        self.isam: gtsam.ISAM2 = gtsam.ISAM2(isam_params)
        self.uwb_positions: UWB_Ancors_Descriptor = UWB_Ancors_Descriptor(DATASET_NUMBER)
        self.ground_truth: GroundTruthEstimates = GroundTruthEstimates(DATASET_NUMBER, pre_initialization=True)
        self.imu_params: IMU = IMU()


        # Tracked variables for IMU and UWB
        self.pose_variables: list = []
        self.velocity_variables: list = []
        self.imu_bias_variables: list = []
        self.landmarks_variables: dict = {}
        self.uwb_counter: set = set()
        self.time_stamps: list = []

        # Setting up gtsam values
        self.graph_values: gtsam.Values = gtsam.Values()
        self.factor_graph: gtsam.NonlinearFactorGraph  = gtsam.NonlinearFactorGraph()
        self.initialize_graph()
        #sns.set()



    def initialize_graph(self):

        # Defining the state
        X1 = X(0)
        V1 = V(0)
        B1 = B(0)
        self.pose_variables.append(X1)
        self.velocity_variables.append(V1)
        self.imu_bias_variables.append(B1)

        # Set priors
        prior_noise_x = gtsam.noiseModel.Isotropic.Precisions([2, 2, 2, 0.1, 0.1, 2])
        prior_noise_v = gtsam.noiseModel.Isotropic.Sigma(3, 100.0)
        prior_noise_b = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 5e-05, 5e-05, 5e-05]))

        R_init = R.from_euler("xyz", self.ground_truth.initial_pose()[:3], degrees=False).as_matrix()
        T_init = self.ground_truth.initial_pose()[3:]

        self.current_pose = gtsam.Pose3(gtsam.Rot3(R_init), T_init)
        self.current_velocity = self.ground_truth.initial_velocity()
        self.current_bias = gtsam.imuBias.ConstantBias(np.zeros((3,)), np.zeros((3,))) 
        self.navstate = gtsam.NavState(self.current_pose.rotation(), self.current_pose.translation(), self.current_velocity)

        self.factor_graph.add(gtsam.PriorFactorPose3(X1, self.current_pose, prior_noise_x))
        self.factor_graph.add(gtsam.PriorFactorVector(V1, self.current_velocity, prior_noise_v))
        self.factor_graph.add(gtsam.PriorFactorConstantBias(B1, self.current_bias, prior_noise_b))

        self.graph_values.insert(X1, self.current_pose)
        self.graph_values.insert(V1, self.current_velocity)
        self.graph_values.insert(B1, self.current_bias)
        self.time_stamps.append(self.ground_truth.time[0])
        print("Initial time",self.ground_truth.time[0])

    def add_UWB_to_graph(self, uwb_measurement):
        
        landmark = self.get_UWB_landmark(uwb_measurement)
        measurement_noise = gtsam.noiseModel.Diagonal.Sigmas([uwb_measurement.std])
        self.factor_graph.add(gtsam.RangeFactor3D(self.pose_variables[-1], landmark, uwb_measurement.range, measurement_noise))


    def get_UWB_landmark(self, uwb_measurement):
        self.uwb_counter.add(uwb_measurement.id)
        if uwb_measurement.id not in self.landmarks_variables.keys():
            self.landmarks_variables[uwb_measurement.id] = L(len(self.landmarks_variables.keys()))

            # Creates an initial estimate of the landmark pose
            self.graph_values.insert(self.landmarks_variables[uwb_measurement.id], gtsam.Point3(self.uwb_positions[uwb_measurement.id].x, self.uwb_positions[uwb_measurement.id].y, self.uwb_positions[uwb_measurement.id].z))
            self.factor_graph.add(gtsam.PriorFactorVector(self.landmarks_variables[uwb_measurement.id], gtsam.Point3(self.uwb_positions[uwb_measurement.id].x, self.uwb_positions[uwb_measurement.id].y, self.uwb_positions[uwb_measurement.id].z), gtsam.noiseModel.Isotropic.Sigma(3, 0.0001)))


        return self.landmarks_variables[uwb_measurement.id]


    def reset_pose_graph_variables(self):
        self.graph_values = gtsam.Values()
        self.factor_graph = gtsam.NonlinearFactorGraph()
        self.uwb_counter = set()

    def pre_integrate_imu_measurement(self, imu_measurements):
        currentBias = gtsam.imuBias.ConstantBias(np.zeros((3,)), np.zeros((3,)))
        summarized_measurement = gtsam.PreintegratedImuMeasurements(self.imu_params.preintegration_param, currentBias)

        deltaT = 1 / self.dataset.dataset_settings.imu_frequency

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
        
        self.factor_graph.add(
            gtsam.BetweenFactorConstantBias(
                self.imu_bias_variables[-2],
                self.imu_bias_variables[-1],
                gtsam.imuBias.ConstantBias(np.zeros((3, 1)), np.zeros((3, 1))),
                gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(len(imu_measurements)) * self.imu_params.sigmaBetweenBias)
            )
        )
        
        self.navstate = integrated_measurement.predict(self.navstate, self.current_bias)
        self.graph_values.insert(self.pose_variables[-1], self.navstate.pose())
        self.graph_values.insert(self.velocity_variables[-1], self.navstate.velocity())
        self.graph_values.insert(self.imu_bias_variables[-1], self.current_bias)
        self.factor_graph.add(gtsam.PriorFactorPose3(self.pose_variables[-1], self.navstate.pose(), gtsam.noiseModel.Diagonal.Sigmas(len(imu_measurements)*imu_measurements[0].variance_vector())))

    def add_imu_factor_gnss(self, integrated_measurement, imu_measurements):
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

    def add_GNSS_to_graph(self, factor_graph, measurement):
        pose = gtsam.Pose3(self.current_pose.rotation(), measurement.position)
        factor_graph.add(gtsam.PriorFactorPose3(self.pose_variables[-1], pose, measurement.noise_model))
        return pose


    def run(self):
        # Dummy variable for storing imu measurements
        imu_measurements = []
        iteration_number = 0
        gnss_counter = 0
        # 10 secs of GNSS
        for measurement in self.dataset.generate_initialization_gnss_imu():
            if measurement.measurement_type.value == "GNSS":
                if imu_measurements:
                    self.time_stamps.append(measurement.time.to_time())
                    integrated_measurement = self.pre_integrate_imu_measurement(imu_measurements)
                    self.add_imu_factor_gnss(integrated_measurement, imu_measurements) 

                    # Reset the IMU measurement list
                    imu_measurements = []
                    self.isam.update()

                gnss_pose = self.add_GNSS_to_graph(self.factor_graph, measurement)
                gnss_counter += 1
                self.graph_values.insert(self.pose_variables[-1], gnss_pose)
                self.graph_values.insert(self.velocity_variables[-1], self.current_velocity)
                self.graph_values.insert(self.imu_bias_variables[-1], self.current_bias)


            elif measurement.measurement_type.value == "IMU":
                imu_measurements.append(measurement)

            if gnss_counter == 2:
                print("Her :)")
                self.isam.update(self.factor_graph, self.graph_values)
                result = self.isam.calculateEstimate()

                self.reset_pose_graph_variables()
                self.current_pose = result.atPose3(self.pose_variables[-1])
                self.current_velocity = result.atVector(self.velocity_variables[-1])
                self.current_bias = result.atConstantBias(self.imu_bias_variables[-1])
                gnss_counter = 0



        imu_measurements = []
        for measurement in self.dataset.generate_measurements():
            
            if measurement.measurement_type.value == "UWB":
                if imu_measurements:
                    self.time_stamps.append(measurement.time.to_time())
                    integrated_measurement = self.pre_integrate_imu_measurement(imu_measurements)
                    self.add_imu_factor(integrated_measurement, imu_measurements) 

                    # Reset the IMU measurement list
                    imu_measurements = []
                    self.isam.update()

                self.add_UWB_to_graph(measurement)


            elif measurement.measurement_type.value == "IMU":
                # Store the IMU factors unntil a new UWB measurement is recieved
                imu_measurements.append(measurement)
            
            iteration_number += 1
            print("Iteration", iteration_number, len(self.pose_variables), len(self.time_stamps))




            # Update ISAM with graph and initial_values
            if len(self.uwb_counter) == 3:
                
                self.isam.update(self.factor_graph, self.graph_values)
                result = self.isam.calculateEstimate()
                positions, eulers = gtsam_pose_from_result(result)

                
                # Reset the graph and initial values
                self.reset_pose_graph_variables()
                
                
                self.current_pose = result.atPose3(self.pose_variables[-1]) 
                self.current_velocity = result.atVector(self.velocity_variables[-1])
                self.current_bias = result.atConstantBias(self.imu_bias_variables[-1])

                #print("Current pose", self.current_pose)
                self.navstate = gtsam.NavState(self.current_pose.rotation(), self.current_pose.translation(), self.current_velocity)

                if len(self.pose_variables) > 30:
                    break

        result = self.isam.calculateBestEstimate()
        positions, eulers = gtsam_pose_from_result(result)



        print("\n-- Plot pose")
        plt.figure(1)
        plot_horizontal_trajectory(positions, [-200, 200], [-200, 200], gtsam_landmark_from_results(result, self.landmarks_variables.values()), self.ground_truth)
        plt.figure(2)
        plot_position(positions, self.ground_truth, self.time_stamps)
        plt.figure(3)
        plot_angels(eulers, self.ground_truth, self.time_stamps)

        plt.show()

testing = GtSAMTest()
testing.run()