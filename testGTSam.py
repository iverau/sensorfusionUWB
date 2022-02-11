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
from Utils.gtsam_pose_utils import gtsam_pose_from_result, gtsam_pose_to_numpy
from Plotting.plot_gtsam import plot_horizontal_trajectory


class GtSAMTest:


    def __init__(self) -> None:
        self.dataset = ROSData(DATASET_NUMBER)
        isam_params = gtsam.ISAM2Params()
        isam_params.setFactorization("QR")
        isam_params.setRelinearizeSkip(10)
        self.isam = gtsam.ISAM2(isam_params)
        self.uwb_positions = UWB_Ancors_Descriptor(DATASET_NUMBER)
        self.ground_truth = GroundTruthEstimates(DATASET_NUMBER)
        self.imu_params = IMU()

        # Tracked variables for IMU and UWB
        self.pose_variables = []
        self.velocity_variables = []
        self.imu_bias_variables = []
        self.landmarks_variables = {}

        # Setting up gtsam values
        self.initial_values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.initialize_graph()
        
        # Dummy variables for counting amount of seen uwbs in the current pose graph
        self.uwb_counter = set()


    def initialize_graph(self):

        graph = gtsam.NonlinearFactorGraph()

        # Defining the state
        X1 = X(0)
        V1 = V(0)
        B1 = B(0)
        self.pose_variables.append(X1)
        self.velocity_variables.append(V1)
        self.imu_bias_variables.append(B1)

        # Set priors
        prior_noise_x = gtsam.noiseModel.Isotropic.Precisions([0.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5])
        prior_noise_v = gtsam.noiseModel.Isotropic.Sigma(3, 1000.0)
        prior_noise_b = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 5e-05, 5e-05, 5e-05]))

        R_init = R.from_euler("xyz", self.ground_truth.initial_pose()[:3], degrees=False).as_matrix()
        T_init = self.ground_truth.initial_pose()[3:]
        self.initial_pose = gtsam.Pose3(gtsam.Rot3(R_init), T_init)


        #self.initial_pose = gtsam.Pose3(self.ground_truth.initial_pose())
        self.iniial_velocity = self.ground_truth.initial_velocity()
        self.initial_bias = gtsam.imuBias.ConstantBias(np.zeros((3,)), np.zeros((3,))) 

        self.navstate = gtsam.NavState(self.initial_pose.rotation(), self.initial_pose.translation(), self.iniial_velocity)

        self.graph.add(gtsam.PriorFactorPose3(X1, self.initial_pose, prior_noise_x))
        self.graph.add(gtsam.PriorFactorVector(V1, self.iniial_velocity, prior_noise_v))
        self.graph.add(gtsam.PriorFactorConstantBias(B1, self.initial_bias, prior_noise_b))

        self.initial_values.insert(X1, self.initial_pose)
        self.initial_values.insert(V1, self.iniial_velocity)
        self.initial_values.insert(B1, self.initial_bias)

        #self.isam.update(graph, initial_estimates)
    

    def add_UWB_to_graph(self, graph, uwb_measurement):
        
        #print("ISAM object", self.isam)
        landmark = self.get_UWB_landmark(uwb_measurement)
        measurement_noise = gtsam.noiseModel.Diagonal.Sigmas([uwb_measurement.std])
        self.graph.add(gtsam.RangeFactor3D(self.pose_variables[-1], landmark, uwb_measurement.range, measurement_noise))


    def get_UWB_landmark(self, uwb_measurement):
        self.uwb_counter.add(uwb_measurement.id)
        if uwb_measurement.id not in self.landmarks_variables.keys():
            self.landmarks_variables[uwb_measurement.id] = L(len(self.landmarks_variables.keys()))

            # Creates an initial estimate of the landmark pose
            self.initial_values.insert(self.landmarks_variables[uwb_measurement.id], gtsam.Point3(self.uwb_positions[uwb_measurement.id].x, self.uwb_positions[uwb_measurement.id].y, self.uwb_positions[uwb_measurement.id].z))
            self.graph.add(gtsam.PriorFactorPoint3(self.landmarks_variables[uwb_measurement.id], gtsam.Point3(self.uwb_positions[uwb_measurement.id].x, self.uwb_positions[uwb_measurement.id].y, self.uwb_positions[uwb_measurement.id].z), gtsam.noiseModel.Isotropic.Sigma(3, 1000.0)))
            #self.graph.add(gtsam.PriorFactorPose3(X1, self.initial_pose, prior_noise_x))

        return self.landmarks_variables[uwb_measurement.id]


    def reset_pose_graph_variables(self):
        self.initial_values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
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
        self.graph.add(gtsam.ImuFactor(
            self.pose_variables[-2],
            self.velocity_variables[-2],
            self.pose_variables[-1],
            self.velocity_variables[-1],
            self.imu_bias_variables[-2],
            integrated_measurement
        ))
        
        self.graph.add(
            gtsam.BetweenFactorConstantBias(
                self.imu_bias_variables[-2],
                self.imu_bias_variables[-1],
                gtsam.imuBias.ConstantBias(np.zeros((3, 1)), np.zeros((3, 1))),
                gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(len(imu_measurements)) * self.imu_params.sigmaBetweenBias)
            )
        )
        
        self.navstate = integrated_measurement.predict(self.navstate, self.initial_bias)
        self.initial_values.insert(self.pose_variables[-1], self.navstate.pose())
        self.initial_values.insert(self.velocity_variables[-1], self.navstate.velocity())
        self.initial_values.insert(self.imu_bias_variables[-1], self.initial_bias)
        self.graph.add(gtsam.PriorFactorPose3(self.pose_variables[-1], self.navstate.pose(), gtsam.noiseModel.Diagonal.Precisions(np.array([0.0, 0.0, 0.0, 1.0 / 1 ** 2, 1.0 / 1 ** 2, 1.0 / 9 ** 2]))))


    def run(self):
        # Dummy variable for storing imu measurements
        imu_measurements = []

        for measurement in self.dataset.generate_measurements():
            if measurement.measurement_type.value == "UWB":
                integrated_measurement = None
                if imu_measurements:
                    integrated_measurement = self.pre_integrate_imu_measurement(imu_measurements)
                    self.add_imu_factor(integrated_measurement, imu_measurements) 

                    # Reset the IMU measurement list
                    imu_measurements = []

                self.add_UWB_to_graph(self.graph, measurement)

            elif measurement.measurement_type.value == "IMU":
                # Store the IMU factors unntil a new UWB measurement is recieved
                imu_measurements.append(measurement)
                
        
            # Update ISAM with graph and initial_values
            if len(self.uwb_counter) == 5:
                print("Her :)", len(self.pose_variables))
                #print(self.graph)
                #print(self.initial_values)
                self.isam.update(self.graph, self.initial_values)
                result = self.isam.calculateEstimate()
                
                # Reset the graph and initial values
                self.reset_pose_graph_variables()
                
                self.initial_pose = result.atPose3(self.pose_variables[-1]) 
                self.iniial_velocity = result.atVector(self.velocity_variables[-1])
                self.initial_bias = result.atConstantBias(self.imu_bias_variables[-1])

                if len(self.pose_variables)  > 569:
                    break

        positions, eulers = gtsam_pose_from_result(result)
        print("\n-- Plot pose")
        plt.figure(1)
        plot_horizontal_trajectory(positions, [-200, 200], [-200, 200])
        plt.show()

testing = GtSAMTest()
testing.run()