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

class TrilaterationEstimates:

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
        self.initial_values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.initialize_graph()


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


        self.iniial_velocity = self.ground_truth.initial_velocity()
        self.initial_bias = gtsam.imuBias.ConstantBias(np.zeros((3,)), np.zeros((3,))) 

        self.graph.add(gtsam.PriorFactorPose3(X1, self.initial_pose, prior_noise_x))
        self.graph.add(gtsam.PriorFactorVector(V1, self.iniial_velocity, prior_noise_v))
        self.graph.add(gtsam.PriorFactorConstantBias(B1, self.initial_bias, prior_noise_b))

        self.initial_values.insert(X1, self.initial_pose)
        self.initial_values.insert(V1, self.iniial_velocity)
        self.initial_values.insert(B1, self.initial_bias)

        #self.isam.update(graph, initial_estimates)

    def reset_pose_graph_variables(self):
        self.initial_values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.uwb_counter = 0

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
    
    def add_UWB_to_graph(self, graph, uwb_measurement):
        print("I was here")
        pose = gtsam.Pose3(self.initial_pose.rotation(), uwb_measurement.position)
        graph.add(gtsam.PriorFactorPose3(self.pose_variables[-1], pose, uwb_measurement.noise_model))
        
        return pose
        #print("ISAM object", self.isam)
        #landmark = self.get_UWB_landmark(uwb_measurement)
        #measurement_noise = gtsam.noiseModel.Diagonal.Sigmas([uwb_measurement.std])
        #graph.add(gtsam.RangeFactor3D(self.pose_variables[-1], landmark, uwb_measurement.range, measurement_noise))

    def run(self):
        imu_measurements = []
        for measurement in self.dataset.generate_trilateration_combo_measurements():
            #print(measurement)
            # Pre integrer states til man når en ny landmark
            # Når man når ny landmark, sett initial value til den forrige staten pluss odometri resultatet
            # Legg så inn landmarket i grafen
            # Legge inn landmarks målinger helt til det kommer en IMU måling
            # Oppdater initial values når alle UWB nodene er sett
            # TODO: Få lagt inn rett transformasjoner 
            #print(measurement)

            if measurement.measurement_type.value == "UWB_Tri":
                if imu_measurements:
                    integrated_measurement = self.pre_integrate_imu_measurement(imu_measurements)
                    self.add_imu_factor(integrated_measurement, imu_measurements) 

                    # Reset the IMU measurement list
                    imu_measurements = []
                
                uwb_pose = self.add_UWB_to_graph(self.graph, measurement)
                self.uwb_counter += 1
                self.initial_values.insert(self.pose_variables[-1], uwb_pose)
                self.initial_values.insert(self.velocity_variables[-1], self.iniial_velocity)
                self.initial_values.insert(self.imu_bias_variables[-1], self.initial_bias)
                # TODO: What this should do

            elif measurement.measurement_type.value == "IMU":
                # Store the IMU factors unntil a new UWB measurement is recieved
                imu_measurements.append(measurement)
                
        
            # Update ISAM with graph and initial_values
            #print("Counter",self.uwb_counter)
            if self.uwb_counter == 2:
                #print(self.ground_truth.initial_pose())
                #print(self.initial_pose)
                print("Her :)")
                #print(self.initial_values)
                self.isam.update(self.graph, self.initial_values)
                
                result = self.isam.calculateEstimate()

                self.reset_pose_graph_variables()
                self.initial_pose = result.atPose3(self.pose_variables[-1])
                self.iniial_velocity = result.atVector(self.velocity_variables[-1])
                self.initial_bias = result.atConstantBias(self.imu_bias_variables[-1])
                if len(self.imu_bias_variables)  == 569:
                    break
                # Reset the graph and initial values
                
        positions, eulers = gtsam_pose_from_result(result)
        print("\n-- Plot pose")
        plt.figure(1)
        plot_horizontal_trajectory(positions, [-20, 20], [-100, -65])
        plt.show()

def gtsam_pose_from_result(gtsam_result):
    poses = gtsam.utilities.allPose3s(gtsam_result)
    keys = gtsam.KeyVector(poses.keys())

    positions, eulers = [], []
    for key in keys:
        if gtsam_result.exists(key):
            pose = gtsam_result.atPose3(key)
            pos, euler = gtsam_pose_to_numpy(pose)
            positions.append(pos)
            eulers.append(euler)
    positions = np.array(positions)
    eulers = np.array(eulers)
    return positions, eulers

def gtsam_pose_to_numpy(gtsam_pose):
    """Convert GTSAM pose to numpy arrays 
    (position, orientation)"""
    position = np.array([
        gtsam_pose.x(),
        gtsam_pose.y(),
        gtsam_pose.z()])
    euler = np.array([
        gtsam_pose.rotation().roll(),
        gtsam_pose.rotation().pitch(),
        gtsam_pose.rotation().yaw()])
    return position, euler


def plot_horizontal_trajectory(pose_estimate, x_lim, y_lim):
    plt.suptitle("Horizontal trajectory")

    plt.plot(pose_estimate[:, 1], pose_estimate[:, 0], color="blue")
    #plt.plot(ground_truth[:, 1], ground_truth[:, 0], color="gray", linestyle="dashed")
    plt.xlabel("y [m]")
    plt.ylabel("x [m]")
    plt.legend(["estimate", "ground truth"])
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.grid()


# Noise priors på uwb
# rotasjonene på uwb ser merkelige ut (sykt små tall)

test = TrilaterationEstimates()
data = test.run()