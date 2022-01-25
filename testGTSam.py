import gtsam
from DataSets.extractData import ROSData
from gtsam.symbol_shorthand import X, L, V, B
import numpy as np
from settings import DATASET_NUMBER
from DataTypes.uwb_position import UWB_Ancors_Descriptor

from Sensors.IMU import IMU

class GtSAMTest:


    def __init__(self) -> None:
        self.dataset = ROSData(DATASET_NUMBER)
        isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(isam_params)
        self.uwb_positions = UWB_Ancors_Descriptor(DATASET_NUMBER)

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

        X1 = X(1)

        self.pose_variables.append(X1)

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
        graph.add(gtsam.PriorFactorPose2(X1, gtsam.Pose2(0.0, 0.0, 0.0), prior_noise))

        initial_estimates = gtsam.Values()
        initial_estimates.insert(X1, gtsam.Pose2(0.0, 0.0, 0.0))

        self.isam.update(graph, initial_estimates)
    

    def add_UWB_to_graph(self, graph, uwb_measurement):
        
        #print("ISAM object", self.isam)
        landmark = self.get_UWB_landmark(uwb_measurement)
        measurement_noise = gtsam.noiseModel.Diagonal.Sigmas([uwb_measurement.std])
        graph.add(gtsam.RangeFactor2D(self.pose_variable[-1], landmark, uwb_measurement.range, measurement_noise))


    def get_UWB_landmark(self, uwb_measurement):
        initial_estimate = None
        if uwb_measurement.id not in self.landmarks_variables.keys():
            self.landmarks_variables[uwb_measurement.id] = L(len(self.landmarks_variables.keys()))

            # Creates an initial estimate of the landmark pose
            self.initial_values.insert(self.landmarks_variables[uwb_measurement.id], gtsam.Point2(self.uwb_positions[uwb_measurement.id].x, self.uwb_positions[uwb_measurement.id].y))
            
        return self.landmarks_variables[uwb_measurement.id]


    def reset_pose_graph_variables(self):
        self.initial_values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.uwb_counter = set()

    def pre_integrate_imu_measurement(self, imu_measurements):
        currentBias = gtsam.imuBias.ConstantBias(np.zeros((3,)), np.zeros((3,)))
        summarized_measurement = gtsam.PreintegratedImuMeasurement(self.imu_params.preintegration_param, currentBias)

        deltaT = 0.1

        for measurement in imu_measurements:
            summarized_measurement.integrateMeasurement(measurement.linear_vel, measurement.angular_vel, deltaT)

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


    def run(self):
        # Dummy variable for storing imu measurements
        imu_measurements = []

        for measurement in self.dataset.generate_measurements():
            
            # Pre integrer states til man når en ny landmark
            # Når man når ny landmark, sett initial value til den forrige staten pluss odometri resultatet
            # Legg så inn landmarket i grafen
            # Legge inn landmarks målinger helt til det kommer en IMU måling
            # Oppdater initial values når alle UWB nodene er sett
            # TODO: Få lagt inn rett transformasjoner 

            if measurement.measurement_type.value == "UWB":
                if imu_measurements:
                    integrated_measurement = self.pre_integrate_imu_measurement(imu_measurements)
                    self.add_imu_factor(integrated_measurement, imu_measurements) 

                    # Reset the IMU measurement list
                    imu_measurements = []
                
                self.add_UWB_to_graph(self.graph, measurement)
                # TODO: What this should do

            elif measurement.measurement_type.value == "IMU":
                # Store the IMU factors unntil a new UWB measurement is recieved
                imu_measurements.append(measurement)
                
        
            # Update ISAM with graph and initial_values
            if len(self.uwb_counter) == 5:
                self.isam.update(self.graph, self.initial_values)
                
                # Reset the graph and initial values
                self.reset_pose_graph_variables()

testing = GtSAMTest()
testing.run()