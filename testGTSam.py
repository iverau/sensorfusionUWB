import gtsam
from DataSets.extractData import ROSData
from gtsam.symbol_shorthand import X, L
import numpy as np
from settings import DATASET_NUMBER
from DataTypes.uwb_position import UWB_Ancors_Descriptor

class GtSAMTest:


    def __init__(self) -> None:
        self.dataset = ROSData(DATASET_NUMBER)
        isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(isam_params)
        self.uwb_positions = UWB_Ancors_Descriptor(DATASET_NUMBER)

        # Tracked variables for IMU and UWB
        self.pose_variables = []
        self.velocity_variables = []
        self.imu_bias_variables = []
        self.landmarks_variables = {}

        # Setting up gtsam values
        self.initial_values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.initialize_graph()


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

    def pre_integrate_imu_measurement(self, imu_measurements):
        pass

    def run(self):

        imu_counter = 0
        uwb_counter = set()
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
                    # Reset the IMU measurement list
                    imu_measurements = []
                
                self.add_UWB_to_graph(self.graph, measurement)
                # TODO: What this should do
                """
                if imu measurements are avaialble
                    - pre integrate it 
                    - create a new state for it
                else:
                    - Add UWB measurement to current state

                """

                #self.add_UWB_to_graph(graph, measurement, X1)
                #print("Imu counter", imu_counter)
                imu_counter = 0
                print(imu_measurements)
                #break
                #print(measurement)
                #print("UWB sett", self.initial_values.size())


            elif measurement.measurement_type.value == "IMU":
                # Store the IMU factors unntil a new UWB measurement is recieved
                imu_measurements.append(measurement)
                
        
            # Update ISAM with graph and initial_values
            if len(uwb_counter) == 5:
                uwb_counter = set()
                self.isam.update(self.graph,self.initial_values)
                
                # Reset the graph and initial values
                self.initial_values = gtsam.Values()
                self.graph = gtsam.NonlinearFactorGraph()

        #print(graph)
        #print(self.initial_values)
        #self.isam.update(graph, self.initial_values)
                
        """
        if dataFrame.topic != "/ublox1/fix":
            print(dataFrame.msg)
        """
testing = GtSAMTest()
testing.run()