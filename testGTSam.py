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
        self.pose_variables = []
        self.landmarks_variables = {}
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
    

    def add_UWB_to_graph(self, graph, uwb_measurement, pose_variable):



        landmark, initial_estimate = self.get_UWB_landmark(uwb_measurement)
        measurement_noise = gtsam.noiseModel.Diagonal.Sigmas([uwb_measurement.std])
        graph.add(gtsam.RangeFactor2D(pose_variable, landmark, uwb_measurement.range, measurement_noise))

        if initial_estimate:
            self.isam.update(graph, initial_estimate)


    def get_UWB_landmark(self, uwb_measurement):
        initial_estimate = None
        if uwb_measurement.id not in self.landmarks_variables.keys():
            self.landmarks_variables[uwb_measurement.id] = L(len(self.landmarks_variables.keys()))
            uwb_pos =  self.uwb_positions[uwb_measurement.id]
            initial_estimate = gtsam.Values()
            initial_estimate.insert(self.landmarks_variables[uwb_measurement.id],gtsam.Point2(uwb_pos.x, uwb_pos.y))
        return self.landmarks_variables[uwb_measurement.id], initial_estimate

    def run(self):
        result = self.isam.calculateEstimate()
        print(result)
        X1 = X(1)
        graph = gtsam.NonlinearFactorGraph()
        imu_counter = 0
        for measurement in self.dataset.generate_measurements():

            # Pre integrer states til man når en ny landmark
            # Når man når ny landmark, sett initial value til den forrige staten pluss odometri resultatet
            # Legg så inn landmarket i grafen
            # Legge inn landmarks målinger helt til det kommer en IMU måling
            # TODO: Få lagt inn rett transformasjoner 

            if measurement.measurement_type.value == "UWB":
                #self.add_UWB_to_graph(graph, measurement, X1)
                print("Imu counter", imu_counter)
                imu_counter = 0
                #break
                #print(measurement)
            else:
                imu_counter += 1

        print(graph)
                
        """
        if dataFrame.topic != "/ublox1/fix":
            print(dataFrame.msg)
        """
testing = GtSAMTest()
testing.run()