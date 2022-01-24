import gtsam
from DataSets.extractData import ROSData
from gtsam.symbol_shorthand import X, L
import numpy as np

class GtSAMTest:
    DATASET_NUMBER = 4

    def __init__(self) -> None:
        self.dataset = ROSData(self.DATASET_NUMBER)
        isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(isam_params)

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

        #TODO: Få inn UWB posisjonene som initial values
        #TODO: Få lagt inn rett transformasjoner 

        landmark = self.get_UWB_landmark(uwb_measurement)
        measurement_noise = gtsam.noiseModel.Diagonal.Sigmas([uwb_measurement.std])
        graph.add(gtsam.RangeFactor2D(pose_variable, landmark, uwb_measurement.range, measurement_noise))

    def get_UWB_landmark(self, uwb_measurement):
        if uwb_measurement.id not in self.landmarks_variables.keys():
            self.landmarks_variables[uwb_measurement.id] = L(len(self.landmarks_variables.keys()))
        return self.landmarks_variables[uwb_measurement.id]

    def run(self):
        result = self.isam.calculateEstimate()
        print(result)
        X1 = X(1)
        graph = gtsam.NonlinearFactorGraph()
        for measurement in self.dataset.generate_measurements():
            if measurement.measurement_type.value == "UWB":
                self.add_UWB_to_graph(graph, measurement, X1)

        print(graph)
                
        """
        if dataFrame.topic != "/ublox1/fix":
            print(dataFrame.msg)
        """
testing = GtSAMTest()
testing.run()