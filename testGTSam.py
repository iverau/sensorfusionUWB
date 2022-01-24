import gtsam
from DataSets.extractData import ROSData
from gtsam.symbol_shorthand import X, L


class GtSAMTest:
    DATASET_NUMBER = 4

    def __init__(self) -> None:
        self.dataset = ROSData(self.DATASET_NUMBER)
        isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(isam_params)

        self.pose_variables = []
        self.landmarks_variables = []


    def initialize_graph(self):

        graph = gtsam.NonlinearFactorGraph()

        X1 = X(1)

        self.pose_variables.append(X1)

    
    def run(self):
        for dataFrame in self.dataset.generate_measurements():
            print(dataFrame)
            """
            if dataFrame.topic != "/ublox1/fix":
                print(dataFrame.msg)
            """
testing = GtSAMTest()
testing.run()