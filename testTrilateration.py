from DataSets.extractData import RosDataTrilateration
import gtsam
from DataSets.extractData import RosDataTrilateration

class TrilaterationEstimates:

    def __init__(self) -> None:
        self.dataset = RosDataTrilateration(4)

        # Tracked variables for IMU and UWB
        self.pose_variables = []
        self.velocity_variables = []
        self.imu_bias_variables = []
        self.landmarks_variables = {}

        # Setting up gtsam values
        #self.initial_values = gtsam.Values()
        #self.graph = gtsam.NonlinearFactorGraph()
        #self.initialize_graph()

    def testTrilaterationData(self):

        for measurement in self.dataset.generate_trilateration_combo_measurements():
            print(measurement)
        """
        data = self.dataset.generate_trilateration_measurement()
        print("Data pos:",data)
        x = data["pos_sensor"][0][0][0][0]
        y = data["pos_sensor"][0][0][1][0]
        z = data["pos_sensor"][0][0][4][0]
        time = data["pos_sensor"][0][0][6][0]
        print("Data keys", y)
        """

test = TrilaterationEstimates()
data = test.testTrilaterationData()