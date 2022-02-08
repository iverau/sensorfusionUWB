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
        self.initial_values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.initialize_graph()