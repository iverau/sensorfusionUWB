import gtsam
from DataSets.extractData import ROSData
from DataSets.extractGt import GroundTruthEstimates
from DataTypes.uwb_position import UWB_Ancors_Descriptor
from Sensors.IMU import IMU
from Sensors.GNSS import GNSS
from settings import DATASET_NUMBER
from gtsam.symbol_shorthand import X, L, V, B


class CameraUwbImuFusion:

    def __init__(self) -> None:
        self.dataset = ROSData(DATASET_NUMBER)
        isam_params: gtsam.ISAM2Params = gtsam.ISAM2Params()
        isam_params.setFactorization("QR")
        isam_params.setRelinearizeSkip(1)
        self.isam: gtsam.ISAM2 = gtsam.ISAM2(isam_params)
        self.uwb_positions: UWB_Ancors_Descriptor = UWB_Ancors_Descriptor(DATASET_NUMBER)
        self.ground_truth: GroundTruthEstimates = GroundTruthEstimates(DATASET_NUMBER, pre_initialization=True)
        self.imu_params: IMU = IMU()
        self.gnss_params: GNSS = GNSS()

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
        self.prior_noise_x = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_POSE_SIGMAS)
        self.prior_noise_v = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_VEL_SIGMAS)
        self.prior_noise_b = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_BIAS_SIGMAS)
        R_init = R.from_euler("xyz", self.ground_truth.initial_pose()[:3], degrees=False).as_matrix()
        T_init = self.ground_truth.initial_pose()[3:]
        T_init[2] = DOWN_INITIAL_VALUE

        self.current_pose = gtsam.Pose3(gtsam.Rot3(R_init), T_init)


        self.current_velocity = self.ground_truth.initial_velocity()
        self.current_bias = gtsam.imuBias.ConstantBias(np.zeros((3,)), BIAS_INITIAL_VALUE) 
        self.navstate = gtsam.NavState(self.current_pose.rotation(), self.current_pose.translation(), self.current_pose.rotation().matrix().T @ self.current_velocity)

        self.factor_graph.add(gtsam.PriorFactorPose3(X1, self.current_pose, self.prior_noise_x))
        self.factor_graph.add(gtsam.PriorFactorVector(V1, self.current_velocity, self.prior_noise_v))
        self.factor_graph.add(gtsam.PriorFactorConstantBias(B1, self.current_bias, self.prior_noise_b))

        self.graph_values.insert(X1, self.current_pose)
        self.graph_values.insert(V1, self.current_velocity)
        self.graph_values.insert(B1, self.current_bias)
        self.time_stamps.append(self.ground_truth.time[0])

        def run(self):
            imu_measurements = []
            iteration_number = 0

            for measurement in self.dataset.generate_measurements():

                if measurement.measurement_type.value == "Camera":
                    print("Camera ")


fusion = CameraUwbImuFusion()
fusion.run()