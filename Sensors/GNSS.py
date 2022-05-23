import gtsam
import numpy as np


class GNSS:
    def __init__(self):
        self.setup_gnss_params()

    def setup_gnss_params(self):
        scaling = 1000  # 1000, 10000
        xSigma = 0.26 * scaling
        ySigma = 0.26 * scaling
        zSigma = 0.26 * scaling * 2  # larger vertical uncertainty
        # A diagonal noise model created by specifying a Vector of precisions,
        # i.e. the diagonal of the information matrix, i.e., weights
        # i.e. precision = 0 means infinite sigma
        #                                                                 Rx   Ry   Rz   Tx                 Ty                 Tz
        self.noise_model = gtsam.noiseModel.Diagonal.Precisions(np.array([0.0, 0.0, 0.0, 1.0 / xSigma ** 2, 1.0 / ySigma ** 2, 1.0 / zSigma ** 2]))

    def T_in_body(self):
        """GNSS position given in body frame"""
        #             x     y      z
        T = np.array([3.015, 0.0, -1.1])
        return T

    def add_measurement(self, pose_global, pose_key, measurement):
        # NOTE: GPS factor is defined in 6DOF, not 3DOF
        GPS_pose = gtsam.Pose3(pose_global.rotation(), measurement)
        self.factors.add(gtsam.PriorFactorPose3(pose_key, GPS_pose, self.noise))
