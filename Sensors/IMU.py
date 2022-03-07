import gtsam
import numpy as np


# NOTE: This contains IMU preintegration factor implementation
# NOTE: IMU frame should be given in body frame (x-forward, 
# y-right, z-downward)


class IMU:
    def __init__(self):
        self.setup_imu_params()


    def setup_imu_params(self):
        """Setup IMU preintegration and bias parameters"""
        AccSigma        = 0.1
        GyroSigma       = 0.00175
        IntSigma        = 0.000167  # integtation sigma
        AccBiasSigma    = 2.91e-006
        GyroBiasSigma   = 0.0100395199348279
        self.preintegration_param = gtsam.PreintegrationParams(np.array([0, 0, 9.82175]))
        self.preintegration_param .setAccelerometerCovariance(AccSigma ** 2 * np.eye(3))
        self.preintegration_param .setGyroscopeCovariance(GyroSigma ** 2 * np.eye(3))
        self.preintegration_param .setIntegrationCovariance(IntSigma ** 2 * np.eye(3))
        self.preintegration_param .setOmegaCoriolis(np.array([0, 0, 0]))  # account for earth's rotation
        self.sigmaBetweenBias = np.array([AccBiasSigma, AccBiasSigma, AccBiasSigma, GyroBiasSigma, GyroBiasSigma, GyroBiasSigma])

    def R_in_body(self):
        """IMU to body frame rotation"""
        R = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0,-1],
            ]
        )
        return R

