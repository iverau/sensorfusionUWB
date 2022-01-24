from enum import Enum
import numpy as np

UWB_OFFSET = 0.85
UWB_STD = 0.2

class MeasurementType(Enum):
    GNSS = "GNSS"
    IMU = "IMU"
    UWB = "UWB"

class Measurement:

    def __init__(self, topic, msg, t) -> None:
        self.measurement_type = Measurement.select_measurement_type(topic)
        self.msg = msg
        self.create_measurement(msg)
        self.time = t

    @staticmethod
    def select_measurement_type(topic):
        if topic == "/os1_cloud_node/imu":
            return MeasurementType.IMU
        elif topic == "/uwb_beacons_parsed":
            return MeasurementType.UWB
        elif topic == "/ublox1/fix":
            return MeasurementType.GNSS
        else:
            raise NotImplementedError

    def create_measurement(self, msg):
        if self.measurement_type == MeasurementType.IMU:
            self.create_IMU_measurement(msg)
        elif self.measurement_type == MeasurementType.UWB:
            self.create_UWB_measurement(msg)
        else:
            raise NotImplementedError

    def create_IMU_measurement(self, msg):
        self.angular_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.angular_vel_covariance = np.diag([msg.angular_velocity_covariance[0], msg.angular_velocity_covariance[4], msg.angular_velocity_covariance[8]])
        self.linear_vel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.linear_vel_covariance = np.diag([msg.linear_acceleration_covariance[0], msg.linear_acceleration_covariance[4], msg.linear_acceleration_covariance[8]])

    def create_UWB_measurement(self, msg):
        self.range = msg.Dist - UWB_OFFSET
        self.std = UWB_STD
        self.id = msg.SRC


    def __repr__(self) -> str:
        if self.measurement_type == MeasurementType.UWB:
            return f"Measurement[Type={self.measurement_type.value}, Time={self.time}, Range={self.range}, Id={self.id}]" 

        return f"Measurement[Type={self.measurement_type.value}, Time={self.time}, Range={self.range}, Id={self.id}]" 