from enum import Enum
import numpy as np
import gtsam
import pymap3d as pm


UWB_OFFSET = 0.85
UWB_STD = 0.2

class MeasurementType(Enum):
    GNSS = "GNSS"
    IMU = "IMU"
    UWB = "UWB"
    UWB_TRI="UWB_Tri"

class Measurement:

    def __init__(self, topic, t) -> None:
        self.measurement_type = Measurement.select_measurement_type(topic)
        self.time = t

    @staticmethod
    def select_measurement_type(topic):
        if topic == "/os1_cloud_node/imu":
            return MeasurementType.IMU
        elif topic == "/uwb_beacons_parsed":
            return MeasurementType.UWB
        elif topic == "/ublox1/fix":
            return MeasurementType.GNSS
        elif topic == "uwb_trilateration":
            return MeasurementType.UWB_TRI
        elif topic == "/sentiboard/adis":
            return MeasurementType.IMU
        else:
            raise NotImplementedError



    def __repr__(self) -> str:
        if self.measurement_type == MeasurementType.UWB:
            return f"Measurement[Type={self.measurement_type.value}, Time={self.time}, Range={self.range}, Id={self.id}]" 

        return f"Measurement[Type={self.measurement_type.value}, Time={self.time}, Range={self.range}, Id={self.id}]" 


class UWB_Measurement(Measurement):


    def __init__(self, topic, msg, t) -> None:
        super().__init__(topic, t)
        self.extract_measurement_data(msg)

    def extract_measurement_data(self, msg):
        self.range = msg.Dist - UWB_OFFSET
        self.std = UWB_STD
        self.id = msg.SRC

    def __repr__(self) -> str:
        return f"Measurement[Type={self.measurement_type.value}, Time={self.time}, Range={self.range}, Id={self.id}]" 

class IMU_Measurement(Measurement):

    R_IMU_BODY = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ]
    )

    def __init__(self, topic, msg, t) -> None:
        super().__init__(topic, t)
        self.extract_measurement_data(msg)
        


    def imu_to_body(self, data):
        return self.R_IMU_BODY @ data

    def extract_measurement_data(self, msg):
        # Data converted to body
        self.angular_vel = np.array([msg.angular_velocity.y, msg.angular_velocity.x, -msg.angular_velocity.z])
        self.angular_vel_covariance = np.diag([msg.angular_velocity_covariance[4], msg.angular_velocity_covariance[0], msg.angular_velocity_covariance[8]])
        self.linear_vel = np.array([msg.linear_acceleration.y, msg.linear_acceleration.x, -msg.linear_acceleration.z])
        self.linear_vel_covariance = np.diag([msg.linear_acceleration_covariance[4], msg.linear_acceleration_covariance[0], msg.linear_acceleration_covariance[8]])
        self.linear_vel_covariance = np.diag([0.01 for i in range(3)])
        self.angular_vel_covariance = np.diag([0.0000175 for i in range(3)])


    def variance_vector(self):
        return np.array([self.angular_vel_covariance[0,0], self.angular_vel_covariance[1,1], self.angular_vel_covariance[2,2], self.linear_vel_covariance[0,0], self.linear_vel_covariance[1,1], self.linear_vel_covariance[2,2]])

    def __repr__(self) -> str:
        return f"Measurement[Type={self.measurement_type.value}, Time={self.time}, Angular_vel={self.angular_vel}, Linear_vel={self.linear_vel}]" 

class UWB_Trilateration_Measurement(Measurement):

    def __init__(self, topic, msg, t) -> None:
        super().__init__(topic, t)
        self.extract_measurement(msg)

    def extract_measurement(self, msg):
        self.x = msg["x"]
        self.y = msg["y"]
        self.z = msg["z"]
        self.covX = 1
        self.covY = 1
        self.covZ = 9

        self.position = [self.x, self.y, self.z]
        self.covariance = np.diag([self.covX, self.covY, self.covZ])
        self.noise_model = gtsam.noiseModel.Diagonal.Precisions(np.array([0.0, 0.0, 0.0, 1.0 / self.covX ** 2, 1.0 / self.covY ** 2, 1.0 / self.covZ ** 2]))

    def __repr__(self) -> str:
        return f"Measurement[Type={self.measurement_type.value}, Time={self.time}, X={self.x}, Y={self.y}, Z={self.z}]" 


class GNSS_Measurement(Measurement):

    def __init__(self, topic, msg, t) -> None:
        super().__init__(topic, t)
        self.extract_measurement(msg)

    def convert_GNSS_to_NED(self, msg):
        ned_origin = [63.43888731, 10.39601287, 41.59585029]
        n, e, d = pm.geodetic2ned(
                msg.latitude,
                msg.longitude,
                msg.altitude,
                ned_origin[0],  # NED origin
                ned_origin[1],  # NED origin
                ned_origin[2],  # NED origin
                ell=pm.Ellipsoid("wgs84"),
                deg=True,
        )
        #print("NED Origin",np.array([n, e, d]))
        return np.array([n, e, d])


    def extract_measurement(self, msg):
        # TODO: Finne ut av rekkefølgen på ting her :)
        ned_data = self.convert_GNSS_to_NED(msg)
        self.north = ned_data[0]
        self.east = ned_data[1]
        self.down = ned_data[2]
        self.covX = 1
        self.covY = 1
        self.covZ = 9

        self.position = [self.north, self.east, self.down]
        self.covariance = np.diag([self.covX, self.covY, self.covZ])
        self.noise_model = gtsam.noiseModel.Diagonal.Precisions(np.array([0.0, 0.0, 0.0, 1e-8, 1e-8, 1e-8]))

    def __repr__(self) -> str:
        return f"Measurement[Type={self.measurement_type.value}, Time={self.time}, X={self.x}, Y={self.y}, Z={self.z}]" 


def generate_measurement(topic, msg, t):
    measurement_type =  Measurement.select_measurement_type(topic)
    if measurement_type == MeasurementType.UWB:
        return UWB_Measurement(topic, msg, t)
    elif measurement_type == MeasurementType.IMU:
        return IMU_Measurement(topic, msg, t)
    elif measurement_type == MeasurementType.UWB_TRI:
        return UWB_Trilateration_Measurement(topic, msg, t)
    elif measurement_type == MeasurementType.GNSS:
        return GNSS_Measurement(topic, msg, t)
    else:
        raise NotImplementedError