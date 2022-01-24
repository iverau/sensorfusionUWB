from enum import Enum


class MeasurementType(Enum):
    GNSS = "GNSS"
    IMU = "IMU"
    UWB = "UWB"

class Measurement:

    def __init__(self, topic, msg, t) -> None:
        self.measurement_type = Measurement.select_measurement_type(topic)
        #self.topic = topic
        self.msg = msg
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

    def __repr__(self) -> str:
        return f"Measurement[Type={self.measurement_type.value}, Time={self.time}]" 