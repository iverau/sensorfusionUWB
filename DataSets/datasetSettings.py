from pathlib import Path

class SensorConfigurations:
    UWB_IMU = ["/os1_cloud_node/imu", "/uwb_beacons_parsed"]
    UWB = ["/uwb_beacons_parsed"]
    IMU_GNSS = ["/os1_cloud_node/imu", "/ublox1/fix"]
    GNSS = ["/ublox1/fix"]


class DatasetSettingsBase:
    #bag_start_time_offset = 0
    bag_duration = 100
    enabled_topics = SensorConfigurations.UWB_IMU
    _filename = None

    #IMU settings
    imu_frequency = 100

        
    @property
    def filepath(self):
        return Path.joinpath(Path(__file__).parent.absolute(), self._filename)

    @filepath.setter
    def filepath(self, value):
        self._filepath = value

    def __repr__(self) -> str:
        return f"DatasetSettings[filename={self._filename}, start_time={self.bag_start_time_offset}, duration={self.bag_duration}]"


class DatasetSettings_Trondheim1(DatasetSettingsBase):
    bag_start_time_offset = 930
    _filename = "trondheim1_inn.bag"


class DatasetSettings_Trondheim3(DatasetSettingsBase):
    bag_start_time_offset = 840

    _filename = "trondheim3_inn.bag"


class DatasetSettings_Trondheim4(DatasetSettingsBase):
    bag_start_time_offset = 840

    _filename = "trondheim4_inn.bag"
