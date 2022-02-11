from pathlib import Path

class SensorConfigurations:
    UWB_IMU = ["/sentiboard/adis", "/uwb_beacons_parsed"]
    UWB_LIDAR_IMU = ["/os1_cloud_node/imu", "/uwb_beacons_parsed"]
    UWB = ["/uwb_beacons_parsed"]
    IMU_GNSS = ["/os1_cloud_node/imu", "/ublox1/fix"]
    GNSS = ["/ublox1/fix"]
    IMU_TRI = ["/sentiboard/adis"]
    IMU_LIDAR_TRI = ["/os1_cloud_node/imu"]


class DatasetSettingsBase:
    #bag_start_time_offset = 0
    bag_duration = 100
    enabled_topics = SensorConfigurations.UWB_IMU
    _filename = None

    def __init__(self, dataset_number: int):
        self.dataset_number = dataset_number

    #IMU settings
    imu_frequency = 100

    def ned_origin_filepath(self):
        return Path.joinpath(Path(__file__).parent.absolute(), "Gnssdata/ned_origin.mat")

    def trilateration_filepath(self):
        return Path.joinpath(Path(__file__).parent.absolute(), f"Trilateration/trondheim{self.dataset_number}/trilateration_3d.mat")
        
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
    gt_time_offset = 18.24 
    dataset_number = 1


    def __init__(self):
        super(DatasetSettings_Trondheim1, self).__init__(self.dataset_number)


class DatasetSettings_Trondheim3(DatasetSettingsBase):
    bag_start_time_offset = 840
    _filename = "trondheim3_inn.bag"
    gt_time_offset = 18.42
    dataset_number = 3

    def __init__(self):
        super(DatasetSettings_Trondheim3, self).__init__(self.dataset_number)


class DatasetSettings_Trondheim4(DatasetSettingsBase):
    bag_start_time_offset = 840

    _filename = "trondheim4_inn.bag"
    gt_time_offset = 18.55
    dataset_number = 4


    def __init__(self):
        super(DatasetSettings_Trondheim4, self).__init__(self.dataset_number)
