import rosbag
from .datasetSettings import *
import rospy
from DataTypes.measurement import generate_measurement
import scipy.io
import pymap3d as pm
import numpy as np
class ROSData:

    def __init__(self, dataset_number: int) -> None:
        print("Initialize ROS dataset number ",dataset_number,".\n",end="")
        
        # Initializes the dataset settings
        self.dataset_settings = ROSData.select_dataset(dataset_number)

        # Initializes the rosbag
        self.bag = rosbag.Bag(self.dataset_settings.filepath)
        self.bag_start_time = rospy.Time(self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset)
        self.bag_end_time = self.get_bag_end_time()
        self.extract_initial_pose()


    def extract_initial_pose(self):
        for _, msg, t in self.bag.read_messages(topics=["/ublox1/fix"], start_time=self.bag_start_time):
            data = msg
            time = t
            break
        self.bag_start_time = time
        return self.convert_GNSS_to_NED(data)

    def extract_ned_origin(self):
        data = scipy.io.loadmat(self.dataset_settings.ned_origin_filepath())
        latitude = data["lat0"][0][0]
        longitude = data["lon0"][0][0]
        altitiude = data["height0"][0][0]
        return np.array([latitude, longitude, altitiude])

    def convert_GNSS_to_NED(self, msg):
        ned_origin = self.extract_ned_origin()
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
        return np.array([n, e, d])

    def generate_measurements(self):
        for topic, msg, t in self.bag.read_messages(topics=self.dataset_settings.enabled_topics, start_time=self.bag_start_time, end_time=self.bag_end_time):
            yield generate_measurement(topic, msg, t)

    def get_bag_end_time(self):
        if self.dataset_settings.bag_duration < 0:
            return rospy.Time(self.bag.get_end_time())
        return rospy.Time(self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset + self.dataset_settings.bag_duration)

    @staticmethod
    def select_dataset(id : int):
        if id == 1:
            return DatasetSettings_Trondheim1()
        elif id == 3:
            return DatasetSettings_Trondheim3()
        else:
            return DatasetSettings_Trondheim4()


class RosDataTrilateration:


    def __init__(self, dataset_number: int) -> None:
        print("Initialize ROS dataset number ",dataset_number,".\n",end="")
        
        # Initializes the dataset settings
        self.dataset_settings = RosDataTrilateration.select_dataset(dataset_number)

        # Initializes the rosbag
        self.bag = rosbag.Bag(self.dataset_settings.filepath)
        self.bag_start_time = rospy.Time(self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset)
        self.bag_end_time = self.get_bag_end_time()
        self.extract_initial_pose()


    def extract_initial_pose(self):
        for _, msg, t in self.bag.read_messages(topics=["/ublox1/fix"], start_time=self.bag_start_time):
            data = msg
            time = t
            break
        self.bag_start_time = time
        return self.convert_GNSS_to_NED(data)

    def extract_ned_origin(self):
        data = scipy.io.loadmat(self.dataset_settings.ned_origin_filepath())
        latitude = data["lat0"][0][0]
        longitude = data["lon0"][0][0]
        altitiude = data["height0"][0][0]
        return np.array([latitude, longitude, altitiude])
    
    def get_bag_end_time(self):
        if self.dataset_settings.bag_duration < 0:
            return rospy.Time(self.bag.get_end_time())
        return rospy.Time(self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset + self.dataset_settings.bag_duration)

    def generate_measurements(self):

        """
            Sjekke hvem som har størst tidssteg
            Returnere den som har minst
        """
        for topic, msg, t in self.bag.read_messages(topics=self.dataset_settings.enabled_topics, start_time=self.bag_start_time, end_time=self.bag_end_time):
            yield generate_measurement(topic, msg, t)


    def generate_rosbag_measurements(self):
        for topic, msg, t in self.bag.read_messages(topics=self.dataset_settings.enabled_topics, start_time=self.bag_start_time, end_time=self.bag_end_time):
            yield generate_measurement(topic, msg, t)

    def generate_trilateration_measurement(self):
        trilateration_data = scipy.io.loadmat(self.dataset_settings.trilateration_filepath())
        print(trilateration_data)
    
    @staticmethod
    def select_dataset(id : int):
        if id == 1:
            return DatasetSettings_Trondheim1()
        elif id == 3:
            return DatasetSettings_Trondheim3()
        else:
            return DatasetSettings_Trondheim4()


datasets = RosDataTrilateration(4)