import rosbag
from .datasetSettings import *
import rospy
from DataTypes.measurement import generate_measurement
import scipy.io
import pymap3d as pm
import numpy as np


class ROSData:

    def __init__(self, dataset_number: int) -> None:
        print("Initialize ROS dataset number ", dataset_number, ".\n", end="")

        # Initializes the dataset settings
        self.dataset_settings = ROSData.select_dataset(dataset_number)

        self.initialization_step_time = 0

        # Initializes the rosbag
        self.bag = rosbag.Bag(self.dataset_settings.filepath)
        self.bag_start_time = rospy.Time(self.bag.get_start_time(
        ) + self.dataset_settings.bag_start_time_offset + self.initialization_step_time)
        self.bag_end_time = self.get_bag_end_time()
        self.extract_initial_pose()

    def extract_initial_pose(self):
        for _, msg, t in self.bag.read_messages(topics=["/ublox2/fix"], start_time=self.bag_start_time):
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
        #print("NED Origin",np.array([n, e, d]))
        return np.array([n, e, d])

    def generate_initialization_gnss_imu(self):
        start_time = rospy.Time(self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset - 10)
        end_time = rospy.Time(self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset + self.initialization_step_time)
        topics = ["/sentiboard/adis", "/ublox2/fix", "/camera/image_raw/compressed"]
        for topic, msg, t in self.bag.read_messages(topics=topics, start_time=start_time, end_time=end_time):
            yield generate_measurement(topic, msg, t)

    def generate_measurements(self):
        for topic, msg, t in self.bag.read_messages(topics=self.dataset_settings.enabled_topics, start_time=self.bag_start_time, end_time=self.bag_end_time):
            yield generate_measurement(topic, msg, t)

    def get_bag_end_time(self):
        if self.dataset_settings.bag_duration < 0:
            return rospy.Time(self.bag.get_end_time())
        return rospy.Time(self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset + self.dataset_settings.bag_duration)

    @staticmethod
    def select_dataset(id: int):
        if id == 1:
            return DatasetSettings_Trondheim1()
        elif id == 3:
            return DatasetSettings_Trondheim3()
        else:
            return DatasetSettings_Trondheim4()


class RosDataTrilateration:

    # TODO: Sørg for at UWB starter på rett sted

    def __init__(self, dataset_number: int) -> None:
        print("Initialize ROS dataset number ", dataset_number, ".\n", end="")

        # Initializes the dataset settings
        self.dataset_settings = RosDataTrilateration.select_dataset(
            dataset_number)

        # Initializes the rosbag
        self.bag = rosbag.Bag(self.dataset_settings.filepath)
        self.bag_start_time = rospy.Time(
            self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset)
        self.bag_end_time = self.get_bag_end_time()
        self.extract_initial_pose()
        print("Starttime", self.bag_start_time)

    def extract_initial_pose(self):
        for _, msg, t in self.bag.read_messages(topics=["/ublox2/fix"], start_time=self.bag_start_time):
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

    def get_bag_end_time(self):
        if self.dataset_settings.bag_duration < 0:
            return rospy.Time(self.bag.get_end_time())
        return rospy.Time(self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset + self.dataset_settings.bag_duration)

    def generate_trilateration_combo_measurements(self):
        """
            Sjekke hvem som har størst tidssteg
            Returnere den som har minst
        """
        tri_generator = self.skip_to_right_time_step(
            self.generate_trilateration_measurement())
        imu_generator = self.generate_measurements()

        tri_meas = next(tri_generator)
        imu_meas = next(imu_generator)
        while True:
            if tri_meas.time < imu_meas.time.to_time():
                yield tri_meas
                tri_meas = next(tri_generator)
            else:
                yield imu_meas
                imu_meas = next(imu_generator)

            if not tri_meas and not imu_meas:
                return None

    def skip_to_right_time_step(self, trilateration_generator):
        tri_time = next(trilateration_generator)
        while tri_time.time < self.bag_start_time.to_time():
            tri_time = next(trilateration_generator)
        return trilateration_generator

    def generate_measurements(self):
        for topic, msg, t in self.bag.read_messages(topics=self.dataset_settings.enabled_topics, start_time=self.bag_start_time, end_time=self.bag_end_time):
            yield generate_measurement(topic, msg, t)

    def extract_trilateration_measurements(self):
        trilateration_data = scipy.io.loadmat(
            self.dataset_settings.trilateration_filepath())
        x_list = trilateration_data["pos_sensor"][0][0][0][0]
        y_list = trilateration_data["pos_sensor"][0][0][1][0]
        z_list = trilateration_data["pos_sensor"][0][0][4][0]
        time_list = trilateration_data["pos_sensor"][0][0][6][0]
        list_of_measurements = []

        for x, y, z, time in zip(x_list, y_list, z_list, time_list):
            list_of_measurements.append(generate_measurement(
                "uwb_trilateration", {"x": x, "y": y, "z": z}, time))

        return list_of_measurements

    def generate_trilateration_measurement(self):
        measurements = self.extract_trilateration_measurements()
        for measurement in measurements:
            yield measurement

    @staticmethod
    def select_dataset(id: int):
        if id == 1:
            return DatasetSettings_Trondheim1()
        elif id == 3:
            return DatasetSettings_Trondheim3()
        else:
            return DatasetSettings_Trondheim4()
