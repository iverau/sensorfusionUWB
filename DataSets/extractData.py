import rosbag
from .datasetSettings import *
import rospy
from .dataFrame import DataFrame
from pathlib import Path

class ROSData:

    def __init__(self, dataset_number: int) -> None:
        print("Initialize ROS dataset number ",dataset_number,".\n",end="")
        
        # Initializes the dataset settings
        self.dataset_settings = ROSData.select_dataset(dataset_number)


        print("Test av property",self.dataset_settings.filepath)

        # Initializes the rosbag
        self.bag = rosbag.Bag(self.dataset_settings.filepath)
        self.bag_start_time = rospy.Time(self.bag.get_start_time() + self.dataset_settings.bag_start_time_offset)
        self.bag_end_time = self.get_bag_end_time()



    def generate_measurements(self):
        for topic, msg, t in self.bag.read_messages(topics=self.dataset_settings.enabled_topics):
            yield DataFrame(topic, msg, t)

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

        

rosdata = ROSData(1)
print(next(rosdata.generate_measurements()).msg)
