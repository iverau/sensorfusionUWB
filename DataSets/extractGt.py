from pathlib import Path
from time import time
import scipy.io
from .datasetSettings import *
import numpy as np

"""
Hente ut x,y, theta for et hvist timestep
Kalkulere hastigheten i samme tidspunkt

"""


class GroundTruthEstimates:

    def __init__(self, dataset_id, pre_initialization=None) -> None:
        self.files = scipy.io.loadmat(
            GroundTruthEstimates.generate_path(dataset_id))
        self.datasetSettings = GroundTruthEstimates.select_dataset(dataset_id)

        """
              dtype=[('tow', 'O'), ('navigaton_frame', 'O'), ('roll_hat', 'O'), ('pitch_hat', 'O'), ('yaw_hat', 'O'), 
              ('omega_ib_b_hat', 'O'), ('ars_bias_hat', 'O'), ('ars_bias_total_hat', 'O'), ('acc_bias_hat', 'O'), 
              ('gravity_hat', 'O'), ('tmo_innovation', 'O'), ('T_tmo_innovation', 'O'), ('T_tmo_innovation_sum', 'O'), 
              ('p_Lb_L_hat', 'O'), ('v_eb_n_hat', 'O'), ('speed_course_hat', 'O'), ('innov', 'O'), 
              ('innov_covariance', 'O'), ('P_hat', 'O')])}
        """

        self.extract_data(pre_initialization=pre_initialization)

    def initial_pose(self):
        print("Initial pose:", self.east[0], self.north[0], self.down[0])
        return np.array([self.roll[0], self.pitch[0], self.yaw[0], self.east[0], self.north[0], self.down[0]])

    def initial_velocity(self):
        # Initial velcoity is set to 0 as it moves close to a straight line
        return np.array([self.v_north[0], self.v_east[0], self.v_down[0]])

    def extract_data(self, pre_initialization=None):
        self.mat_file_to_dict()
        self.time = np.array(self.data_dictionary["tow"][0])

        # Compensate for time offset
        if pre_initialization:
            self.time_offset = self.datasetSettings.gt_time_offset - 10
        else:
            self.time_offset = self.datasetSettings.gt_time_offset

        self.start_index = self.find_index_closest(self.time, self.datasetSettings.bag_start_time_offset)
        self.time = self.time[self.start_index:]
        print("Start time of ground truth:", self.time[0])

        self.north = np.array(self.data_dictionary["p_lb_L_hat"][0])[
            self.start_index:]
        self.east = np.array(self.data_dictionary["p_lb_L_hat"][1])[
            self.start_index:]
        self.down = np.array(self.data_dictionary["p_lb_L_hat"][2])[
            self.start_index:]
        self.roll = np.array(self.data_dictionary["roll_hat"][0])[
            self.start_index:]
        self.pitch = np.array(self.data_dictionary["pitch_hat"][0])[
            self.start_index:]
        self.yaw = np.array(self.data_dictionary["yaw_hat"][0])[
            self.start_index:]

        self.v_north = self.data_dictionary["v_eb_n_hat"][0][self.start_index:]
        self.v_east = self.data_dictionary["v_eb_n_hat"][1][self.start_index:]
        self.v_down = self.data_dictionary["v_eb_n_hat"][2][self.start_index:]

        self.gt_transelation = np.array(self.data_dictionary["p_lb_L_hat"]).astype(
            "float")[:, self.start_index:]
        self.gt_angels = np.zeros((len(self.time), 3)).astype("float")
        self.gt_angels[:, 0] = self.roll.copy()
        self.gt_angels[:, 1] = self.pitch.copy()
        self.gt_angels[:, 2] = self.yaw.copy()

    def find_index_closest(self, time_array, start_time):
        temp_array = time_array - (time_array[0] - self.time_offset)
        return (np.abs(temp_array - start_time)).argmin()

    def mat_file_to_dict(self):
        keys = ["tow", "navigation_frame", "roll_hat", "pitch_hat", "yaw_hat", "omega_ib_b_hat", "ars_bias_hat", "ars_bias_total_hat", "acc_bias_hat", "gravity_hat",
                "tmo_innovation", "T_tmo_innovation", "T_tmo_innovation_sum", "p_lb_L_hat", "v_eb_n_hat", "speed_course_hat", "innov", "innov_covariance", "P_hat"]
        self.data_dictionary = {}
        for element, key in zip(self.files["obsv_estimates"][0][0], keys):
            self.data_dictionary[key] = element

    @staticmethod
    def generate_path(dataset_id):
        return Path.joinpath(Path(__file__).parent.absolute(), "Gnssdata/obsv_estimates" + "4" + ".mat")

    @staticmethod
    def select_dataset(id: int):
        if id == 1:
            return DatasetSettings_Trondheim1()
        elif id == 3:
            return DatasetSettings_Trondheim3()
        else:
            return DatasetSettings_Trondheim4()
