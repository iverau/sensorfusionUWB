from pathlib import Path
from time import time
import scipy.io
from datasetSettings import *
import numpy as np

"""
Hente ut x,y, theta for et hvist timestep
Kalkulere hastigheten i samme tidspunkt

"""


class GroundTruthEstimates:

    def __init__(self, dataset_id) -> None:
        self.files = scipy.io.loadmat(GroundTruthEstimates.generate_path(dataset_id))
        self.datasetSettings = GroundTruthEstimates.select_dataset(dataset_id)

        """
              dtype=[('tow', 'O'), ('navigaton_frame', 'O'), ('roll_hat', 'O'), ('pitch_hat', 'O'), ('yaw_hat', 'O'), 
              ('omega_ib_b_hat', 'O'), ('ars_bias_hat', 'O'), ('ars_bias_total_hat', 'O'), ('acc_bias_hat', 'O'), 
              ('gravity_hat', 'O'), ('tmo_innovation', 'O'), ('T_tmo_innovation', 'O'), ('T_tmo_innovation_sum', 'O'), 
              ('p_Lb_L_hat', 'O'), ('v_eb_n_hat', 'O'), ('speed_course_hat', 'O'), ('innov', 'O'), 
              ('innov_covariance', 'O'), ('P_hat', 'O')])}
        """

        self.extract_data()
        self.start_index = GroundTruthEstimates.find_index_closest(self.time, self.datasetSettings.bag_start_time_offset)


    def initial_pose(self):
        return np.array([self.north[self.start_index], self.east[self.start_index], self.heading[self.start_index]])

    def initial_velocity(self):
        # Initial velcoity is set to 0 as it moves close to a straight line
        return np.array([self.v_north[self.start_index], self.v_east[self.start_index], 0])

    def extract_data(self):
        self.mat_file_to_dict()
        self.time = np.array(self.data_dictionary["tow"][0])
        self.north = np.array(self.data_dictionary["p_lb_L_hat"][0])
        self.east = np.array(self.data_dictionary["p_lb_L_hat"][1])
        self.heading = np.array(self.data_dictionary["yaw_hat"][0])

        self.v_north = self.data_dictionary["v_eb_n_hat"][0]
        self.v_east = self.data_dictionary["v_eb_n_hat"][1]
        
        # Compensate for time offset
        self.time -= self.datasetSettings.gt_time_offset

    @staticmethod
    def find_index_closest(time_array, start_time):
        time_array -= time_array[0]
        return (np.abs(time_array - start_time)).argmin()

    def mat_file_to_dict(self):
        keys = ["tow", "navigation_frame", "roll_hat", "pitch_hat", "yaw_hat", "omega_ib_b_hat", "ars_bias_hat", "ars_bias_total_hat", "acc_bias_hat", "gravity_hat", "tmo_innovation", "T_tmo_innovation", "T_tmo_innovation_sum", "p_lb_L_hat", "v_eb_n_hat", "speed_course_hat", "innov", "innov_covariance", "P_hat"]
        self.data_dictionary = {}
        for element, key in zip(self.files["obsv_estimates"][0][0], keys):
            self.data_dictionary[key] = element

    @staticmethod
    def generate_path(dataset_id):
        return Path.joinpath(Path(__file__).parent.absolute(), "Gnssdata/obsv_estimates" + "4" + ".mat")


    @staticmethod
    def select_dataset(id : int):
        if id == 1:
            return DatasetSettings_Trondheim1()
        elif id == 3:
            return DatasetSettings_Trondheim3()
        else:
            return DatasetSettings_Trondheim4()

gt = GroundTruthEstimates(4)