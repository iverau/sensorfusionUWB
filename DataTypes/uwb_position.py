import numpy as np
import scipy.io

class UWB_Ancors_Descriptor:

    AMOUNT_OF_BEACONS = 5

    def __init__(self, dataset_number) -> None:
        self.UWB_position_map = {}
        self.dataset_number = dataset_number
        self.create_UWB_mapping()

    def create_UWB_mapping(self):
        for beacon_id in range(1, self.AMOUNT_OF_BEACONS + 1):
            mat_file = scipy.io.loadmat(self.generate_beacon_data_path(beacon_id))
            uwb_pos = UWB_Position(mat_file)
            self.UWB_position_map[uwb_pos.id] = uwb_pos

    def __getitem__(self, key):
        return self.UWB_position_map[key]

    def generate_beacon_data_path(self,id):
        return f"DataSets/Beacondata/trondheim{self.dataset_number}/beacon{id}_time_fix.mat"

    def __repr__(self) -> str:
        return f"UWB_Ancors_Descriptor[Dataset_number = {self.dataset_number}, Amount_of_beacons = {self.AMOUNT_OF_BEACONS}]"

class UWB_Position:
    def __init__(self, mat_file) -> None:
        self.covariance = np.diag([0.1, 0.1, 0.1])
        self.extract_coordinates(mat_file)

    def extract_coordinates(self, mat_file):
        self.id =   mat_file["beacon"][0][0][0][0][0]
        self.x =    mat_file["beacon"][0][0][1][0][0]
        self.y =    mat_file["beacon"][0][0][2][0][0]
        self.z =    mat_file["beacon"][0][0][3][0][0]

    def position(self):
        return [self.x, self.y, self.z]

    def __repr__(self) -> str:
        return f"UWB_Position[id = {self.id}, x = {self.x}, y = {self.y}, z = {self.z}]"
