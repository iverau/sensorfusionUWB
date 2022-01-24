import gtsam
from DataSets.extractData import ROSData


class GtSAMTest:
    DATASET_NUMBER = 4

    def __init__(self) -> None:
        self.dataset = ROSData(self.DATASET_NUMBER)

    
    def run(self):
        for dataFrame in self.dataset.generate_measurements():
            print(dataFrame)

testing = GtSAMTest()
testing.run()