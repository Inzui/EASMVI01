from photoProcessor import PhotoProcessor
from dataSetService import DataSetService
from os.path import *

class Main():
    def __init__(self, dataSetDir: str) -> None:
        self.dataSetDir = dataSetDir
        self.photoProcessor = PhotoProcessor()
        self.dataSet = DataSetService(dataSetDir, "Training.csv")

    def run(self):
        pass
    
    def picturesToDataSet(self):
        self.dataSet.append('A', self.photoProcessor.run(f"{self.dataSetDir}\\Training\\A\\A.1.png"))

if __name__ == "__main__":
    main = Main(f"{dirname(realpath(__file__))}\\Dataset")
    main.run()
    main.picturesToDataSet()