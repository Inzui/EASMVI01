from photoProcessor import PhotoProcessor
from dataSetService import DataSetService
from os.path import *

class Main():
    def __init__(self, photoProcessor: PhotoProcessor, dataSet: DataSetService) -> None:
        self.photoProcessor = photoProcessor
        self.dataSet = dataSet

    def run(self):
        self.dataSet.append(None, None)
        # for i in range(50):
        #     try:
        #         print(self.photoProcessor.run(f"C:\\Users\\ianzu\\OneDrive - Hogeschool Rotterdam\\Machine Vision\\Dataset\\Training\\A\\A.{i}.png"))
        #     except:
        #         print(f"Rejected image '{i}'")

if __name__ == "__main__":
    main = Main(PhotoProcessor(), DataSetService(f"{dirname(realpath(__file__))}\\Dataset", "Training.csv"))
    main.run()