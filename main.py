from photoProcessor import PhotoProcessor
from dataSetService import DataSetService
import os

class Main():
    def __init__(self, identifiersDir: str, dataSetType: str) -> None:
        self.identifiersDir = identifiersDir
        self.dataSetType = dataSetType

        self.photoProcessor = PhotoProcessor()
        self.dataSet = DataSetService(identifiersDir, f"{self.dataSetType}.csv")

    def run(self):
        df = self.dataSet.load()
        print(df)
    
    def picturesToDataSet(self):
        dataSetDir = os.path.join(self.identifiersDir, self.dataSetType)
        self.dataSet.clear()

        identifiers = os.listdir(dataSetDir)
        for identifier in identifiers:
            pictures = os.listdir(os.path.join(dataSetDir, identifier))
            for pictureName in pictures:
                try:
                    self.dataSet.append(identifier, self.photoProcessor.run(os.path.join(dataSetDir, identifier, pictureName)))
                except Exception as e:
                    print(f"Rejected: '{pictureName}', reason: '{e}'")

if __name__ == "__main__":
    main = Main(f"{os.path.dirname(os.path.realpath(__file__))}\\Dataset", "Training")
    # main.picturesToDataSet()
    main.run()