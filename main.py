from photoProcessor import PhotoProcessor
from dataSetService import DataSetService
from classifier import Classifier
import os

class Main():
    def __init__(self, identifiersDir: str) -> None:
        self.identifiersDir = identifiersDir

        self.photoProcessor = PhotoProcessor()
        self.trainingDataSet = DataSetService(identifiersDir, "Training")
        self.testDataSet = DataSetService(identifiersDir, "Validation")
        self.classifier: Classifier = None

    def run(self, forceConvert: bool = False):
        # Convert pictures to CSV
        if (forceConvert or not self.trainingDataSet.exists()):
            print("Training CSV does not exist.")
            self._picturesToDataSet(self.trainingDataSet)
        if (forceConvert or not self.testDataSet.exists()):
            print("Test CSV does not exist.")
            self._picturesToDataSet(self.testDataSet)
    
    def _picturesToDataSet(self, dataSet: DataSetService):
        print(f"Converting pictures from '{dataSet.dataSetType}' to CSV.")
        dataSetDir = os.path.join(self.identifiersDir, dataSet.dataSetType)
        dataSet.clear()

        identifiers = [identifier for identifier in os.listdir(dataSetDir) if os.path.isdir(os.path.join(dataSetDir, identifier))]
        for identifier in identifiers:
            pictures = os.listdir(os.path.join(dataSetDir, identifier))
            for pictureName in pictures:
                try:
                    dataSet.append(identifier, self.photoProcessor.run(os.path.join(dataSetDir, identifier, pictureName)))
                except Exception as e:
                    print(f"Rejected: '{pictureName}', reason: '{e}'")

if __name__ == "__main__":
    main = Main(f"{os.path.dirname(os.path.realpath(__file__))}\\DataSet")
    main.run()