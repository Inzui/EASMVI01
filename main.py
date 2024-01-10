from photoProcessor import PhotoProcessor
from dataSetService import DataSetService
from classifier import Classifier
from datetime import datetime
import os, cv2, argparse

FRAMES_PER_SECOND = 2

class Main():
    def __init__(self, identifiersDir: str) -> None:
        self.identifiersDir = identifiersDir

        self.photoProcessor = PhotoProcessor()
        self.trainingDataSet = DataSetService(identifiersDir, "Training")
        self.testDataSet = DataSetService(identifiersDir, "Validation")
        self.classifier: Classifier = None

    def run(self, forceConvert: bool = False, forceTrain: bool = False, showImages: bool = False):
        # Convert pictures to CSV
        if (forceConvert or not self.trainingDataSet.exists()):
            print("Training CSV does not exist.")
            self.__picturesToDataSet(self.trainingDataSet)
        if (forceConvert or not self.testDataSet.exists()):
            print("Validation CSV does not exist.")
            self.__picturesToDataSet(self.testDataSet)

        self.classifier = Classifier()
        if (forceTrain or not os.path.isfile(self.classifier.filename)):
            self.classifier.train(self.trainingDataSet.load(), self.testDataSet.load())
        
        # Get pictures from webcam and use as input.
        print("Loading camera")
        cam = cv2.VideoCapture(0)
        lastCaptureTime = datetime.now()
        prediction = ("?", "")

        while True:
            _, frame = cam.read()
            frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
            cv2.putText(frame, f"Prediction: {prediction[0]}        Confindence: {prediction[1]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Input', frame)

            deltaS = (datetime.now() - lastCaptureTime).total_seconds()
            if (deltaS > 1/FRAMES_PER_SECOND):
                try:
                    coordinates = DataSetService.unpack(self.photoProcessor.run(frame, showImages))
                    print(coordinates)
                    prediction = self.classifier.run(coordinates)
                    print(prediction[0])

                except Exception as e:
                    prediction = ("?", "")
                    print(e)
                finally:
                    lastCaptureTime = datetime.now()

            c = cv2.waitKey(1)
            if c == 27:
                break
    
    def __picturesToDataSet(self, dataSet: DataSetService):
        print(f"Converting pictures from '{dataSet.dataSetType}' to CSV.")
        dataSetDir = os.path.join(self.identifiersDir, dataSet.dataSetType)
        dataSet.clear()

        identifiers = [identifier for identifier in os.listdir(dataSetDir) if os.path.isdir(os.path.join(dataSetDir, identifier))]
        for identifier in identifiers:
            print(f"Converting '{identifier}'")
            pictures = os.listdir(os.path.join(dataSetDir, identifier))
            for pictureName in pictures:
                try:
                    dataSet.append(identifier, self.photoProcessor.run(os.path.join(dataSetDir, identifier, pictureName)))
                except Exception as e:
                    print(f"Removing: '{pictureName}', reason: '{e}'")
                    os.remove(os.path.join(dataSetDir, identifier, pictureName))
            dataSet.renumber(identifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "EASMVI01 Assignment for recognizing Dutch sign language.")
    parser.add_argument("-fc", "--forceConvert", action = "store_true", help = "Force the conversion of training and test images to CSV.")
    parser.add_argument("-ft", "--forceTrain", action = "store_true", help = "Force the training of the Machine Learning Model, even if one already exists.")
    parser.add_argument("-si", "--showImages", action = "store_true", help = "Shows the detected hand with drawn landmarks while running.")
    args = parser.parse_args()

    main = Main(f"{os.path.dirname(os.path.realpath(__file__))}\\DataSet",)
    main.run(args.forceConvert, args.forceTrain, args.showImages)