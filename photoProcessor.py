import cv2, numpy
import mediapipe as mp

class PhotoProcessor:
    def __init__(self, cropMargin = 50) -> None:
        self.cropMargin = cropMargin
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def run(self, arg1, showImage: bool = False):
        if (isinstance(arg1, str)):
            image = cv2.imread(arg1)
            return self._run(image, showImage)
        else:
            return self._run(arg1, showImage)

    def _run(self, image: numpy.ndarray, showImage: bool):
        processedImage = self._processImage(image)
        imageBorders = self._getImageBorders(image, processedImage)
        croppedImage = image[imageBorders[0]:imageBorders[1], imageBorders[2]:imageBorders[3]]

        if (showImage):
            self._drawHandConnections(image, processedImage)
            cv2.imshow("Test", croppedImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        jointCoordinates = self._getJointCoordinates(croppedImage, processedImage)
        if (len(jointCoordinates) != 21):
            raise Exception("Too few/many joints")

        return jointCoordinates

    def _processImage(self, img):
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.hands.process(grayImage)
    
    def _getImageBorders(self, img, processedImage) -> ():
        jointCoordinates = self._getJointCoordinates(img, processedImage)
        xCoordinates = [x[0] for x in jointCoordinates]
        yCoordinates = [y[1] for y in jointCoordinates]

        return (min(yCoordinates) - self.cropMargin, max(yCoordinates) + self.cropMargin, min(xCoordinates) - self.cropMargin, max(xCoordinates) + self.cropMargin)
    
    def _getJointCoordinates(self, img, processedImage) -> []:
        if (processedImage.multi_hand_landmarks == None):
            raise Exception("Could not identify a hand")  
         
        coordinates = []
        if processedImage.multi_hand_landmarks:
            for handLms in processedImage.multi_hand_landmarks:
                for _, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    coordinates.append((cx, cy))
            return coordinates

    def _drawHandConnections(self, img, results):
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape

                    # Finding the coordinates of each landmark
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Creating a circle around each landmark
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                            cv2.FILLED)
                    # Drawing the landmark connections
                    self.mpDraw.draw_landmarks(img, handLms,
                                        self.mpHands.HAND_CONNECTIONS)
            return img