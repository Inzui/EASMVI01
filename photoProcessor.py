import cv2, numpy, math
import mediapipe as mp

class PhotoProcessor:
    def __init__(self, cropMargin: int = 25) -> None:
        self.cropMargin = cropMargin
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

    def run(self, arg1, showImage: bool = False):
        if (isinstance(arg1, str)):
            image = cv2.imread(arg1)
            return self._run(image, showImage)
        else:
            return self._run(arg1, showImage)

    def _run(self, image: numpy.ndarray, showImage: bool):
        # Find the hand in the image.
        results = self._detectHands(image)
        handCoordinates = self._getJointCoordinates(image, results)

        # Rotate the hand so it is always oriented upwards.
        image = self._rotateImage(image, self._getHandOrientation(handCoordinates))
        results = self._detectHands(image)

        # Crop the image so only the hand remains in the image.
        image = self._cropImage(image, self._getJointCoordinates(image, results))
        results = self._detectHands(image)
        
        if (showImage):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self._drawHandConnections(image, results)
            cv2.imshow("Hand Connections", image)
        
        # Return the joint coordinates.
        return self._getJointCoordinates(image, results)

    def _cropImage(self, image: numpy.ndarray, jointCoordinates: [()], desiredHeight: int = 500):
        imageBorders = self._getImageBorders(jointCoordinates)
        image = image[imageBorders[0]:imageBorders[1], imageBorders[2]:imageBorders[3]]
        image = numpy.ascontiguousarray(image)
        imageHeight, imageWidth, _ = image.shape
        aspectRatio = float(imageWidth) / float(imageHeight)
        desiredWidth = int(desiredHeight * aspectRatio)
        image = cv2.resize(image, (desiredWidth, desiredHeight), interpolation = cv2.INTER_AREA)
        return image

    def _rotateImage(self, image: numpy.ndarray, rotationDegrees: float):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        rotationMatrix = cv2.getRotationMatrix2D(center, rotationDegrees, 1.0)
        return cv2.warpAffine(image, rotationMatrix, (w, h))

    def _getHandOrientation(self, jointCoordinates: [()]):
        x1, y1 = jointCoordinates[0][0], jointCoordinates[0][1]
        x2, y2 = jointCoordinates[12][0], jointCoordinates[12][1]
        return math.degrees(math.atan2(y2-y1, x2-x1)) + 90

    def _detectHands(self, image: numpy.ndarray, minDetectionConfidence: float = 0.5, minTrackingConfidence: float = 0.5, maxNumHands: int = 1):
        with self.mpHands.Hands(min_detection_confidence = minDetectionConfidence, min_tracking_confidence = minTrackingConfidence, max_num_hands = maxNumHands) as hands:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            return results
    
    def _getImageBorders(self, jointCoordinates: [()]) -> ():
        xCoordinates = [x[0] for x in jointCoordinates]
        yCoordinates = [y[1] for y in jointCoordinates]
        return (min(yCoordinates) - self.cropMargin, max(yCoordinates) + self.cropMargin, min(xCoordinates) - self.cropMargin, max(xCoordinates) + self.cropMargin)
    
    # More information about the meaning of the coordinates:
    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    def _getJointCoordinates(self, image: numpy.ndarray, results) -> []:
        if (results.multi_hand_landmarks == None):
            raise Exception("Could not identify a hand") 
            
        coordinates = []
        for handLandmarks in results.multi_hand_landmarks:
            imageHeight, imageWidth, _ = image.shape
            for _, landmark in enumerate(handLandmarks.landmark):
                coordinates.append((int(landmark.x * imageWidth), int(landmark.y * imageHeight)))
        return coordinates

    def _drawHandConnections(self, image: numpy.ndarray, results):
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, handLandmarks, self.mpHands.HAND_CONNECTIONS)