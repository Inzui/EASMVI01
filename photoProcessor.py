import cv2, numpy, math
import mediapipe as mp

class PhotoProcessor:
    def __init__(self, cropMargin: int = 25):
        self.cropMargin = cropMargin
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

    def run(self, arg1, showImage: bool = False) -> [()]:
        """
            Gets the landmarks of a hand in a CV2 image and returns them. 
            Landmarks of the hand always get translated so the middle vinger points upwards from the wrist.
            More information about the meaning of the landmarks:
            https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
        """
        if (isinstance(arg1, str)):
            image = cv2.imread(arg1)
            return self.__run(image, showImage)
        else:
            return self.__run(arg1, showImage)

    def __run(self, image: numpy.ndarray, showImage: bool) -> [()]:
        # Find the hand in the image.
        results = self.__detectHands(image)
        handCoordinates = self.__getJointCoordinates(image, results)

        # Rotate the hand so it is always oriented upwards.
        image = self.__rotateImage(image, self.__getHandOrientation(handCoordinates))
        results = self.__detectHands(image)

        # Crop the image so only the hand remains in the image.
        image = self.__cropImage(image, self.__getJointCoordinates(image, results))
        results = self.__detectHands(image)
        
        if (showImage):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.__drawHandConnections(image, results)
            cv2.imshow("Hand Connections", image)
        
        # Return the joint coordinates.
        return self.__getJointCoordinates(image, results)

    def __cropImage(self, image: numpy.ndarray, jointCoordinates: [()], desiredHeight: int = 500) -> numpy.ndarray:
        """
            Crops the hand from the image and converts the image to the desired height, respecting the aspect ratio of the original image, and returns it.
        """
        imageBorders = self.__getImageBorders(jointCoordinates)
        image = image[imageBorders[0]:imageBorders[1], imageBorders[2]:imageBorders[3]]
        image = numpy.ascontiguousarray(image)
        imageHeight, imageWidth, _ = image.shape
        aspectRatio = float(imageWidth) / float(imageHeight)
        desiredWidth = int(desiredHeight * aspectRatio)
        image = cv2.resize(image, (desiredWidth, desiredHeight), interpolation = cv2.INTER_AREA)
        return image

    def __rotateImage(self, image: numpy.ndarray, rotationDegrees: float) -> numpy.ndarray:
        """
            Rotates the image x degrees and returns it.
        """
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        rotationMatrix = cv2.getRotationMatrix2D(center, rotationDegrees, 1.0)
        return cv2.warpAffine(image, rotationMatrix, (w, h))

    def __getHandOrientation(self, jointCoordinates: [()]) -> float:
        """
            Gets the orientation of the hand in degrees by comparing the angle between the wrist and the middle vinger and returns it.
        """
        x1, y1 = jointCoordinates[0][0], jointCoordinates[0][1]
        x2, y2 = jointCoordinates[12][0], jointCoordinates[12][1]
        return math.degrees(math.atan2(y2-y1, x2-x1)) + 90

    def __detectHands(self, image: numpy.ndarray, minDetectionConfidence: float = 0.3, minTrackingConfidence: float = 0.5, maxNumHands: int = 1):
        """
            Detects the hand in the CV2 image using MediaPipe and returns the landmarks.
        """
        with self.mpHands.Hands(min_detection_confidence = minDetectionConfidence, min_tracking_confidence = minTrackingConfidence, max_num_hands = maxNumHands) as hands:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            return results
    
    def __getImageBorders(self, jointCoordinates: [()]) -> ():
        """
            Gets the image border by using the smallest x, biggest x, smallest y and biggest y and adding the Crop Margin. Then returns the image border coordinates.
        """
        xCoordinates = [x[0] for x in jointCoordinates]
        yCoordinates = [y[1] for y in jointCoordinates]
        return (min(yCoordinates) - self.cropMargin, max(yCoordinates) + self.cropMargin, min(xCoordinates) - self.cropMargin, max(xCoordinates) + self.cropMargin)
    
    def __getJointCoordinates(self, image: numpy.ndarray, results) -> []:
        """
            Gets the joint coordinates by extracting them from the MediaPipe landmarks and returns them.
        """
        if (results.multi_hand_landmarks == None):
            raise Exception("Could not identify a hand") 
            
        coordinates = []
        for handLandmarks in results.multi_hand_landmarks:
            imageHeight, imageWidth, _ = image.shape
            for _, landmark in enumerate(handLandmarks.landmark):
                coordinates.append((int(landmark.x * imageWidth), int(landmark.y * imageHeight)))
        return coordinates

    def __drawHandConnections(self, image: numpy.ndarray, results):
        """
            Draws dots on the joints of the CV2 Image and draws lines between them by extracting the MediaPipe hand landmarks.
        """
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, handLandmarks, self.mpHands.HAND_CONNECTIONS)