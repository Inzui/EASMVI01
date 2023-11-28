import cv2
import mediapipe as mp

class PhotoProcessor:
    def __init__(self, cropMargin = 50) -> None:
        self.cropMargin = cropMargin
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def run(self, imagePath, showImage = False):
        image = cv2.imread(imagePath) 
        processedImage = self.processImage(image)
        imageBorders = self.getImageBorders(image, processedImage)
        croppedImage = image[imageBorders[0]:imageBorders[1], imageBorders[2]:imageBorders[3]]

        if (showImage):
            self.drawHandConnections(image, processedImage)
            cv2.imshow("Test", croppedImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        jointCoordinates = self.getJointCoordinates(croppedImage, processedImage)
        if (len(jointCoordinates) != 21):
            raise Exception("Too few/many joints")

        return jointCoordinates

    def processImage(self, img):
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.hands.process(grayImage)
    
    def getImageBorders(self, img, processedImage) -> ():
        jointCoordinates = self.getJointCoordinates(img, processedImage)
        xCoordinates = [x[0] for x in jointCoordinates]
        yCoordinates = [y[1] for y in jointCoordinates]

        return (min(yCoordinates) - self.cropMargin, max(yCoordinates) + self.cropMargin, min(xCoordinates) - self.cropMargin, max(xCoordinates) + self.cropMargin)
    
    def getJointCoordinates(self, img, processedImage) -> []:
        coordinates = []
        if processedImage.multi_hand_landmarks:
            for handLms in processedImage.multi_hand_landmarks:
                for _, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    coordinates.append((cx, cy))
            return coordinates

    def drawHandConnections(self, img, results):
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