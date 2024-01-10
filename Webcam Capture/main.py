import cv2, os
from datetime import datetime

IDENTIFIER = "I"
AMOUNT_OF_PICTURES = 100
BEGIN_INDEX = 20
FRAMES_PER_SECOND = 4

print("Loading camera")
cam = cv2.VideoCapture(0)
saveToPath = f"{os.path.dirname(os.path.realpath(__file__))}\\{IDENTIFIER}"
if (not os.path.exists(saveToPath)):
    os.makedirs(saveToPath)

lastCaptureTime = datetime.now()
capturedImages = BEGIN_INDEX
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    deltaS = (datetime.now() - lastCaptureTime).total_seconds()
    if (deltaS > 1/FRAMES_PER_SECOND):
        print(f"Capturing image {capturedImages+1} of {AMOUNT_OF_PICTURES}")
        cv2.imwrite(os.path.join(saveToPath, f"{IDENTIFIER}.{capturedImages}.png"), frame)

        capturedImages += 1
        lastCaptureTime = datetime.now()
        if (capturedImages >= AMOUNT_OF_PICTURES):
            break
    
    c = cv2.waitKey(1)
    if c == 27:
        break

print(f"Saved {AMOUNT_OF_PICTURES} images to path: '{saveToPath}'")