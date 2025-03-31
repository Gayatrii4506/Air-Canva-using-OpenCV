import cv2
import numpy as np
import os
import HandTrackingModule as htm


brushThickness = 15
eraserThickness = 50

# Load header images
folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = [cv2.resize(cv2.imread(f'{folderPath}/{imPath}'), (1280, 125)) for imPath in myList]
header = overlayList[0]
drawColor = (255, 0, 255)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


# Initialize hand detector with higher confidence
detector = htm.handDetector(detectionCon=0.8)
xp, yp = 0, 0
imgCanvas = np.zeros((720,1280, 3), np.uint8)

while True:
    # Capture frame and flip for mirror effect
    success, img = cap.read()
    if not success:
        continue  # Skip frame if capture fails

    img = cv2.flip(img, 1)

    # Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        
        # Extract finger tip positions
        x1, y1 = lmList[8][1:]  # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger

        # Get which fingers are up
        fingers = detector.fingersUP()
        print("Fingers detected:", fingers)  # Debugging print

        if len(fingers) >= 3:
            # Selection Mode: If both index & middle fingers are up
            if fingers[1] and fingers[2]:
                xp ,yp = 0, 0
                print("Selection Mode")
                #Checking for the click
                if y1 < 125:
                    if 250 < x1 <450:
                        header = overlayList[0]
                        drawColor = (255, 0, 0)
                    elif 550 < x1 <750:
                        header = overlayList[1]
                        drawColor = (255, 0, 255)
                    elif 800 < x1 <950:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 <1200:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            # Drawing Mode: If only index finger is up
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                print("Drawing Mode")
                if xp ==0 and yp == 0:
                    xp,yp = x1,y1
                

                if drawColor == (0,0,0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor,eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)


                xp,yp = x1,y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    # Set header image at the top
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    # Show output
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)
