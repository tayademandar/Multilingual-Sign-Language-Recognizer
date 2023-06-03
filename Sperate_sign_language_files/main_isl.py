import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0) #camera id == 0
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
counter = 0

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

classifier = Classifier("model_isl/keras_model.h5","model_isl/labels.txt")

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    try:
        if hands:
            # Crop and resize both hands together
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']
            x, y, w, h = min(x1, x2), min(y1, y2), max(x1+w1, x2+w2)-min(x1, x2), max(y1+h1, y2+h2)-min(y1, y2)
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h/w
            if aspectRatio > 1: # for width
                k = imgSize/h
                wCal= math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            else: # for height
                k = imgSize/w 
                hCal= math.ceil(k*w) 
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize

            # Perform classification on the resulting image
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Draw classification results on the combined hand region
            cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Image", imgOutput)
        cv2.imshow("ImageWhite", imgWhite)

    except:
        pass

    key = cv2.waitKey(1) 
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
