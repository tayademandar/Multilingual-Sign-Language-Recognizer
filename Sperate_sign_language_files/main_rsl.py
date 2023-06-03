import cv2
from cvzone.HandTrackingModule import HandDetector
from UTF8ClassificationModule import UTF8Classifier
import numpy as np
import math
import time
import sys
import io

# Set the default encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

cap = cv2.VideoCapture(0) # Camera ID == 0
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

labels = ["A","Б","B","Г","Д","Ё","Ж","3","Й","K","Л","M","H","M","O","П","P","C","T","y","Ф","X"]
classifier = UTF8Classifier("model_rsl/keras_model.h5", "model_rsl/labels.txt")

# Print all labels
for label in classifier.list_labels:
    print(label)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            
            imgCropShape = imgCrop.shape
            
            aspectRatio = h / w
            if aspectRatio > 1:  # for width
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal+wGap] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
                
            else:  # for height
                k = imgSize / w
                hCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
    except:
        pass

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1) 
    if key == ord('q'):
        break
