import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
cap = cv2.VideoCapture(0) #camera id == 0
detector = HandDetector(maxHands = 1)

offset = 20
imgSize = 300
counter = 0

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

classifier = Classifier("model_asl/keras_model.h5","model_asl/labels.txt")
#D:/asl_project_own/latest_asl_model/keras_model.h5
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    try:
        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize,3),np.uint8)*255 # multiplying by 255 makes use of color, default value is 1 which gives black output
            imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
            
            imgCropShape = imgCrop.shape
            
            
            aspectRatio = h/w
            if aspectRatio >1: #for width
                k = imgSize/h #contastant k
                wCal= math.ceil(k*w) #caluclatated width
                imgResize = cv2.resize(imgCrop, (wCal,imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize- wCal)/2)#width gap
                imgWhite[:,wGap:wCal+wGap] = imgResize

                prediction, index =classifier.getPrediction(imgWhite, draw = False)
                print(prediction, index)
                
            else: #for height
                k = imgSize/w 
                hCal= math.ceil(k*w) 
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize- hCal)/2)
                imgWhite[hGap:hCal+hGap,:] = imgResize
                prediction, index =classifier.getPrediction(imgWhite, draw = False)
            cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED)
            cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255), 2)
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
    except:
        pass
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1) 
    if key == ord('q'):
        break


