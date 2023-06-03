import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0) #camera id == 0
detector = HandDetector(maxHands = 1)

offset = 20
imgSize = 300
counter = 0

folder = 'D:/asl_project_own/dataset_rsl/imgs' #set folder name

while True:
    success, img = cap.read()
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
            
            else: #for height
                k = imgSize/w 
                hCal= math.ceil(k*w) 
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize- hCal)/2)

                imgWhite[hGap:hCal+hGap,:] = imgResize
            
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
    except:
        pass
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("c"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord('q'):
        break


