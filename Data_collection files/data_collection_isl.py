import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0) #camera id == 0
detector = HandDetector(maxHands = 2)

offset = 20
imgSize = 300
counter = 0

folder = 'D:/asl_project_own/dataset_isl/Z' 

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    try:
        if hands:
            # Set the coordinates of the second hand outside image bounds if only one hand is detected
            if len(hands) == 1:
                x2, y2 = img.shape[1] + 100, img.shape[0] + 100
            else:
                x2, y2 = hands[1]['bbox'][0], hands[1]['bbox'][1]

            # Get the bounding box of all hands
            x, y, w, h = hands[0]['bbox']
            for hand in hands[1:]:
                x = min(x, hand['bbox'][0])
                y = min(y, hand['bbox'][1])
                w = max(w, hand['bbox'][0] + hand['bbox'][2] - x)
                h = max(h, hand['bbox'][1] + hand['bbox'][3] - y)

            imgWhite = np.ones((imgSize, imgSize,3),np.uint8)*255 # multiplying by 255 makes use of color, default value is 1 which gives black output
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
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