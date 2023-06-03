import cv2
from HandTrackingModule import HandDetector
from ClassificationModule import Classifier
from UTF8ClassificationModule import UTF8Classifier
import numpy as np
import math
import time
import tkinter as tk
from PIL import ImageTk, Image

# Initialize variables
cap = cv2.VideoCapture(0)  # Camera ID == 0
detector1 = HandDetector(maxHands=1)
detector3 = HandDetector(maxHands=1)
detector2 = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
          "X", "Y", "Z"]
labels_r = ["A","Б","B","Г","Д","Ё","Ж","3","Й","K","Л","M","H","O","П","P","C","T","y","Ф","X"]

classifier1 = Classifier("model_asl/keras_model.h5",
                         "model_asl/labels.txt")
classifier2 = Classifier("model_isl/keras_model.h5",
                         "model_isl/labels.txt")
classifier3 = UTF8Classifier("model_rsl/keras_model.h5", "model_rsl/labels.txt")
use_code1 = True
use_code2 = False
use_code3 = False

# Create Tkinter window
window = tk.Tk()
window.title("Multilingual Sign Language Recognizer")

# Tkinter icon
window.iconbitmap("logo.ico")

# Create a label widget to display the OpenCV output
label = tk.Label(window)
label.pack()

# Define switch code functions
def switch_to_code1():
    global use_code1, use_code2, use_code3
    use_code1 = True
    use_code2 = False
    use_code3 = False
    code_label.config(text="Sign Language: American Sign Language", font=("Arial", 14), fg="white", bg="black")

def switch_to_code2():
    global use_code1, use_code2, use_code3
    use_code1 = False
    use_code2 = True
    use_code3 = False
    code_label.config(text="Sign Language: Indian Sign Language",font=("Arial", 14), fg="white", bg="black")

def switch_to_code3():
    global use_code1, use_code2, use_code3
    use_code1 = False
    use_code2 = False
    use_code3 = True
    code_label.config(text="Sign Language: Russian Sign Language",font=("Arial", 14), fg="white", bg="black")

# Define show_chart function
def show_chart(chart_path):
    global current_chart_window

    # Close the current chart window if it exists
    if current_chart_window is not None:
        current_chart_window.destroy()

    chart = Image.open(chart_path)
    chart = chart.resize((400, 400), Image.ANTIALIAS)
    chartTk = ImageTk.PhotoImage(chart)

    # Create a new Tkinter window
    chart_window = tk.Toplevel(window)
    chart_window.title("Sign Language Charts")
    
    # Create a label widget to display the chart
    chart_label = tk.Label(chart_window, image=chartTk)
    chart_label.pack()

    # Update the current chart window
    current_chart_window = chart_window

    # Run the Tkinter event loop for the chart window
    chart_window.mainloop()

# Create a frame for the first pair of buttons
frame1 = tk.Frame(window)
frame1.pack()

# Create switch code button
switch_button1 = tk.Button(frame1, text="American Sign Language", command=switch_to_code1, width=30, height=2)
switch_button1.pack(side=tk.LEFT)

# Create chart button
chart_button1 = tk.Button(frame1, text="ASL Chart", command=lambda: show_chart("Charts/ASL_CHART.png"), width=10, height=2)
chart_button1.pack(side=tk.LEFT)

# Create a frame for the second pair of buttons
frame2 = tk.Frame(window)
frame2.pack()

# Create switch code button
switch_button2 = tk.Button(frame2, text="Indian Sign Language", command=switch_to_code2, width=30, height=2)
switch_button2.pack(side=tk.LEFT)

# Create chart button
chart_button2 = tk.Button(frame2, text="ISL Chart", command=lambda: show_chart("Charts/ISL_CHART.jpg"), width=10, height=2)
chart_button2.pack(side=tk.LEFT)

# Create a frame for the third pair of buttons
frame3 = tk.Frame(window)
frame3.pack()

# Create switch code button
switch_button3 = tk.Button(frame3, text="Russian Sign Language", command=switch_to_code3, width=30, height=2)
switch_button3.pack(side=tk.LEFT)

# Create chart button
chart_button3 = tk.Button(frame3, text="RSL Chart", command=lambda: show_chart("Charts/RSL_CHART.png"), width=10, height=2)
chart_button3.pack(side=tk.LEFT)

# Create code label
code_label = tk.Label(window, text="Sign Language: American Sign Language", font=("Arial", 14), fg="white", bg="black")
code_label.pack()

# Initialize the current chart window
current_chart_window = None

# Define video loop function
def video_loop():
    global use_code1, use_code2, use_code3
    success, img = cap.read()
    imgOutput = img.copy()

    # Run code 1, code 2, or code 3 based on button state
    if use_code1:
        # Rest of the code for code 1...
        hands, img = detector1.findHands(img)
        try:
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier1.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier1.getPrediction(imgWhite, draw=False)
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                              (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        except:
            pass

    elif use_code2:
        # Rest of the code for code 2...
        hands, img = detector2.findHands(img)
        try:
            if len(hands) == 1 or len(hands) == 2:  # Check if there are one or two hands detected
                if len(hands) == 1:
                    x1, y1, w1, h1 = hands[0]['bbox']
                    x, y, w, h = x1, y1, w1, h1
                else:
                    x1, y1, w1, h1 = hands[0]['bbox']
                    x2, y2, w2, h2 = hands[1]['bbox']
                    x, y, w, h = min(x1, x2), min(y1, y2), max(x1 + w1, x2 + w2) - min(x1, x2), max(y1 + h1, y2 + h2) - min(
                        y1, y2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier2.getPrediction(imgWhite, draw=False)
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                              (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        except:
            pass

    elif use_code3:
        # Rest of the code for code 3...
        hands, img = detector3.findHands(img)
        try:
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier3.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier3.getPrediction(imgWhite, draw=False)
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                              (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels_r[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255),
                            2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        except:
            pass

    # Convert the OpenCV image to a PIL image
    imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgOutput)

    # Convert the PIL image to a Tkinter image
    imgTk = ImageTk.PhotoImage(image=imgPIL)

    # Update the label with the new image
    label.config(image=imgTk)
    label.image = imgTk

    # Call the video_loop function after 1ms
    window.after(1, video_loop)

# Start the video loop
video_loop()

# Start the Tkinter event loop
window.mainloop()
