from sys import flags
import cv2, os
import numpy as np
import pickle
import RPi.GPIO as GPIO
from time import sleep
from PIL import Image

id = 0

names = ["None"]

relay_pin = 13
GPIO.setmode(GPIO.BOARD)
GPIO.setup(relay_pin, GPIO.OUT)
p = GPIO.PWM(relay_pin, 80)

with open("labels", "rb") as f:
    dictionary = pickle.load(f)
    f.close()

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

path = os.path.dirname(os.path.abspath(__file__))

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (100,100), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x,y), (x + w, y + h), (0,255,0),2)

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less then 100 => "0" is perfect match
        if (confidence < 100):
            p.start(2.5)
            id = names[id]
            confidence = " {0}%".format(round(100 - confidence))
        else:
            p.stop(0)
            id = "unknwon"
            confidence = " {0}%".format(round(100 - confidence))

        cv2.putText(im, str(id), (x + 5, y - 5), font, 1, (255,255,255),2)
        cv2.putText(im, str(confidence), (x + 5, h - 5), font, 1, (255,255,0),1)

    cv2.imshow("Camera",im)

    k = cv2.waitKey(10) & 0xff # Press "ESC" for exiting video
    if k == 27:
        break

# Cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destryAllWindows()