import cv2
import random

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while True:
    frame_read, frame = cam.read()
    grayScaledImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cordinants = trained_face_data.detectMultiScale(frame)
    for (x,y,w,h) in face_cordinants :
        cv2.rectangle(frame, ( x, y),(x+ w , y+ h),(random.randrange(255),random.randrange(255),random.randrange(255)), 2)

    cv2.imshow('predictedimg', frame)
    cv2.waitKey(1)


