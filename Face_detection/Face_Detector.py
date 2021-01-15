import cv2
import random

#loading the trained model into the application
trained_face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Getting the default web-cam into a variable
cam = cv2.VideoCapture(0)

while True:
    #Read the frames from your web-cam
    frame_read, frame = cam.read()

    #Extract the array of detected coordinates 
    face_cordinants = trained_face_detector.detectMultiScale(frame)

    #Draw the squire using the identified coordinates
    for (x,y,w,h) in face_cordinants :
        cv2.rectangle(frame, ( x, y),(x+ w , y+ h),(random.randrange(255),random.randrange(255),random.randrange(255)), 2)

    #Start the GUI to open up the web-cam
    cv2.imshow('predictedimg', frame)
    
    cv2.waitKey(1)


