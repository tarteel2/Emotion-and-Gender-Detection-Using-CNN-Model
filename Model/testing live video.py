#Import Dependencies
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model

model = load_model("Model/Output/Emotion_Gender_Model.keras")

emotion_classes = {0:'Happy', 1:'Sad'}
gender_classes = {0:'Male', 1:'Female'}

#Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Start capturing video
cap = cv2.VideoCapture(0)

while True:
    #Capture frame-by-frame
    ret, frame = cap.read()

    #Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces in frame
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)

    for (x, y, w, h) in faces:
        #Extract face ROI (Region of Interest)
        face_roi = gray_frame[y:y + h, x:x + w]
        #Resize image to match model's expected sizing
        img = cv2.resize(face_roi ,(224,224))     
        img = img.reshape(-1,224,224,1) 
        
        #Extract face ROI (Region of Interest)
        face_col = gray_frame[y:y + h, x:x + w]
        #Resize image to match model's expected sizing
        img2 = cv2.resize(face_col ,(224,224))
        img2 = img2.reshape(-1,224,224,1)
        
        #Perform emotion and gender analysis on face ROI
        #Determine emotion and gender
        predicted_emotion = model.predict(img)
        predicted_gender = model.predict(img2)
        
        index = int(np.argmax(predicted_emotion))
        index2 = int(np.argmax(predicted_gender))

        #Draw rectangle around face and label with predicted emotion and gender
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_classes[index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, gender_classes[index2], (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    #Display resulting frame
    cv2.imshow('Real-time Emotion and Gender Detection', frame)

    #Press 'q' to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

#Release capture and close all windows
cap.release()
cv2.destroyAllWindows()