#Import Dependencies
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import Image
from keras.models import load_model
from keras.preprocessing import image

model = load_model("Model/Output/Emotion_Gender_Model.keras")

#Creating two class dictionary
emotion_classes = {0:'Happy', 1:'Sad'}
gender_classes = {0:'Male', 1:'Female'}

#Read downloaded test image in Opencv
test_img = cv2.imread('Model/Photos/1happy.png')
test_img = cv2.imread('Model/Photos/1sad.png')

#Convert image to gray scale OpenCV
gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

#Define haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  
#Detect face using haar cascade classifier
faces_coordinates = face_classifier.detectMultiScale(gray_img, 1.3, 5)
 
i = 0
#Draw a rectangle around the faces
for (x, y, w, h) in faces_coordinates:
    i = i + 1
    #Draw rectangle around face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
         
    #Crop face from image
    cropped_face = gray_img[y:y+h, x:x+w]

    final_image1 = cv2.resize(cropped_face, (224, 224), interpolation = cv2.INTER_AREA)
    final_image_array = np.array(final_image1)
    input_test = np.expand_dims(final_image_array, axis = 0) ## Need 4th dimension
    output_test = emotion_classes[np.argmax(model.predict(input_test))]
    output_str = str(i) + ": " + output_test
    print(output_str)
    
    #Define opencv font style
    font = cv2.FONT_HERSHEY_SIMPLEX
    col = (0, 0, 255)
    cv2.putText(test_img, output_test, (x, y), font, 2, col, 2)
    
    #Crop face from image
    cropped_face2 = gray_img[y:y+h, x:x+w]

    final_image2 = cv2.resize(cropped_face2, (224, 224), interpolation = cv2.INTER_AREA)
    final_image_array2 = np.array(final_image2)
    input_test2 = np.expand_dims(final_image_array2, axis = 0) ## Need 4th dimension
    output_test2 = gender_classes[np.argmax(model.predict(input_test2))]
    output_str2 = str(i) + ": " + output_test2
    print(output_str2)
    
    #Define opencv font style
    font = cv2.FONT_HERSHEY_SIMPLEX
    col = (0, 0, 255)
    cv2.putText(test_img, output_test2, (x, y + h + 25), font, 2, col, 2)
     
plt.imshow(cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB))
plt.show()