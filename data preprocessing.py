import os
from PIL import Image
import cv2

#Images folder
path='images/Male/'

#Delete special char from file names
for filename in os.listdir(path):
    os.rename(path + filename, path + filename.replace('ć',''))
    
#Convert jpg,jpeg to png and delete old format images
files = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG')]
#Convert each file to JPEG
for filename in files:
    full_image_path=os.path.join(path, filename)
    image = Image.open(full_image_path)
    image.convert('RGB').save(os.path.join(path, os.path.splitext(filename)[0] + '.png'))
    image.close()
    os.remove(full_image_path)

def save_faces(cascade, imgname):
    img = cv2.imread(os.path.join(path, imgname))
    #Convert into grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    for i, face in enumerate(cascade.detectMultiScale(gray, 1.1, 4)):
        x, y, w, h = face
        sub_face = gray[y:y + h, x:x + w]
        #face_file_name = "faces/" + imgname + "_" + str(img) 
        #cv2.imwrite(face_file_name, sub_face)
        #cv2.imshow("face",sub_face) 
        #cv2.waitKey(0)  
        cv2.imwrite(os.path.join("New_Data/Male/", "{}_{}.png".format(imgname, i)), sub_face)
        
if __name__ == '__main__':
    face_cascade ='haarcascade_frontalface_alt2.xml'
    cascade = cv2.CascadeClassifier(face_cascade)
    #Iterate through files
    for f in [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]:
        save_faces(cascade, f)