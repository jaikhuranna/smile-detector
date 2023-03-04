import cv2 as cv
import numpy as np
from PIL import Image
from keras import models

fclass = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eclass = cv.CascadeClassifier('haarcascade_eye.xml')

def fd(img, size=0.5):    
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = fclass.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img

    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eclass.detectMultiScale(roi_gray)
    
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    roi_color = cv.flip(roi_color,1)
    return roi_color
    
#Load the saved model
model = models.load_model('model140.h5')
cap = cv.VideoCapture(0)
_, frame = cap.read()
frame = fd(frame)

while True:        #Convert the captured frame into RGB
    im = frame
    im = np.resize(im, (140, 140))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = int(model.predict(img_array)[0][0])

    #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
    if prediction == 0:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow("Capturing", frame)
    key=cv.waitKey(1)
    if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
