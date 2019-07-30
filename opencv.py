
#IMPORT LIBRARIES
import cv2


#Import xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#USE webcam (Use 0 in paranthese for webcam) or path to video
cap = cv2.VideoCapture(0)

while 1:
    #Video is series of frames.
    #ret gives True if frame is captured otherwise false
    #img contains matrix of pixels of frame
    ret, img = cap.read()
    
    #converts img to gray scale (so that we can find dimensions of face esaily because 
    #color frames are 3 channel i.e. more complex where gray scale frames are 1 channel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #To detect faces in frame 
    #1.05 reflects scale factor which specifying how much the image size is reduced at each image scale.
    #current scale factor is 5%
    #faces contains the dimensions of face
    faces = face_cascade.detectMultiScale(gray, 1.05)
    print(faces)    
    for (x, y, w, h) in faces:
        #draws rectangle where face is detected (paramters : frame, vertex1 of rectangle,
        #vertex2 of rectangle, color, width of rectangle)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #extracting faces from frame to perform same process for eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        #find dimensions of eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #rectangle for eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
           
    cv2.imshow('Face and Eye detection', img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()