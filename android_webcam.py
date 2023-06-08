# Import OpenCV
import cv2
import requests 
import numpy as np
url="http://ip_over_LAN/shot.jpg"
# Define haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
# Read webcam video
cap = cv2.VideoCapture(0)
 
while True:
    # Run video frame by frame
    # read_ok, frame = cap.read()
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, 1)
    frame = cv2.resize(frame,dsize=(800,1200))

    # For images 
    # img = cv2.imread('input_image.jpg')

    labels = []
    # Convert image to gray scale OpenCV
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Detect face using haar cascade classifier
    faces_coordinates = face_classifier.detectMultiScale(gray_img)
 
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    cv2.imshow('Face Detector', frame)

    # Saving the image
    # cv2.imwrite('detected_faces.png', img)
 
    # show output image
    # cv2.imshow('image', img)
 
    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
 
cap.release()
cv2.destroyAllWindows()