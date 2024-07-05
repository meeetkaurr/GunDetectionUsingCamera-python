import numpy as np
import cv2
import imutils
import datetime

# Load the cascade classifier
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Initialize the camera
camera = cv2.VideoCapture(0)

firstFrame = None
gun_exist = False

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns in the frame
    gun = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    # Check if any guns are detected
    if len(gun) > 0:
        gun_exist = True
    else:
        gun_exist = False

    # Draw rectangle around detected guns
    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Security Feed", frame)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Print whether guns were detected
if gun_exist:
    print("Guns detected")
else:
    print("Guns not detected")

# Release the capture and close any OpenCV windows
camera.release()
cv2.destroyAllWindows()
