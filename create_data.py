import cv2
import os

# Haar cascade file for face detection
haar_file = 'haarcascade_frontalface_default.xml'

# Dataset directory where images will be stored
datasets = 'datasets'

# Subfolder name for storing images (change as needed)
sub_data = 'pics'

# Create path for subfolder if it doesn't exist
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.makedirs(path)

# Dimensions for resizing captured images
width, height = 130, 100

# Initialize the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

# Initialize the webcam (0 for default webcam)
webcam = cv2.VideoCapture(0)

# Counter for the number of images captured
count = 1

# Capture loop (captures 30 images)
while count <= 30:
    # Read frame from webcam
    _, frame = webcam.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region and resize it
        face_roi = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face_roi, (width, height))

        # Save the resized face image
        cv2.imwrite(f'{path}/{count}.png', face_resize)

        # Increment image count
        count += 1

        # Break the loop if 30 images are captured
        if count > 30:
            break

    # Display the frame with rectangles around faces
    cv2.imshow('Capturing Faces', frame)

    # Wait for ESC key press to exit
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
