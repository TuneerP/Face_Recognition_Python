import cv2
import numpy as np
import dlib
import os

size = 4
haar_face_cascade = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Load Dlib's pre-trained shape predictor model
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Part 1: Create LBPHFaceRecognizer
print('Recognizing Face. Please be in sufficient light...')

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (130, 100))  # Resize images
            images.append(img)
            labels.append(int(label))
        id += 1

(width, height) = (130, 100)

# Create a Numpy array from the two lists above
images = np.array(images)
labels = np.array(labels)

# OpenCV trains a model from the images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use LBPHFaceRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_face_cascade)

webcam = cv2.VideoCapture(0)

# Parameters for motion detection and blink detection
ret, prev_frame = webcam.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
motion_threshold = 10000  # Adjust this threshold based on your testing
blink_counter = 0
blink_threshold = 5  # Adjust based on typical blink behavior

while True:
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check for motion
    diff_frame = cv2.absdiff(prev_gray, gray)
    _, thresh_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)
    motion = np.sum(thresh_frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 100:
            name = names[prediction[0]]
            cv2.putText(frame, f'{name} - {prediction[1]:.0f}', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(frame, 'Invalid Picture', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        # Convert the face region back to a format dlib can work with
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, dlib_rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = shape[36:42]
        right_eye = shape[42:48]

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)

        if (ear_left + ear_right) / 2.0 < 0.25:
            blink_counter += 1
            if blink_counter >= blink_threshold:
                cv2.putText(frame, 'Blink Detected', (x - 10, y + h + 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
                blink_counter = 0

        # Liveness detection
        if motion < motion_threshold:
            cv2.putText(frame, 'Static Image Detected', (x - 10, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        else:
            cv2.putText(frame, 'Live Person', (x - 10, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow('OpenCV', frame)
    prev_gray = gray

    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
