# Face_Recognition_Python
This project utilizes OpenCV Library to make a Real-Time Face Detection using your webcam as a primary camera.

Approach/Algorithms used: 
 
 1.This project uses LBPH (Local Binary Patterns Histograms) Algorithm to detect faces. It labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number.
 2.LBPH uses 4 parameters : 
  (i) Radius: the radius is used to build the circular local binary pattern and represents the radius around the  central pixel. 
  (ii) Neighbors : the number of sample points to build the circular local binary pattern. 
  (iii) Grid X : the number of cells in the horizontal direction. 
  (iv) Grid Y : the number of cells in the vertical direction.
 3.The model built is trained with the faces with tag given to them, and later on, the machine is given a test data and machine decides the correct label for it.


How to Run:
Install all of the packages listed in requirements.txt.
Run create_data.py == the webcam will be turned on and take 30 pictures of the user(You can change the threshold)
The images will be saved in the datasets.
Download and unzip "shape_predictor_68_face_landmarks.dat" into the main directory
Run face_recognize.py == webcam will be turned on and check if the user is valid or not.
