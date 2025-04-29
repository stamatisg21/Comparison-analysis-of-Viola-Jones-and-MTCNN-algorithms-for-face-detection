import cv2
import os
import time
from mtcnn import MTCNN

# Load Viola-Jones cascade classifier
face_cascade_path = "C:\\Users\\user\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)


# Initialize MTCNN detector
detector = MTCNN()

# Function to measure inference time for Viola-Jones algorithm
def measure_viola_jones_inference_time(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))
    end_time = time.time()
    return end_time - start_time

# Function to measure inference time for MTCNN algorithm
def measure_mtcnn_inference_time(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    faces = detector.detect_faces(img_rgb)
    end_time = time.time()
    return end_time - start_time

# Load an example image
image_path = "C:/Users/user/Desktop/biomedical/RAF/fear/train_07393.jpg"

# Measure inference time for Viola-Jones algorithm
viola_jones_time = measure_viola_jones_inference_time(image_path)

# Measure inference time for MTCNN algorithm
mtcnn_time = measure_mtcnn_inference_time(image_path)

# Print results
print("Viola-Jones Inference Time:", viola_jones_time, "seconds")
print("MTCNN Inference Time:", mtcnn_time, "seconds")
