import cv2
import os

# Load the pre-trained face detection model
print(cv2.data.haarcascades)
face_cascade_path = "C:\\Users\\user\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Function to detect faces in an image, crop them, and save as separate images
def detect_faces_and_save(image_path, output_dir):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))
    # Iterate through detected faces
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the detected face
        face = img[y:y+h, x:x+w]
        # Save the cropped face as a separate image
        face_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_face_{i}.jpg"
        output_path = os.path.join(output_dir, face_filename)
        cv2.imwrite(output_path, face)
        print(f"Face {i+1} saved as: {output_path}")

# Example usage
if __name__ == "__main__":
    # Directory containing images
    dataset_path = "C:/Users/user/Desktop/biomedical/RAF/fear"
    # Output directory for cropped faces
    output_dir = "C:/Users/user/Desktop/biomedical/outViola"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Iterate through each image in the dataset directory
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):  # Assuming all images are in JPG format
            image_path = os.path.join(dataset_path, filename)
            detect_faces_and_save(image_path, output_dir)
