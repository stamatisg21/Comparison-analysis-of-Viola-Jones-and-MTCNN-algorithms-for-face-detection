import cv2
import os
from mtcnn import MTCNN

# Function to detect faces in an image, crop them, and save as separate images
def detect_faces_and_save(image_path, output_dir):
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize MTCNN detector
    detector = MTCNN()
    
    # Detect faces
    faces = detector.detect_faces(img_rgb)
    
    # Iterate through detected faces
    for i, face in enumerate(faces):
        # Get bounding box coordinates
        x, y, w, h = face['box']
        
        # Crop the detected face
        face_crop = img[y:y+h, x:x+w]
        
        # Save the cropped face as a separate image
        face_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_face_{i}.jpg"
        output_path = os.path.join(output_dir, face_filename)
        cv2.imwrite(output_path, face_crop)
        print(f"Face {i+1} saved as: {output_path}")

# Example usage
if __name__ == "__main__":
    # Directory containing images
    dataset_path = "C:/Users/user/Desktop/biomedical/RAF/fear"
    # Output directory for cropped faces
    output_dir = "C:/Users/user/Desktop/biomedical/outMTC"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Iterate through each image in the dataset directory
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):  # Assuming all images are in JPG format
            image_path = os.path.join(dataset_path, filename)
            detect_faces_and_save(image_path, output_dir)
