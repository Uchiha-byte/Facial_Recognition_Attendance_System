import cv2
import pickle
import numpy as np
import os
import argparse
from flask import Flask
from models import db, Student

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def print_message(message):
    print(message)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--id', required=True)
    args = parser.parse_args()

    name = args.name
    user_id = args.id

    with app.app_context():
        existing_student = Student.query.filter_by(student_id=user_id).first()
        if existing_student:
            print_message(f"ID {user_id} already exists. Exiting the program.")
            return

    video = cv2.VideoCapture(0)
    
    if not video.isOpened():
        print_message("Error: Could not open camera. Please check if camera is connected.")
        return

    # Set camera properties for better quality
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_FPS, 30)

    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces_data = []
    i = 0

    print_message("Starting facial data collection. Please look at the camera.")
    print_message("Press 'q' to quit or wait for 50 faces to be collected.")

    while True:
        ret, frame = video.read()

        if not ret:
            print_message("Failed to capture frame. Exiting the program.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for better face detection
        gray = cv2.equalizeHist(gray)
        
        # Use improved face detection parameters
        faces = face_detect.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Validate face size and quality
            if w >= 50 and h >= 50:
                # Use grayscale for consistent processing (same as recognition)
                crop_img = gray[y:y + h, x:x + w]
                
                # Apply preprocessing for better quality
                crop_img = cv2.equalizeHist(crop_img)
                crop_img = cv2.GaussianBlur(crop_img, (3, 3), 0)
                
                resize_img = cv2.resize(crop_img, (50, 50))

                if len(faces_data) < 50 and i % 8 == 0:
                    faces_data.append(resize_img)

                i += 1
                
                # Draw bounding box with progress indicator
                progress = len(faces_data) / 50
                if progress >= 1.0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green when complete
                elif progress >= 0.5:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow when half done
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red when starting
                
                # Add progress text
                cv2.putText(frame, f"Progress: {len(faces_data)}/50", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Face too small
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)  # Gray for small face
                cv2.putText(frame, "Move closer", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Face Capture - Press 'q' to quit", frame)

        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) == 50:
            print_message("Facial data collection completed. Saving your data.")
            break

    video.release()
    cv2.destroyAllWindows()

    if len(faces_data) == 0:
        print_message("No face data collected. Please try again.")
        return

    faces_data = np.array(faces_data)
    faces_data = faces_data.reshape(len(faces_data), -1)

    with app.app_context():
        new_student = Student(
            student_id=user_id,
            name=name
        )
        new_student.set_face_data(faces_data)
        
        db.session.add(new_student)
        db.session.commit()
        
        print_message(f"Student {name} (ID: {user_id}) added to database successfully!")
        print_message(f"Collected {len(faces_data)} face samples.")

    print_message("Your facial data has been saved successfully. Thank you!")

if __name__ == "__main__":
    main()
