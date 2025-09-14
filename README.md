# Facial Recognition Attendance System

A web-based attendance system using facial recognition technology powered by Flask, OpenCV, and machine learning.

## Features
*Real-time Face Recognition*: Uses Haar Cascade and K-Nearest Neighbors (KNN) algorithm

*Student Registration*: Capture facial data through webcam directly in the web browser

*Attendance Tracking*: Automatically records attendance with timestamps

*Attendance Reports*: View attendance records in CSV format

***Student Management*: Add/remove students from the system

*Web-based Camera*: Camera feed integrated directly into the web interface

*Real-time Updates*: WebSocket support for live camera streaming

*Responsive Design*: Works on both desktop and mobile devices

- **Face Recognition**: Uses OpenCV and machine learning for face detection
- **Student Registration**: Add students by capturing their facial data
- **Attendance Tracking**: Automatically record attendance with timestamps
- **Simple Interface**: Easy-to-use web interface
- **Attendance Reports**: View and download attendance records
- **Student Management**: Add or remove students from the system

- **Backend:** Python, Flask, Flask-SocketIO  
- **Computer Vision:** OpenCV, Haar Cascade  
- **Machine Learning:** scikit-learn, KNN classifier  
- **Frontend:** HTML5, CSS3, JavaScript, WebSocket  
- **Data Storage:** SQLite database with binary face data storage  
- **Real-time Communication:** WebSocket for live camera streaming  

- Python 3.7 or higher
- Webcam
- Internet connection for dependencies

## Installation

1. Download or clone this project
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your web browser and go to:
   ```
   http://localhost:5000
   ```

## How to Use

### Adding Students
1. Click "Register Student"
2. Enter the student's name and ID
3. Click "Start Face Capture"
4. Look at the camera until it collects enough face data
5. The system will save the student automatically

### Taking Attendance
1. Click "Take Attendance"
2. Look at the camera
3. The system will recognize your face and mark attendance automatically
4. Each person can only mark attendance once per day

### Viewing Records
1. Click "View Attendance" to see all attendance records
2. Click on any date to see who was present
3. Use "Manage Students" to add or remove students

## Files

- `app.py` - Main Flask application
- `models.py` - Database models
- `camera_service.py` - Face recognition service
- `templates/` - HTML templates
- `static/` - CSS and JavaScript files
- `requirements.txt` - Required Python packages

## Notes

- Make sure you have good lighting when registering faces
- Keep your face centered in the camera view
- The system works best with clear, front-facing photos
