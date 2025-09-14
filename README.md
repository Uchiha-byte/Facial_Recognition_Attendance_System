# 📸 **Facial Recognition Attendance System**

A **web-based attendance system** that leverages **facial recognition technology** using **Flask**, **OpenCV**, and **machine learning**. Designed to make attendance tracking simple, secure, and efficient!

---

## 🚀 **Features**

- ✅ **Real-time Face Recognition**  
  Powered by Haar Cascade and K-Nearest Neighbors (KNN) algorithms for accurate face detection.

- ✅ **Student Registration**  
  Capture facial data through webcam directly in the web interface.

- ✅ **Attendance Tracking**  
  Automatically records attendance with timestamps.

- ✅ **Attendance Reports**  
  View and download attendance records in CSV format for easy tracking.

- ✅ **Student Management**  
  Add or remove students from the system effortlessly.

- ✅ **Web-based Camera**  
  Integrated camera feed in the browser with live updates using WebSocket.

- ✅ **Real-time Communication**  
  WebSocket support for smooth live camera streaming.

- ✅ **Responsive Design**  
  Optimized for both desktop and mobile devices.

---

## ⚙ **Tech Stack**

### 🔧 **Backend**
- Python 3.7+
- Flask, Flask-SocketIO

### 🤖 **Computer Vision**
- OpenCV (Haar Cascade)

### 🧠 **Machine Learning**
- scikit-learn (KNN classifier)

### 🌐 **Frontend**
- HTML5, CSS3, JavaScript, WebSocket

### 💾 **Data Storage**
- SQLite with binary face data storage

---

## 📋 **Prerequisites**

✔ Python 3.7 or higher  
✔ Webcam for face capture  
✔ Internet connection to install dependencies  

---

## 💻 **Installation Guide**

1️⃣ Clone or download the project  
```bash
git clone https://github.com/your-repo/facial-recognition-attendance.git

2️⃣ Install the required Python packages

pip install -r requirements.txt


3️⃣ Run the Flask application

python app.py


4️⃣ Open the web app in your browser

http://localhost:5000

🧑‍🎓 How to Use
➕ Adding Students

Click Register Student

Enter the student's name and ID

Click Start Face Capture

Look into the camera until the face data is collected

The student is saved automatically!

✅ Taking Attendance

Click Take Attendance

Look at the camera

The system recognizes your face and marks attendance

Attendance can be recorded only once per day per student

📊 Viewing Records

Click View Attendance to see attendance logs

Select any date to view attendance details

Use Manage Students to add or delete students

📂 Project Structure
facial-recognition-attendance/
├── app.py              # Main Flask application
├── models.py           # Database models and schema
├── camera_service.py   # Face recognition logic
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
├── static/             # CSS and JavaScript files
└── README.md           # Project documentation
