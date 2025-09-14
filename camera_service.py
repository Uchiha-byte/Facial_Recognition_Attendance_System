import cv2
import pickle
import numpy as np
import threading
import time
from datetime import datetime
from flask_socketio import emit
from models import db, Student, AttendanceRecord
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class CameraService:
    def __init__(self, socketio, app):
        self.socketio = socketio
        self.app = app
        self.video_capture = None
        self.face_detect = None
        self.is_running = False
        self.current_mode = None
        self.current_student = None
        self.faces_data = []
        self.face_count = 0
        self.knn = None
        self.id_to_name = {}
        self.attendance_data = None
        self.last_attendance_time = {}
        self.attendance_cooldown = 5
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        self.student_data_loaded = False
        self.last_data_load_time = 0
        self.data_cache_duration = 300
        
        # Accuracy improvement parameters
        self.confidence_threshold = 0.6  # Minimum confidence for recognition
        self.face_size_threshold = 50    # Minimum face size for processing
        self.recognition_history = []    # Track recent predictions for stability
        self.history_size = 5            # Number of recent predictions to consider
        
    def initialize_camera(self):
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                return False
                
        if self.face_detect is None:
            # Use improved face detection parameters
            self.face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            # Set camera properties for better quality
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            
        return True
    
    def start_register_mode(self, name, user_id):
        if not self.initialize_camera():
            return False
            
        self.current_mode = 'register'
        self.current_student = {'name': name, 'id': user_id}
        self.faces_data = []
        self.face_count = 0
        self.is_running = True
        
        thread = threading.Thread(target=self._register_loop)
        thread.daemon = True
        thread.start()
        return True
    
    def start_attendance_mode(self):
        if not self.initialize_camera():
            return False
        
        self.socketio.emit('loading_status', {
            'loading': True,
            'message': 'Loading student data for recognition...'
        })
            
        if not self._load_student_data():
            self.socketio.emit('loading_status', {
                'loading': False,
                'error': 'No students found or error loading student data. Please register students first.'
            })
            return False
        
        self.socketio.emit('loading_status', {
            'loading': False,
            'message': 'Recognition system ready!'
        })
            
        self.current_mode = 'attendance'
        self.is_running = True
        
        thread = threading.Thread(target=self._attendance_loop)
        thread.daemon = True
        thread.start()
        return True
    
    def stop_camera(self):
        self.is_running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.current_mode = None
        self.current_student = None
        self.faces_data = []
        self.face_count = 0
        self.attendance_data = None
    
    def _load_student_data(self):
        current_time = time.time()
        
        if (self.student_data_loaded and 
            current_time - self.last_data_load_time < self.data_cache_duration):
            return True
            
        try:
            with self.app.app_context():
                students = Student.query.all()
                
                if not students:
                    return False
                
                all_faces = []
                all_ids = []
                id_to_name = {}
                
                for student in students:
                    try:
                        face_data = student.get_face_data()
                        
                        if len(face_data.shape) == 2 and face_data.shape[1] == 7500:
                            pass
                        elif len(face_data.shape) == 3:
                            face_data = face_data.reshape(len(face_data), -1)
                        else:
                            continue
                        
                        # Use more samples for better accuracy
                        max_samples = min(len(face_data), 25)
                        if max_samples > 0:
                            indices = np.random.choice(len(face_data), max_samples, replace=False)
                            sampled_faces = face_data[indices]
                            
                            all_faces.append(sampled_faces)
                            all_ids.extend([student.student_id] * len(sampled_faces))
                            id_to_name[student.student_id] = student.name
                    except Exception as e:
                        continue
                
                if not all_faces:
                    return False
                
                FACES = np.vstack(all_faces)
                
                # Use improved KNN parameters for better accuracy
                self.knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', weights='distance')
                self.knn.fit(FACES, all_ids)
                self.id_to_name = id_to_name
                
                self.student_data_loaded = True
                self.last_data_load_time = current_time
                
                return True
        except Exception as e:
            return False
    
    def _register_loop(self):
        """Main loop for face registration"""
        while self.is_running and self.current_mode == 'register':
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization for better face detection
            gray = cv2.equalizeHist(gray)
            
            # Use improved face detection parameters
            faces = self.face_detect.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=6, 
                minSize=(self.face_size_threshold, self.face_size_threshold),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to RGB for web display - FIXED COLOR ISSUE
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process detected faces
            for (x, y, w, h) in faces:
                # Validate face size and quality
                if w >= self.face_size_threshold and h >= self.face_size_threshold:
                    # Use grayscale for consistent processing (same as recognition)
                    crop_img = gray[y:y + h, x:x + w]
                    
                    # Apply preprocessing for better quality
                    crop_img = cv2.equalizeHist(crop_img)
                    crop_img = cv2.GaussianBlur(crop_img, (3, 3), 0)
                    
                    resize_img = cv2.resize(crop_img, (50, 50))
                    
                    # Collect face data every 8th frame for better quality
                    if len(self.faces_data) < 50 and self.face_count % 8 == 0:
                        self.faces_data.append(resize_img)
                    
                    self.face_count += 1
                    
                    # Draw bounding box with progress indicator
                    progress = len(self.faces_data) / 50
                    if progress >= 1.0:
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green when complete
                    elif progress >= 0.5:
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow when half done
                    else:
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red when starting
                    
                    # Add progress text
                    cv2.putText(frame_rgb, f"Progress: {len(self.faces_data)}/50", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Face too small
                    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (128, 128, 128), 2)  # Gray for small face
                    cv2.putText(frame_rgb, "Move closer", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame_rgb)
            frame_bytes = buffer.tobytes()
            
            # Send frame to client
            self.socketio.emit('camera_frame', {
                'frame': frame_bytes.hex(),
                'faces_collected': len(self.faces_data),
                'mode': 'register'
            })
            
            # Check if collection is complete
            if len(self.faces_data) >= 50:
                self._save_student_data()
                break
            
            time.sleep(0.1)  # Control frame rate
    
    def _attendance_loop(self):
        """Main loop for attendance taking with performance optimization"""
        while self.is_running and self.current_mode == 'attendance':
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            # Performance monitoring
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:  # Update FPS every second
                self.current_fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization for better face detection
            gray = cv2.equalizeHist(gray)
            
            # Use improved face detection parameters
            faces = self.face_detect.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=6, 
                minSize=(self.face_size_threshold, self.face_size_threshold),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to RGB for web display - FIXED COLOR ISSUE
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.attendance_data = None
            
            # Process detected faces (limit to first face for performance)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Process only the first detected face
                
                # Validate face size and quality
                if w >= self.face_size_threshold and h >= self.face_size_threshold:
                    # Use grayscale for consistent processing (same as training)
                    crop_img = gray[y:y + h, x:x + w]
                    
                    # Apply preprocessing for better recognition
                    crop_img = cv2.equalizeHist(crop_img)
                    crop_img = cv2.GaussianBlur(crop_img, (3, 3), 0)
                    
                    resize_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                    
                    # Get prediction with confidence
                    distances, indices = self.knn.kneighbors(resize_img, n_neighbors=5)
                    
                    # Calculate confidence based on distance
                    avg_distance = np.mean(distances[0])
                    confidence = max(0, 1 - (avg_distance / 1000))  # Normalize distance to confidence
                    
                    # Only accept predictions above confidence threshold
                    if confidence >= self.confidence_threshold:
                        person_id = self.knn.predict(resize_img)[0]
                        person_name = self.id_to_name.get(person_id, "Unknown")
                        
                        # Add to recognition history for stability
                        self.recognition_history.append({
                            'id': person_id,
                            'name': person_name,
                            'confidence': confidence,
                            'timestamp': time.time()
                        })
                        
                        # Keep only recent history
                        if len(self.recognition_history) > self.history_size:
                            self.recognition_history.pop(0)
                        
                        # Use majority vote from recent predictions for stability
                        if len(self.recognition_history) >= 3:
                            recent_ids = [r['id'] for r in self.recognition_history[-3:]]
                            person_id = max(set(recent_ids), key=recent_ids.count)
                            person_name = self.id_to_name.get(person_id, "Unknown")
                        
                        ts = time.time()
                        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
                        
                        # Check if this person can take attendance (cooldown check)
                        can_take_attendance = True
                        if person_id in self.last_attendance_time:
                            time_since_last = ts - self.last_attendance_time[person_id]
                            if time_since_last < self.attendance_cooldown:
                                can_take_attendance = False
                        
                        # Draw bounding boxes with confidence color coding
                        if can_take_attendance:
                            if confidence >= 0.8:
                                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green for high confidence
                            else:
                                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for medium confidence
                        else:
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red for cooldown
                        
                        # Add confidence text overlay
                        cv2.putText(frame_rgb, f"{person_name} ({confidence:.2f})", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        self.attendance_data = [person_id, person_name, timestamp, f"{confidence:.2f}"]
                        
                        # Auto-capture attendance if person is ready and confidence is high
                        if can_take_attendance and person_id != "Unknown" and confidence >= 0.7:
                            self._auto_take_attendance(person_id, person_name, ts)
                            self.last_attendance_time[person_id] = ts
                    else:
                        # Low confidence - show as unknown
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (128, 128, 128), 2)  # Gray for unknown
                        cv2.putText(frame_rgb, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        self.attendance_data = None
                else:
                    # Face too small - show as unknown
                    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (128, 128, 128), 2)  # Gray for small face
                    cv2.putText(frame_rgb, "Face too small", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    self.attendance_data = None
            
            # Encode frame as JPEG with quality optimization
            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Send frame to client
            self.socketio.emit('camera_frame', {
                'frame': frame_bytes.hex(),
                'attendance_data': self.attendance_data,
                'mode': 'attendance',
                'fps': self.current_fps
            })
            
            time.sleep(0.05)  # Reduced sleep time for better responsiveness
    
    def _save_student_data(self):
        """Save collected face data to database"""
        try:
            if len(self.faces_data) == 0:
                return False
            
            with self.app.app_context():
                # Convert faces_data to numpy array and flatten each image
                faces_array = np.array(self.faces_data)
                faces_array = faces_array.reshape(len(faces_array), -1)
                
                # Create new student record
                new_student = Student(
                    student_id=self.current_student['id'],
                    name=self.current_student['name']
                )
                new_student.set_face_data(faces_array)
                
                # Add to database
                db.session.add(new_student)
                db.session.commit()
            
            # Clear cache to force reload of student data
            self.student_data_loaded = False
            self.last_data_load_time = 0
            
            # Notify client of success
            self.socketio.emit('registration_complete', {
                'success': True,
                'message': f"Student {self.current_student['name']} registered successfully!"
            })
            
            return True
        except Exception as e:
            self.socketio.emit('registration_complete', {
                'success': False,
                'message': f"Error saving student data: {str(e)}"
            })
            return False
    
    def _auto_take_attendance(self, person_id, person_name, ts):
        """Automatically take attendance for detected person"""
        try:
            current_date = datetime.fromtimestamp(ts).date()
            current_time = datetime.fromtimestamp(ts).time()
            
            with self.app.app_context():
                # Find student by ID
                student = Student.query.filter_by(student_id=person_id).first()
                if not student:
                    return False
                
                # Check if attendance already taken today
                existing_attendance = AttendanceRecord.query.filter_by(
                    student_id=student.id,
                    date=current_date
                ).first()
                
                if existing_attendance:
                    # Already taken today, don't show error for auto-capture
                    return False
                else:
                    # Create new attendance record
                    new_attendance = AttendanceRecord(
                        student_id=student.id,
                        date=current_date,
                        time=current_time
                    )
                    
                    db.session.add(new_attendance)
                    db.session.commit()
            
            # Send success notification
            self.socketio.emit('attendance_result', {
                'success': True,
                'message': f"✅ {person_name} attendance recorded automatically!",
                'auto_capture': True
            })
            return True
        except Exception as e:
            # Don't show error for auto-capture failures
            return False

    def take_attendance(self):
        """Manual attendance taking (for button click)"""
        if not self.attendance_data:
            return False
        
        try:
            person_id, person_name, timestamp = self.attendance_data
            ts = time.time()
            current_date = datetime.fromtimestamp(ts).date()
            current_time = datetime.fromtimestamp(ts).time()
            
            with self.app.app_context():
                # Find student by ID
                student = Student.query.filter_by(student_id=person_id).first()
                if not student:
                    self.socketio.emit('attendance_result', {
                        'success': False,
                        'message': f"Student with ID {person_id} not found."
                    })
                    return False
                
                # Check if attendance already taken today
                existing_attendance = AttendanceRecord.query.filter_by(
                    student_id=student.id,
                    date=current_date
                ).first()
                
                if existing_attendance:
                    self.socketio.emit('attendance_result', {
                        'success': False,
                        'message': f"{person_name} has already taken attendance today."
                    })
                    return False
                else:
                    # Create new attendance record
                    new_attendance = AttendanceRecord(
                        student_id=student.id,
                        date=current_date,
                        time=current_time
                    )
                    
                    db.session.add(new_attendance)
                    db.session.commit()
            
            self.socketio.emit('attendance_result', {
                'success': True,
                'message': f"Attendance for {person_name} recorded successfully!"
            })
            return True
        except Exception as e:
            self.socketio.emit('attendance_result', {
                'success': False,
                'message': f"Error taking attendance: {str(e)}"
            })
            return False
