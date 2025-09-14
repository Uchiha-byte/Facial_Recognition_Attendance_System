from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import numpy as np

db = SQLAlchemy()

class Student(db.Model):
    """Student model to store student information and face data"""
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    face_data = db.Column(db.LargeBinary, nullable=False)  # Pickled numpy array
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to attendance records
    attendance_records = db.relationship('AttendanceRecord', backref='student', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Student {self.name} (ID: {self.student_id})>'
    
    def get_face_data(self):
        """Deserialize face data from database"""
        return pickle.loads(self.face_data)
    
    def set_face_data(self, face_array):
        """Serialize face data for database storage"""
        self.face_data = pickle.dumps(face_array)

class AttendanceRecord(db.Model):
    """Attendance record model"""
    __tablename__ = 'attendance_records'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    
    def __repr__(self):
        return f'<AttendanceRecord {self.student.name} at {self.timestamp}>'
