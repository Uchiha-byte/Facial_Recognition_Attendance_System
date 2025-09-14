from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import subprocess
import threading
import os
import csv
import pickle
import numpy as np
from models import db, Student, AttendanceRecord
from datetime import datetime, date, time
import sys
from camera_service import CameraService

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")
camera_service = CameraService(socketio, app)

def run_add_faces(name, user_id):
    python_path = sys.executable
    subprocess.run([python_path, 'add_faces_simple.py', '--name', name, '--id', user_id])

def run_test():
    python_path = sys.executable
    subprocess.run([python_path, 'test_simple.py'])
@app.route('/')
def home():
    students_count = Student.query.count()
    return render_template('home.html', students_count=students_count)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        user_id = request.form['id']
        
        existing_student = Student.query.filter_by(student_id=user_id).first()
        if existing_student:
            return render_template('register.html', error=f"ID {user_id} already exists. Please choose a different ID.")
        
        if camera_service.start_register_mode(name, user_id):
            return render_template('register_camera.html', name=name, user_id=user_id)
        else:
            return render_template('register.html', error="Could not initialize camera. Please check if camera is connected.")
    
    return render_template('register.html')

@app.route('/take_attendance')
def take_attendance():
    students_count = Student.query.count()
    
    if students_count == 0:
        return render_template('home.html', error="No students registered. Please register students first.")
    
    if camera_service.start_attendance_mode():
        return render_template('attendance_camera.html')
    else:
        return render_template('home.html', error="Could not initialize camera or error loading student data.")

@app.route('/attendance')
def view_attendance():
    dates = db.session.query(AttendanceRecord.date).distinct().order_by(AttendanceRecord.date.desc()).all()
    attendance_files = [f"Attendance_{date[0].strftime('%d-%m-%Y')}.csv" for date in dates]
    return render_template('attendance.html', files=attendance_files)

@app.route('/attendance/<filename>')
def view_attendance_file(filename):
    try:
        date_str = filename.replace('Attendance_', '').replace('.csv', '')
        file_date = datetime.strptime(date_str, '%d-%m-%Y').date()
    except ValueError:
        return "Invalid filename format", 400
    
    records = db.session.query(AttendanceRecord, Student).join(Student).filter(
        AttendanceRecord.date == file_date
    ).order_by(AttendanceRecord.timestamp).all()
    
    headers = ["ID", "NAME", "TIME"]
    data = []
    for record, student in records:
        data.append([student.student_id, student.name, record.time.strftime('%H-%M-%S')])
    
    return render_template('attendance_file.html', 
                         filename=filename,
                         headers=headers,
                         records=data)

@app.route('/students')
def list_students():
    students = Student.query.all()
    students_data = [{"id": student.student_id, "name": student.name} for student in students]
    return render_template('students.html', students=students_data)

@app.route('/remove_student', methods=['POST'])
def remove_student():
    user_id = request.form['id']
    
    student = Student.query.filter_by(student_id=user_id).first()
    if student:
        db.session.delete(student)
        db.session.commit()
        camera_service.student_data_loaded = False
        camera_service.last_data_load_time = 0
    
    return redirect(url_for('list_students'))

@app.route('/download_attendance/<filename>')
def download_attendance(filename):
    try:
        date_str = filename.replace('Attendance_', '').replace('.csv', '')
        file_date = datetime.strptime(date_str, '%d-%m-%Y').date()
    except ValueError:
        return "Invalid filename format", 400
    
    records = db.session.query(AttendanceRecord, Student).join(Student).filter(
        AttendanceRecord.date == file_date
    ).order_by(AttendanceRecord.timestamp).all()
    
    csv_content = "ID,NAME,TIME\n"
    for record, student in records:
        csv_content += f"{student.student_id},{student.name},{record.time.strftime('%H:%M:%S')}\n"
    
    from flask import Response
    return Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )

@app.route('/delete_attendance/<filename>', methods=['POST'])
def delete_attendance(filename):
    try:
        date_str = filename.replace('Attendance_', '').replace('.csv', '')
        file_date = datetime.strptime(date_str, '%d-%m-%Y').date()
    except ValueError:
        return "Invalid filename format", 400
    
    AttendanceRecord.query.filter_by(date=file_date).delete()
    db.session.commit()
    
    return redirect(url_for('view_attendance'))

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    camera_service.stop_camera()

@socketio.on('take_attendance')
def handle_take_attendance():
    camera_service.take_attendance()

@socketio.on('stop_camera')
def handle_stop_camera():
    camera_service.stop_camera()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)