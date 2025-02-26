import cv2
import numpy as np
import mysql.connector
from datetime import datetime

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")  # Load trained model

# Connect to MySQL and fetch user details
def get_user_details():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Change to your MySQL username
        password="12345678@aB",  # Change to your MySQL password
        database="students"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT ID, Name FROM students")
    user_data = {str(row[0]): row[1] for row in cursor.fetchall()}  # Create dictionary {ID: Name}
    conn.close()
    return user_data

# Function to mark attendance
def mark_attendance(user_id, user_name):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Change to your MySQL username
        password="12345678@aB",  # Change to your MySQL password
        database="attendance"  # New database for attendance
    )
    cursor = conn.cursor()

    # Get current date and time
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Check if user has already been marked present today
    cursor.execute("SELECT * FROM attendance WHERE ID = %s AND Date = %s", (user_id, current_date))
    result = cursor.fetchone()

    if result is None:
        cursor.execute("INSERT INTO attendance (ID, Name, Date, Time) VALUES (%s, %s, %s, %s)",
                       (user_id, user_name, current_date, current_time))
        conn.commit()
        print(f"Attendance marked for {user_name} ({user_id}) on {current_date} at {current_time}")

    cursor.close()
    conn.close()

# Get user data from database
user_details = get_user_details()

# Start video capture
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        Id, confidence = recognizer.predict(roi_gray)  # Recognize face

        if confidence < 50:
            name = user_details.get(str(Id), "Unknown")  # Fetch name from database
            text = f"ID: {Id} | Name: {name}"

            # Mark attendance in the new database
            if name != "Unknown":
                mark_attendance(Id, name)
        else:
            text = "Unknown"

        # Display ID and Name
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition & Attendance", img)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cam.release()
cv2.destroyAllWindows()
