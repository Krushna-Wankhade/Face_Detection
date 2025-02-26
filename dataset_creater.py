import cv2
import mysql.connector
import os
import numpy as np

# Load Haar Cascade for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

# Connect to MySQL Database (FIXED: No arguments needed)
def connect_db():
    return mysql.connector.connect(
        host="localhost",  # Change if your MySQL server is remote
        user="root",  # Replace with your MySQL username
        password="12345678@aB",  # Replace with your MySQL password
        database="students"  # Replace with your database name
    )

# Function to Insert or Update Student Data
def insertOrUpdate(Id, Name, Age, Gen):
    conn = connect_db()  # FIXED: No argument passed
    cursor = conn.cursor()

    # Check if the ID already exists
    cursor.execute("SELECT * FROM students WHERE ID = %s", (Id,))
    isRecordExist = cursor.fetchone()

    if isRecordExist:
        cursor.execute("UPDATE students SET Name = %s, Age = %s, Gen = %s WHERE ID = %s", (Name, Age, Gen, Id))
    else:
        cursor.execute("INSERT INTO students (ID, Name, Age, Gen) VALUES (%s, %s, %s, %s)", (Id, Name, Age, Gen))

    conn.commit()
    cursor.close()
    conn.close()

# User Inputs
Id = input('Enter User Id: ')
Name = input('Enter User Name: ')
Age = input('Enter User Age: ')
Gen = input('Enter User Gender: ')

# Insert or update data in MySQL
insertOrUpdate(Id, Name, Age, Gen)

# Capturing Face Samples
sampleNum = 0
dataset_dir = "dataSet"
os.makedirs(dataset_dir, exist_ok=True)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(f"{dataset_dir}/User.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(400)

    cv2.imshow("Face", img)
    cv2.waitKey(1)

    if sampleNum >= 20:
        break

cam.release()
cv2.destroyAllWindows()
