import cv2
import os
import pickle
import face_recognition
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
fileName = "Attendance.csv"
TOLERANCE = 0.6
# ---------------------

print("Loading Encoded Data...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Data Loaded Successfully.")

def markAttendance(name):
    if not os.path.isfile(fileName):
        with open(fileName, 'w') as f:
            f.writelines('Name,Time,Date')

    with open(fileName, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dateString = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{dtString},{dateString}')
            print(f"✅ ATTENDANCE MARKED: {name}")

cap = cv2.VideoCapture(0)
print("Starting Camera... Press 'q' to Exit.")

while True:
    success, img = cap.read()
    if not success:
        print("Camera Error")
        break

    # 1. NO RESIZING (Use full quality)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. HIGH SENSITIVITY MODE (upsample=2 finds smaller faces)
    # This might be slightly laggy, but it WILL find you.
    facesCurFrame = face_recognition.face_locations(imgS, number_of_times_to_upsample=2)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # If no faces found, print it (so we know)
    if len(facesCurFrame) == 0:
        print("Scanning... No faces detected.")

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=TOLERANCE)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(faceDis)
        name = "Unknown"
        color = (0, 0, 255) # Red

        if matches[matchIndex]:
            name = studentIds[matchIndex].upper()
            color = (0, 255, 0) # Green
            markAttendance(name)
            print(f"FOUND: {name}")
        
        # 3. Draw Box (No math needed because we didn't resize)
        y1, x2, y2, x1 = faceLoc
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()