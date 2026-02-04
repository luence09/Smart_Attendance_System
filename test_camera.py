import cv2
import os
import pickle
import face_recognition
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
fileName = "Attendance.csv"
TOLERANCE = 0.6  # Lower = Stricter, Higher = More Lenient (0.6 is standard)
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

    # 1. Resize to 0.5 (Half size) for speed, but better than 0.25
    imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # 2. Find faces
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=TOLERANCE)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        # Log the difference (0.0 is exact match, 1.0 is total stranger)
        best_match_score = faceDis[0] if len(faceDis) > 0 else 1.0
        print(f"Difference: {best_match_score:.2f} (Tolerance is {TOLERANCE})")

        matchIndex = np.argmin(faceDis)

        name = "Unknown"
        color = (0, 0, 255) # Red for Unknown

        # If the best match is within our tolerance
        if matches[matchIndex]:
            name = studentIds[matchIndex].upper()
            color = (0, 255, 0) # Green for Match
            markAttendance(name)
        
        # 3. Scale coordinates back up (x2 because we resized by 0.5)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()