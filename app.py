import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Attendance System", layout="wide")

st.title("🎓 Smart Attendance System")
st.sidebar.header("Attendance Settings")

# --- INPUTS ---
subject = st.sidebar.text_input("Subject Name", "Computer Vision")
class_section = st.sidebar.text_input("Class/Section", "MCA-Final")
teacher_name = st.sidebar.text_input("Teacher Name", "Dr. Sharma")

# --- LOAD ENCODINGS ---
# We load this once so the app is fast
if 'encodeListKnown' not in st.session_state:
    st.write("🔄 Loading Student Database...")
    try:
        file = open('EncodeFile.p', 'rb')
        encodeListKnownWithIds = pickle.load(file)
        file.close()
        st.session_state['encodeListKnown'], st.session_state['studentIds'] = encodeListKnownWithIds
        st.success("Database Loaded Successfully!")
    except FileNotFoundError:
        st.error("Error: 'EncodeFile.p' not found. Please run 'encode_faces.py' first.")

# --- ATTENDANCE LOGIC ---
def mark_attendance(name, subject, section):
    filename = f"Attendance_{subject}_{datetime.now().strftime('%d-%m-%Y')}.csv"
    
    # Create file if it doesn't exist
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.writelines('Name,Class,Subject,Time,Date')

    # Read existing data
    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        # If student not present, add them
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dateString = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{section},{subject},{dtString},{dateString}')
            return filename
    return filename

# --- THE MAIN APP UI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Camera Feed")
    run_camera = st.checkbox("Start Attendance Camera")
    FRAME_WINDOW = st.image([]) # This is where the video will show

with col2:
    st.subheader("📝 Attendance Log")
    status_text = st.empty() # Placeholder for status updates

# --- CAMERA LOOP ---
if run_camera:
    cap = cv2.VideoCapture(0)
    
    while run_camera:
        success, img = cap.read()
        if not success:
            st.error("Camera not detected!")
            break

        # 1. NO RESIZE (High Quality for Detection)
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. FIND FACES (Upsample=2 for small faces)
        facesCurFrame = face_recognition.face_locations(imgS, number_of_times_to_upsample=2)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(st.session_state['encodeListKnown'], encodeFace, tolerance=0.6)
            faceDis = face_recognition.face_distance(st.session_state['encodeListKnown'], encodeFace)
            
            matchIndex = np.argmin(faceDis)
            name = "Unknown"
            color = (255, 0, 0) # Red for Unknown

            if matches[matchIndex]:
                name = st.session_state['studentIds'][matchIndex].upper()
                color = (0, 255, 0) # Green for Match
                
                # Mark Attendance
                file_saved = mark_attendance(name, subject, class_section)
                status_text.success(f"✅ Marked Present: {name}")

            # Draw Box
            y1, x2, y2, x1 = faceLoc
            # Draw rectangle on the ORIGINAL image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Display the frame in the Streamlit App
        # Convert BGR (OpenCV) to RGB (Web standard)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img_rgb)

    cap.release()

# --- SHOW DATA TABLE ---
# Check if a CSV exists for today's subject and show it
current_file = f"Attendance_{subject}_{datetime.now().strftime('%d-%m-%Y')}.csv"
if os.path.exists(current_file):
    st.subheader(f"📊 Report: {subject}")
    df = pd.read_csv(current_file)
    st.dataframe(df)