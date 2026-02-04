# 🎓 Smart Attendance System using Face Recognition

This is an AI-powered attendance system built for my MCA Final Semester Project. It uses **Computer Vision** to detect faces in real-time and marks attendance automatically in an Excel sheet.

## 🚀 Features
- **Real-time Face Detection**: Uses `face_recognition` library (99% accuracy).
- **Web Interface**: Built with **Streamlit** for a modern user experience.
- **Anti-Spoofing**: High-sensitivity mode (`upsample=2`) detects faces even at a distance.
- **Automatic Logging**: Saves attendance data to a CSV file (Excel compatible) with Time & Date.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Libraries**: OpenCV, Face_Recognition, Streamlit, Pandas, Numpy
- **Tools**: VS Code, Git/GitHub

## 📸 How It Works
1. The system loads known student faces from the `student_faces/` folder.
2. The webcam captures video in real-time.
3. If a match is found, the student's name is displayed in a **Green Box**.
4. Attendance is instantly marked in the daily CSV report.

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/luence09/Smart_Attendance_System.git](https://github.com/luence09/Smart_Attendance_System.git)
