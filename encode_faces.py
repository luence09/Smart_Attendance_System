import cv2
import face_recognition
import os
import pickle

# --- CONFIGURATION ---
folderPath = 'student_faces'  # Make sure this matches your folder name
# ---------------------

print("Step 1: Loading Student Images...")

# Get list of all images in the folder
pathList = os.listdir(folderPath)
imgList = []
studentIds = []

print(f"Found {len(pathList)} images: {pathList}")

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    # Start splitting the filename to get the name (e.g., "Rahul.jpg" -> "Rahul")
    studentIds.append(os.path.splitext(path)[0])

print("Step 2: Encoding Images (This might take a moment)...")

def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert BGR (OpenCV standard) to RGB (Face Recognition standard)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            # Find the face encoding
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print("Error: Could not find a face in one of the images. Skipping it.")
    return encodeList

# Generate the encodings
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]

print("Step 3: Saving Encodings to File...")

# Save the data to a file so we don't have to process images every time
file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()

print("---------------------------------------")
print("SUCCESS: 'EncodeFile.p' has been created!")
print("You are now ready to run the webcam.")
print("---------------------------------------")