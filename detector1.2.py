import cv2
import face_recognition
import numpy as np
import os
import threading

# Folder where known faces are stored
KNOWN_FACES_DIR = "faces"
TOLERANCE = 0.5  # Lower = stricter matching
RESIZE_SCALE = 0.5  # Reduce resolution for speed

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    img_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:  # Ensure a face is detected
        known_face_encodings.append(encoding[0])
        known_face_names.append(os.path.splitext(filename)[0])  # Remove file extension

# Initialize camera
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 30)  # Increase FPS for smoother video
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

scanning = False
processing = False  # Prevent multiple threads at once

print("Press 'S' to start scanning, 'Q' to quit.")

def recognize_faces(frame):
    """Detect and recognize faces asynchronously."""
    global processing

    processing = True  # Lock processing
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Try detecting faces using 'hog' (faster) or 'cnn' (more accurate)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        name = "Unknown"
        confidence = 0

        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]

            if confidence >= TOLERANCE:
                name = known_face_names[best_match_index]

        # Scale back face location to original frame size
        top, right, bottom, left = [int(coord / RESIZE_SCALE) for coord in face_location]

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # If no faces were found, display "No Face Detected"
    if not face_locations:
        cv2.putText(frame, "No Face Detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    processing = False  # Unlock processing
    cv2.imshow("Face Recognition", frame)


while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    cv2.imshow("Face Recognition", frame)  # Always display live feed

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Toggle scanning
        scanning = not scanning
        print("Scanning started..." if scanning else "Scanning stopped.")

    if scanning and not processing:  # Process only when scanning is active
        threading.Thread(target=recognize_faces, args=(frame.copy(),), daemon=True).start()

    if key == ord('q'):  # Quit program
        print("Exiting...")
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
