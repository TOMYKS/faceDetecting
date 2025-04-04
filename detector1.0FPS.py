import cv2
import face_recognition
import numpy as np
import os
import time

# Folder where known faces are stored
KNOWN_FACES_DIR = "faces"

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    img_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:  # Ensure face is detected
        known_face_encodings.append(encoding[0])
        known_face_names.append(os.path.splitext(filename)[0])  # Remove file extension

# Initialize camera
video_capture = cv2.VideoCapture(0)

# Variables to calculate FPS
fps_start_time = time.time()
fps = 0
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count >= 10:  # Update FPS every 10 frames
        fps = frame_count / (time.time() - fps_start_time)
        fps_start_time = time.time()
        frame_count = 0

    # Convert frame to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.any() else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display FPS in the top-left corner
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the video stream
    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
