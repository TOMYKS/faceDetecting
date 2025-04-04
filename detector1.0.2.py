import cv2
import face_recognition
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor

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
        known_face_names.append(os.path.splitext(filename)[0])

# Shared data structure to hold results
results_lock = threading.Lock()
face_results = []  # Will hold tuples: (top, right, bottom, left, name)

# Face size thresholds
MIN_FACE_SIZE = 120
MAX_FACE_SIZE = 200

def process_faces(rgb_frame):
    global face_results
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    local_results = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        top, right, bottom, left = face_location
        width = right - left
        height = bottom - top

        # Filter by size
        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE or width > MAX_FACE_SIZE or height > MAX_FACE_SIZE:
            continue

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"
        confidence = 0

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.any() else None

        if best_match_index is not None and matches[best_match_index]:
            confidence = (1 - face_distances[best_match_index]) * 100
            if confidence >= 60:
                name = f"{known_face_names[best_match_index]} ({confidence:.2f}%)"

        local_results.append((face_location, name))

    with results_lock:
        face_results = local_results

# Initialize camera
video_capture = cv2.VideoCapture(0)
executor = ThreadPoolExecutor(max_workers=1)

# Frame skipping logic
frame_count = 0
PROCESS_EVERY_N_FRAMES = 5
processing_thread = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Every N frames, launch recognition in a thread
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        if processing_thread is None or not processing_thread.running():
            processing_thread = executor.submit(process_faces, rgb_frame)

    frame_count += 1

    # Draw last known face results
    with results_lock:
        for (top, right, bottom, left), name in face_results:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
executor.shutdown()
