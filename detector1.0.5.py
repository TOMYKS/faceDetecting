import cv2
import face_recognition
import numpy as np
import os
import threading
from queue import Queue

# === Configuration ===
ENCODINGS_DIR = "encodings"
MIN_FACE_SIZE = 110
MAX_FACE_SIZE = 240
FRAME_QUEUE_SIZE = 1  # Only keep the latest frame
CONFIDENCE_THRESHOLD = 60.0

# === Load Known Encodings ===
known_face_encodings = []
known_face_names = []

for filename in os.listdir(ENCODINGS_DIR):
    if filename.endswith(".npy"):
        name = os.path.splitext(filename)[0]
        encoding = np.load(os.path.join(ENCODINGS_DIR, filename))
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# === Shared Resources ===
frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
results_lock = threading.Lock()
face_results = []  # Will hold tuples: (top, right, bottom, left, name)

# === Face Recognition Worker ===
def process_faces_worker():
    global face_results
    while True:
        rgb_frame = frame_queue.get()
        if rgb_frame is None:
            break

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        local_results = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            width = right - left
            height = bottom - top

            if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE or width > MAX_FACE_SIZE or height > MAX_FACE_SIZE:
                continue

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.any() else None

            name = "Desconocido"
            if best_match_index is not None and matches[best_match_index]:
                confidence = (1 - face_distances[best_match_index]) * 100
                if confidence >= CONFIDENCE_THRESHOLD:
                    name = f"{known_face_names[best_match_index]} ({confidence:.2f}%)"

            local_results.append((face_location, name))

        with results_lock:
            face_results = local_results

# === Start Worker Threads ===
for _ in range(2):
    threading.Thread(target=process_faces_worker, daemon=True).start()

# === Main Camera Loop ===
video_capture = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Always push latest frame to queue (discard previous if needed)
        if frame_queue.full():
            try:
                _ = frame_queue.get_nowait()
            except:
                pass
        frame_queue.put_nowait(rgb_frame.copy())

        # Draw last recognized face results
        with results_lock:
            for (top, right, bottom, left), name in face_results:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_capture.release()
    cv2.destroyAllWindows()
    for _ in range(2):
        frame_queue.put(None)  # Signal workers to exit
