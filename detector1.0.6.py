import cv2
import face_recognition
import numpy as np
import os
import threading
import time
from queue import Queue

# === Configuration ===
ENCODINGS_DIR = "encodings"
SUCCESSFUL_DIR = "successful_access"
FAILED_DIR = "failed_access"
MIN_FACE_SIZE = 110
MAX_FACE_SIZE = 240
FRAME_QUEUE_SIZE = 1  # Only keep the latest frame
CONFIDENCE_THRESHOLD = 60.0

os.makedirs(SUCCESSFUL_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)

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
                    name = known_face_names[best_match_index]

            local_results.append((face_location, name))

        # Choose the most centered face only
        if local_results:
            frame_center = (rgb_frame.shape[1] // 2, rgb_frame.shape[0] // 2)
            local_results.sort(key=lambda r: abs(((r[0][1]+r[0][3])//2) - frame_center[0]))
            face_results = [local_results[0]]
        else:
            face_results = []

# === Start Worker Threads ===
for _ in range(2):
    threading.Thread(target=process_faces_worker, daemon=True).start()

video_capture = cv2.VideoCapture(0)
failed_attempts = 0
captured = False

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_queue.full():
            try:
                _ = frame_queue.get_nowait()
            except:
                pass
        frame_queue.put_nowait(rgb_frame.copy())

        with results_lock:
            if face_results:
                (top, right, bottom, left), name = face_results[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if not captured:
                    cv2.putText(frame, "Stand still... Capturing in 3", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Face Recognition", frame)
                    cv2.waitKey(1000)
                    frame = video_capture.read()[1]
                    cv2.putText(frame, "2", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Face Recognition", frame)
                    cv2.waitKey(1000)
                    frame = video_capture.read()[1]
                    cv2.putText(frame, "1", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Face Recognition", frame)
                    cv2.waitKey(1000)

                    screenshot = video_capture.read()[1]
                    if name != "Desconocido":
                        filename = f"{name}_access.jpg"
                        cv2.imwrite(os.path.join(SUCCESSFUL_DIR, filename), screenshot)
                    else:
                        filename = f"unknown_attempt{failed_attempts}.jpg"
                        cv2.imwrite(os.path.join(FAILED_DIR, filename), screenshot)
                        failed_attempts += 1
                    captured = True
            else:
                captured = False

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_capture.release()
    cv2.destroyAllWindows()
    for _ in range(2):
        frame_queue.put(None)