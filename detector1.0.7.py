import cv2
import face_recognition
import numpy as np
import os
import threading
import time
from queue import Queue
from datetime import datetime

# === Configuration ===
ENCODINGS_DIR = "encodings"
MIN_FACE_SIZE = 110
MAX_FACE_SIZE = 240
FRAME_QUEUE_SIZE = 1
CONFIDENCE_THRESHOLD = 60.0
ACCESS_DELAY = 5  # seconds

# === Directory setup ===
os.makedirs("successful_access", exist_ok=True)
os.makedirs("failed_access", exist_ok=True)

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
face_results = []

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

        frame_center_x = rgb_frame.shape[1] // 2
        best_face = None
        best_face_offset = float('inf')

        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            width = right - left
            height = bottom - top

            if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE or width > MAX_FACE_SIZE or height > MAX_FACE_SIZE:
                continue

            face_center_x = (left + right) // 2
            offset = abs(face_center_x - frame_center_x)

            if offset < best_face_offset:
                best_face_offset = offset
                best_face = (face_encoding, face_location)

        if best_face:
            face_encoding, face_location = best_face
            top, right, bottom, left = face_location

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.any() else None

            name = "Desconocido"
            if best_match_index is not None and matches[best_match_index]:
                confidence = (1 - face_distances[best_match_index]) * 100
                if confidence >= CONFIDENCE_THRESHOLD:
                    name = f"{known_face_names[best_match_index]}"

            local_results = [(face_location, name)]

        with results_lock:
            face_results = local_results

# === Start Worker Threads ===
for _ in range(2):
    threading.Thread(target=process_faces_worker, daemon=True).start()

# === Main Loop ===
video_capture = cv2.VideoCapture(0)

current_identity = None
timer_start = None
unknown_attempt_count = 0
flash_color = None
flash_end_time = 0

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Keep latest frame
        if frame_queue.full():
            try:
                _ = frame_queue.get_nowait()
            except:
                pass
        frame_queue.put_nowait(rgb_frame.copy())

        with results_lock:
            detected = face_results[0] if face_results else None

        if detected:
            (top, right, bottom, left), name = detected
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if name != current_identity:
                current_identity = name
                timer_start = time.time()
            else:
                elapsed = time.time() - timer_start
                countdown = max(0, int(ACCESS_DELAY - elapsed))
                cv2.putText(frame, f"Hold still: {countdown}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

                if elapsed >= ACCESS_DELAY:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if name == "Desconocido":
                        filename = f"unknown_attempt{unknown_attempt_count}_{timestamp}.jpg"
                        cv2.imwrite(os.path.join("failed_access", filename), frame)
                        unknown_attempt_count += 1
                        flash_color = (0, 0, 255)  # Red
                    else:
                        filename = f"{name}_access_{timestamp}.jpg"
                        cv2.imwrite(os.path.join("successful_access", filename), frame)
                        flash_color = (0, 255, 0)  # Green

                    flash_end_time = time.time() + 1  # Show flash for 1 second
                    current_identity = None
                    timer_start = None
                    time.sleep(1)

        else:
            current_identity = None
            timer_start = None

        # === Flash effect ===
        if flash_color and time.time() < flash_end_time:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), flash_color, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        else:
            flash_color = None

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_capture.release()
    cv2.destroyAllWindows()
    for _ in range(2):
        frame_queue.put(None)
