import cv2
import face_recognition
import numpy as np
import os
import threading
from queue import Queue
from datetime import datetime
import time
import paho.mqtt.client as mqtt
import ssl
import json
import uuid

import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522

# === Configuration ===
ENCODINGS_DIR = "encodings"
MIN_FACE_SIZE = 100
MAX_FACE_SIZE = 250
FRAME_QUEUE_SIZE = 1
CONFIDENCE_THRESHOLD = 50.0
ACCESS_DELAY = 3.5  # seconds
MOVEMENT_THRESHOLD = 30  # pixels of movement allowed

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

# === MQTT Client (Global) ===
with open("config.json", "r") as f:
    config = json.load(f)
    
DEVICE_ID = config["device_id"]
BUILDING_ID = config["building_id"]
DOOR_ID = config["door_id"]
MQTT_CONFIG = config["mqtt"]

client = mqtt.Client()
client.tls_set(
    ca_certs=MQTT_CONFIG["ca_cert"],
    certfile=MQTT_CONFIG["certfile"],
    keyfile=MQTT_CONFIG["keyfile"],
    tls_version=ssl.PROTOCOL_TLSv1_2
)
client.tls_insecure_set(True)
client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], 60)

def send_access_log(name, last_name, result):
    payload = {
        "logID": str(uuid.uuid4()),
        "building_id": BUILDING_ID,
        "door_id": DOOR_ID,
        "last_name": last_name,
        "name": name,
        "result": result,
        "timestamp": int(time.time())
    }
    
    client.publish("building/door/access", json.dumps(payload))

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

# === RFID Worker ===
def rfid_worker():
    reader = SimpleMFRC522()
    while True:
        try:
            tag_id, _ = reader.read()
            tag_id = str(tag_id).strip()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rfid_file = os.path.join("Rfid", f"{tag_id}.txt")

            if os.path.exists(rfid_file):
                print("Acceso garantizado (RFID)")
                send_access_log(name="RFID", last_name=tag_id, result="Success")
                with open(os.path.join("successful_access", f"{tag_id}_{timestamp}.txt"), "w") as f:
                    f.write(f"RFID: {tag_id}, Timestamp: {timestamp}")
                set_flash_color((0, 255, 0))
            else:
                print("Acceso denegado (RFID)")
                with open(os.path.join("failed_access_rfid", f"{tag_id}_{timestamp}.txt"), "w") as f:
                    f.write(f"Invalid RFID Attempt: {tag_id} at {timestamp}")
                send_access_log(name="RFID", last_name=tag_id, result="Failure")
                set_flash_color((0, 0, 255))

            time.sleep(0.5)  # avoid rapid retries

        except Exception as e:
            print("RFID read error:", e)
            time.sleep(1)

# === Flash Color Management ===
flash_color = None
flash_end_time = 0
flash_lock = threading.Lock()

def set_flash_color(color, duration=1.0):
    global flash_color, flash_end_time
    with flash_lock:
        flash_color = color
        flash_end_time = time.time() + duration

# === Start Worker Threads ===
for _ in range(2):
    threading.Thread(target=process_faces_worker, daemon=True).start()

threading.Thread(target=rfid_worker, daemon=True).start()

# === Main Camera Loop ===
video_capture = cv2.VideoCapture(0)
current_identity = None
timer_start = None
previous_face_location = None
unknown_attempt_count = 0

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

        display_frame = frame.copy()

        with results_lock:
            detected = face_results[0] if face_results else None

        if detected:
            (top, right, bottom, left), name = detected
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(display_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if name != current_identity:
                current_identity = name
                timer_start = datetime.now()
                previous_face_location = (top, right, bottom, left)
            else:
                now = datetime.now()
                elapsed = (now - timer_start).total_seconds()

                move = any(abs(n - o) > MOVEMENT_THRESHOLD for n, o in zip((top, right, bottom, left), previous_face_location))
                if move:
                    timer_start = now
                    elapsed = 0
                previous_face_location = (top, right, bottom, left)

                countdown = max(0, int(ACCESS_DELAY - elapsed))
                cv2.putText(display_frame, f"Quedese quieto: {countdown}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 100, 255), 4)

                if elapsed >= ACCESS_DELAY:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    face_image = frame[top:bottom, left:right]

                    if name == "Desconocido":
                        filename = f"unknown_attempt{unknown_attempt_count}_{timestamp}.jpg"
                        cv2.imwrite(os.path.join("failed_access", filename), face_image)
                        unknown_attempt_count += 1
                        set_flash_color((0, 0, 255))
                        print("Acceso denegado (Facial)")
                        send_access_log(name="Unknown", last_name="Unknown", result="Failure")
                    else:
                        filename = f"{name}_access_{timestamp}.jpg"
                        cv2.imwrite(os.path.join("successful_access", filename), face_image)
                        set_flash_color((0, 255, 0))
                        print("Acceso garantizado (Facial)")

                        try:
                            name_parts = name.split("_")
                            if len(name_parts) == 1:
                                first_name = name_parts[0]
                                last_name = ""
                            else:
                                first_name = " ".join(name_parts[:-1])
                                last_name = name_parts[-1]
                        except:
                            first_name = name
                            last_name = ""

                        send_access_log(name=first_name, last_name=last_name, result="Success")

                    current_identity = None
                    timer_start = None
                    previous_face_location = None

        else:
            current_identity = None
            timer_start = None
            previous_face_location = None

        # Flash
        with flash_lock:
            if flash_color and time.time() < flash_end_time:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), flash_color, -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            elif flash_color:
                flash_color = None

        cv2.imshow("Face & RFID Access System", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_capture.release()
    cv2.destroyAllWindows()
    for _ in range(2):
        frame_queue.put(None)
    GPIO.cleanup()

