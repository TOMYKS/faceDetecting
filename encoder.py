import os
import face_recognition
import numpy as np

# Input folder: original face images
KNOWN_FACES_DIR = "faces"

# Output folder: individual .npy files for each encoding
ENCODINGS_DIR = "encodings"
os.makedirs(ENCODINGS_DIR, exist_ok=True)

# Process each image
for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)

    # Load and encode
    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        encoding = encodings[0]  # Use first face found
        name = os.path.splitext(filename)[0]  # Remove .jpg/.png extension
        npy_path = os.path.join(ENCODINGS_DIR, f"{name}.npy")
        np.save(npy_path, encoding)
        print(f"[✓] Saved encoding for {name}")
    else:
        print(f"[!] No face found in {filename}, skipping.")
