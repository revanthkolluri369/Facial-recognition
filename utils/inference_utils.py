import cv2
import numpy as np
from datetime import datetime
from utils.preprocess import preprocess_face
import config
import os

# Load Haar Cascade face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory to save captured screenshots
SCREENSHOT_OUTPUT_DIR = "screenshots"
os.makedirs(SCREENSHOT_OUTPUT_DIR, exist_ok=True)

# Global variables for the last processed frame
latest_frame = None
latest_detected_emotion = None

def generate_camera_stream(model, emotion_classes):
    global latest_frame, latest_detected_emotion
    webcam_stream = cv2.VideoCapture(0)

    while True:
        is_frame_captured, current_frame = webcam_stream.read()
        if not is_frame_captured:
            break

        grayscale_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector.detectMultiScale(grayscale_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            face_region = grayscale_frame[y:y + h, x:x + w]
            preprocessed_input = preprocess_face(face_region)

            predicted_probabilities = model.predict(preprocessed_input, verbose=0)[0]
            predicted_index = np.argmax(predicted_probabilities)
            predicted_emotion = emotion_classes[predicted_index]
            prediction_confidence = int(predicted_probabilities[predicted_index] * 100)

            # Draw bounding box and label
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(current_frame, f'{predicted_emotion} ({prediction_confidence}%)', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Store latest emotion label
            latest_detected_emotion = predicted_emotion

        latest_frame = current_frame.copy()

        _, encoded_image = cv2.imencode('.jpg', current_frame)
        byte_frame = encoded_image.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')


def save_latest_frame():
    global latest_frame, latest_detected_emotion

    if latest_frame is not None and latest_detected_emotion is not None:
        emotion_folder_path = os.path.join(SCREENSHOT_OUTPUT_DIR, latest_detected_emotion.lower())
        os.makedirs(emotion_folder_path, exist_ok=True)

        screenshot_name = f"{latest_detected_emotion.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        screenshot_path = os.path.join(emotion_folder_path, screenshot_name)
        cv2.imwrite(screenshot_path, latest_frame)
        return screenshot_name
    else:
        return "No frame available"
