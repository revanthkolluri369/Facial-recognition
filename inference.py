import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import config

# Emotion labels
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_model = load_model(config.MODEL_NAME)

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Screenshot output folder
SCREENSHOT_BASE_DIR = "screenshots"
os.makedirs(SCREENSHOT_BASE_DIR, exist_ok=True)

def preprocess_face_image(face_region):
    resized_face = cv2.resize(face_region, config.IMG_SIZE)
    normalized_face = resized_face.astype('float32') / 255.0
    return np.expand_dims(np.expand_dims(normalized_face, -1), axis=0)

# Initialize webcam
webcam = cv2.VideoCapture(0)

while True:
    is_frame_ready, video_frame = webcam.read()
    if not is_frame_ready:
        break

    grayscale_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(grayscale_frame, scaleFactor=1.3, minNeighbors=5)

    latest_detected_emotion = "unknown"

    for (x, y, w, h) in detected_faces:
        face_crop = grayscale_frame[y:y+h, x:x+w]
        preprocessed_input = preprocess_face_image(face_crop)

        emotion_probabilities = emotion_model.predict(preprocessed_input, verbose=0)[0]
        predicted_emotion_index = np.argmax(emotion_probabilities)
        latest_detected_emotion = emotion_classes[predicted_emotion_index]
        prediction_confidence = emotion_probabilities[predicted_emotion_index]

        # Draw bounding box around face
        cv2.rectangle(video_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display emotion label
        cv2.putText(video_frame, f"{latest_detected_emotion}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Draw confidence bar
        confidence_bar_x = x
        confidence_bar_y = y + h + 10
        confidence_bar_full_width = int(w * prediction_confidence)
        cv2.rectangle(video_frame, (confidence_bar_x, confidence_bar_y),
                      (confidence_bar_x + w, confidence_bar_y + 20), (255, 255, 255), 2)
        cv2.rectangle(video_frame, (confidence_bar_x, confidence_bar_y),
                      (confidence_bar_x + confidence_bar_full_width, confidence_bar_y + 20), (0, 255, 0), -1)
        cv2.putText(video_frame, f"{int(prediction_confidence * 100)}%",
                    (confidence_bar_x, confidence_bar_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display real-time frame
    cv2.imshow("Facial Expression Recognition - Live", video_frame)

    pressed_key = cv2.waitKey(10) & 0xFF
    if pressed_key == ord('q'):
        break
    elif pressed_key == ord('s'):
        # Save screenshot in emotion-specific subfolder
        emotion_output_dir = os.path.join(SCREENSHOT_BASE_DIR, latest_detected_emotion.lower())
        os.makedirs(emotion_output_dir, exist_ok=True)

        screenshot_filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        screenshot_path = os.path.join(emotion_output_dir, screenshot_filename)
        cv2.imwrite(screenshot_path, video_frame)
        print(f"[INFO] Screenshot saved: {screenshot_path}")

webcam.release()
cv2.destroyAllWindows()
