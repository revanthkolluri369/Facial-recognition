from flask import Flask, render_template, request, url_for, Response, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from datetime import datetime
from utils.preprocess import preprocess_face
from utils.inference_utils import generate_frames, save_current_frame
import config

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Emotion labels matching your model's output
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load pre-trained model
emotion_model = load_model(config.MODEL_NAME)


@app.route('/', methods=['GET', 'POST'])
def index():
    expression_prediction = None
    prediction_confidence = None
    uploaded_filename = None

    if request.method == 'POST':
        uploaded_file = request.files['uploaded_image']
        if uploaded_file:
            # Save uploaded file
            uploaded_filename = datetime.now().strftime('%Y%m%d_%H%M%S_') + uploaded_file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
            uploaded_file.save(file_path)

            # Read and preprocess
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                processed_image = preprocess_face(image)

                # Predict
                prediction_result = emotion_model.predict(processed_image, verbose=0)[0]
                predicted_index = np.argmax(prediction_result)
                expression_prediction = emotion_classes[predicted_index]
                prediction_confidence = int(prediction_result[predicted_index] * 100)

    return render_template('index.html',
                           expression_prediction=expression_prediction,
                           prediction_confidence=prediction_confidence,
                           uploaded_filename=uploaded_filename)


@app.route('/uploads/<uploaded_filename>')
def uploaded_file(uploaded_filename):
    """Serves uploaded images to browser"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], uploaded_filename)


@app.route('/video_feed')
def video_feed():
    """Live webcam video stream"""
    return Response(generate_frames(emotion_model, emotion_classes),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save_screenshot', methods=['POST'])
def save_screenshot():
    """Save screenshot from live webcam feed"""
    saved_filename = save_current_frame()
    return jsonify({"message": f"Screenshot saved as {saved_filename}"})


if __name__ == '__main__':
    app.run(debug=False)
