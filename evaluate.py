from tensorflow.keras.models import load_model
from utils.preprocess import load_fer2013
from utils.metrics import evaluate_model
import config
import pandas as pd
import numpy as np

# Load the trained model
expression_model = load_model(config.MODEL_NAME)

# Load test data
_, test_images, _, test_labels = load_fer2013()

# Evaluate the model
evaluate_model(expression_model, test_images, test_labels)

# Generate predictions and confidence scores
predicted_probabilities = expression_model.predict(test_images)
true_label_indices = np.argmax(test_labels, axis=1)
predicted_label_indices = np.argmax(predicted_probabilities, axis=1)
prediction_confidences = np.max(predicted_probabilities, axis=1)

# Save predictions to CSV
results_df = pd.DataFrame({
    'True_Label': true_label_indices,
    'Predicted_Label': predicted_label_indices,
    'Prediction_Confidence': prediction_confidences
})
results_df.to_csv('evaluation_results.csv', index=False)
print("[INFO] Evaluation results saved to evaluation_results.csv")
