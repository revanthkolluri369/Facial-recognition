from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def evaluate_expression_model(expression_model, input_images, true_labels_onehot):
    # Generate predicted probabilities and convert to class labels
    predicted_probabilities = expression_model.predict(input_images, verbose=0)
    predicted_labels = np.argmax(predicted_probabilities, axis=1)
    true_labels = np.argmax(true_labels_onehot, axis=1)

    print("📋 Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    print("🧮 Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))
