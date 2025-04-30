from models.model import build_best_frr_cnn
from utils.preprocess import load_fer2013
import config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load preprocessed FER-2013 data
train_generator, validation_generator = load_fer2013()
class_indices = list(train_generator.class_indices.values())
label_array = train_generator.classes
class_weight_map = dict(zip(class_indices, compute_class_weight(class_weight='balanced', classes=np.array(class_indices), y=label_array)))

# Build enhanced FRR-CNN model with trainable VGG16 layers
print("🚀 Training Enhanced FRR-CNN with all VGG16 layers trainable...")
fer_model = build_best_frr_cnn(input_shape=(48, 48, 1), num_classes=config.NUM_CLASSES)

training_callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
]

# Train the model
training_history = fer_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=config.EPOCHS,
    class_weight=class_weight_map,
    callbacks=training_callbacks
)

# Save the trained model
fer_model.save(config.MODEL_NAME)
print(f"✅ Final model saved to: {config.MODEL_NAME}")

# Evaluate model performance
train_loss, train_accuracy = fer_model.evaluate(train_generator)
val_loss, val_accuracy = fer_model.evaluate(validation_generator)
print(f"📊 Train Accuracy: {train_accuracy:.2%}, Validation Accuracy: {val_accuracy:.2%}")

# Classification Metrics
true_labels = validation_generator.classes
predicted_probabilities = fer_model.predict(validation_generator)
predicted_labels = np.argmax(predicted_probabilities, axis=1)

print("\n📋 Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=list(train_generator.class_indices.keys())))

# Confusion Matrix Plot
confusion_mat = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(train_generator.class_indices.keys()),
            yticklabels=list(train_generator.class_indices.keys()))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.show()

# Accuracy Plot
plt.plot(training_history.history['accuracy'], label='Train Accuracy')
plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.show()

# ROC Curve Plot
binarized_true_labels = label_binarize(true_labels, classes=list(range(config.NUM_CLASSES)))
plt.figure(figsize=(10, 7))
for index, emotion_label in enumerate(train_generator.class_indices.keys()):
    fpr, tpr, _ = roc_curve(binarized_true_labels[:, index], predicted_probabilities[:, index])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{emotion_label} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve.png")
plt.show()
