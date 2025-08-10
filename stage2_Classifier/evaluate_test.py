import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
SUB_CATEGORIES = [
    'Potholes',
    'Damaged Signs',
    'Fallen Trees',
    'Graffiti',
    'Garbage',
    'Illegal Parking',
]

# Load trained model
model = load_model('final_model.h5')

# Predict
y_pred_probs = model.predict(X_test)
threshold = 0.5  # Same as used in training/app
y_pred = (y_pred_probs > threshold).astype(int)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro', zero_division=0))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro', zero_division=0))
print("F1-score (macro):", f1_score(y_test, y_pred, average='macro', zero_division=0))

# Per-class metrics
for i, class_name in enumerate(SUB_CATEGORIES):
    print(f"Class {i} ({class_name}):")
    print("  Precision:", precision_score(y_test[:, i], y_pred[:, i], zero_division=0))
    print("  Recall:", recall_score(y_test[:, i], y_pred[:, i], zero_division=0))
    print("  F1-score:", f1_score(y_test[:, i], y_pred[:, i], zero_division=0))