
import csv
import os
import pickle
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Dataset and model paths
DATASET_FILE = "dataset/word_dataset/word_data.csv"
MODEL_FILE = "models/word_model.pkl"


print("\n===== SIGNVOX AI WORD MODEL TRAINING =====")


# Load dataset
X = []
y = []

with open(DATASET_FILE, "r", newline="") as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        if len(row) == 5671:
            y.append(row[0])
            X.append([float(value) for value in row[1:]])


X = np.array(X, dtype=np.float32)
y = np.array(y)


print("\nTotal samples:", len(X))
print("Total features per sample:", X.shape[1])
print("Word classes:", sorted(set(y)))

print("\nSamples per word:")
for word, count in sorted(Counter(y).items()):
    print(word, ":", count)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create and train model
print("\nTraining model... Please wait.")

model = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== MODEL RESULTS =====")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))


# Save model separately
os.makedirs("models", exist_ok=True)

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)


print("\nWord model saved successfully!")
print("Model location:", MODEL_FILE)

print("\n===== TRAINING COMPLETED =====")
