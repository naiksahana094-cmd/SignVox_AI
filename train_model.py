
import csv
import os
import pickle
import numpy as np

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET_FILE = "dataset/isl_data.csv"
NORMALIZED_FILE = "dataset/isl_data_normalized.csv"
MODEL_FILE = "models/isl_model.pkl"


# Normalize each hand separately using wrist position
# and maximum distance from the wrist.
def normalize_hand(values):

    hand = np.array(values, dtype=float).reshape(21, 3)

    # Wrist is landmark 0
    wrist = hand[0].copy()

    hand = hand - wrist

    scale = np.max(np.linalg.norm(hand, axis=1))

    if scale > 0:
        hand = hand / scale

    return hand.flatten().tolist()


# Read original dataset
X = []
y = []

with open(DATASET_FILE, "r", newline="") as file:

    reader = csv.reader(file)
    headings = next(reader)

    for row in reader:

        if len(row) != 127:
            continue

        label = row[0]

        features = [float(value) for value in row[1:]]

        # Normalize first hand
        first_hand = normalize_hand(features[:63])

        # Normalize second hand
        second_hand = normalize_hand(features[63:126])

        normalized_features = first_hand + second_hand

        X.append(normalized_features)
        y.append(label)


# Save normalized dataset
os.makedirs("dataset", exist_ok=True)

with open(NORMALIZED_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(headings)

    for label, features in zip(y, X):
        writer.writerow([label] + features)


print("Normalized dataset created!")
print("Total samples:", len(X))
print("Total features:", len(X[0]))
print("Letters:", sorted(set(y)))


print("\nSamples per letter:")

for letter, count in sorted(Counter(y).items()):
    print(letter, ":", count)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
print("\nTraining normalized Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# Save model
os.makedirs("models", exist_ok=True)

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)

print("\nNew normalized model saved successfully!")
