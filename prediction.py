
import pickle
import numpy as np

MODEL_FILE = "models/isl_model.pkl"

# Load the trained model
with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


def predict_sign(landmarks):
    """
    Predict an ISL alphabet letter from hand landmarks.
    """

    if len(landmarks) != 126:
        return "Invalid input"

    landmarks = np.array(landmarks).reshape(1, -1)

    prediction = model.predict(landmarks)

    return prediction[0]
print("\nModel prediction classes:")
print(model.classes_)

print("\nModel feature count:")
print(model.n_features_in_)