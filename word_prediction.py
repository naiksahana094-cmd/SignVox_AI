
import pickle
import numpy as np

MODEL_FILE = "models/word_model.pkl"

# Load trained word recognition model
with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


def predict_word(sequence_features):

    # Check whether the input contains 45 frames
    # with 2 hands and 21 landmarks per hand
    if len(sequence_features) != 5670:
        return "Invalid input"

    # Convert features into the correct format
    features = np.array(
        sequence_features,
        dtype=np.float32
    ).reshape(1, -1)

    # Predict the word
    prediction = model.predict(features)

    return prediction[0]


def get_word_confidence(sequence_features):

    if len(sequence_features) != 5670:
        return 0.0

    features = np.array(
        sequence_features,
        dtype=np.float32
    ).reshape(1, -1)

    probabilities = model.predict_proba(features)

    return float(np.max(probabilities))


if __name__ == "__main__":

    print("SignVox AI Word Model Loaded Successfully!")

    print("\nRecognized word classes:")

    for word in model.classes_:
        print("-", word)
