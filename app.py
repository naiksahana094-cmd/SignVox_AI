
from analytics import analytics_dashboard
import os
import streamlit as st
import numpy as np
from PIL import Image

from speech import text_to_speech
from text_translator import text_translator
from hand_detection import create_hand_detector, detect_hand
from prediction import predict_sign
from word_recognition import word_recognition


# ==================================
# PAGE CONFIGURATION
# ==================================

st.set_page_config(
    page_title="SignVox AI",
    page_icon="🤟",
    layout="wide"
)


# ==================================
# ALPHABET FEATURE NORMALIZATION
# ==================================

def normalize_hand(values):

    hand = np.array(values, dtype=float).reshape(21, 3)

    # Use wrist as reference point
    wrist = hand[0].copy()

    hand = hand - wrist

    # Scale coordinates
    scale = np.max(np.linalg.norm(hand, axis=1))

    if scale > 0:
        hand = hand / scale

    return hand.flatten().tolist()


# Normalize both hands
def prepare_features(landmarks_list):

    features = []

    for hand_index in range(2):

        if hand_index < len(landmarks_list):

            hand_features = normalize_hand(
                landmarks_list[hand_index]
            )

        else:

            hand_features = [0.0] * 63

        features.extend(hand_features)

    return features


# ==================================
# MAIN HEADING
# ==================================

st.title("🤟 SignVox AI")

st.subheader(
    "AI-Powered Sign Language to Text and Speech Translator"
)

st.write(
    "Breaking communication barriers through "
    "Artificial Intelligence."
)


# ==================================
# NAVIGATION
# ==================================

st.sidebar.title("Navigation")

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Sign Recognition",
        "Word Recognition",
        "Text Translator",
        "Speech Translator",
        "Analytics"
    ]
)


# ==================================
# HOME
# ==================================

if menu == "Home":

    st.header("Welcome to SignVox AI")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Technology", "Computer Vision")
    col2.metric("Input", "Webcam")
    col3.metric("Output", "Text + Speech")

    st.info(
        "Capture a hand gesture and translate it "
        "into understandable communication."
    )

    st.markdown("### Project Features")

    st.write("🤚 Hand Gesture Detection")
    st.write("🧠 AI-Based Alphabet Recognition")
    st.write("💬 Custom Word Gesture Recognition")
    st.write("📝 Text Translation")
    st.write("🔊 Speech Generation")
    st.write("📊 Performance Analytics")


# ==================================
# ALPHABET SIGN RECOGNITION
# ==================================

elif menu == "Sign Recognition":

    st.header("Live Sign Recognition")

    st.write(
        "Capture your alphabet hand gesture "
        "using the webcam."
    )

    picture = st.camera_input("Capture Hand Gesture")

    if picture is not None:

        image = Image.open(picture).convert("RGB")
        frame = np.array(image)

        detector = None

        try:

            detector = create_hand_detector()

            annotated_image, landmarks_list = detect_hand(
                frame,
                detector,
                0
            )

            st.image(
                annotated_image,
                caption="Detected Hand Landmarks",
                width="stretch"
            )

            if landmarks_list:

                # Prepare normalized features
                features = prepare_features(landmarks_list)

                # Predict alphabet
                result = predict_sign(features)

                st.success("Hand detected successfully!")

                st.markdown("### Recognized Alphabet")

                st.title(result)

                # Prediction confidence
                model = predict_sign.__globals__["model"]

                input_data = np.array(
                    features
                ).reshape(1, -1)

                probabilities = model.predict_proba(
                    input_data
                )[0]

                classes = model.classes_

                st.markdown("### Prediction Confidence")

                sorted_predictions = sorted(
                    zip(classes, probabilities),
                    key=lambda item: item[1],
                    reverse=True
                )

                for letter, probability in sorted_predictions:

                    st.write(
                        f"**{letter}**: "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(float(probability))

            else:

                st.warning(
                    "No hand detected. Please try again."
                )

        except Exception as e:

            st.error("Recognition error occurred.")
            st.exception(e)

        finally:

            if detector is not None:
                detector.close()


# ==================================
# WORD RECOGNITION
# ==================================

elif menu == "Word Recognition":

    st.header("Custom Word Gesture Recognition")

    st.write(
        "Recognize the 10 custom hand gestures "
        "using your webcam."
    )

    word_recognition()


# ==================================
# TEXT TRANSLATOR
# ==================================

elif menu == "Text Translator":

    text_translator()

    st.header("Text Translator")

    st.info(
        "Recognized gestures can be added to your translated text."
    )


# ==================================
# SPEECH TRANSLATOR
# ==================================

elif menu == "Speech Translator":

    st.header("🔊 Speech Translator")

    st.write(
        "Convert your translated text into spoken audio."
    )

    if "translated_text" not in st.session_state:
        st.session_state.translated_text = ""

    text = st.text_area(
        "Enter text to convert into speech:",
        value=st.session_state.translated_text,
        height=120
    )

    if st.button("Generate Speech"):

        if text.strip():

            try:

                with st.spinner("Generating speech..."):

                    audio_path = text_to_speech(text)

                if audio_path:

                    st.success("Speech generated successfully!")

                    with open(audio_path, "rb") as audio_file:

                        st.audio(
                            audio_file.read(),
                            format="audio/mp3"
                        )

                    os.remove(audio_path)

            except Exception as e:

                st.error(str(e))

        else:

            st.warning("Please enter some text first.")


# ==================================
# ANALYTICS
# ==================================

elif menu == "Analytics":

    analytics_dashboard()

    st.header("AI Performance Analytics")

    st.info(
        "Gesture recognition graphs will appear here."
    )
