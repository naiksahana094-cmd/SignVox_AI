
import cv2
import av
import streamlit as st

from threading import Lock

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from hand_detection import create_hand_detector, detect_hand
from word_prediction import predict_word, get_word_confidence


FRAMES_PER_SAMPLE = 45
FEATURES_PER_FRAME = 126


def get_frame_features(landmarks_list):
    features = []

    for hand_index in range(2):
        if hand_index < len(landmarks_list):
            hand_data = landmarks_list[hand_index]
            features.extend(hand_data[:63])

            if len(hand_data) < 63:
                features.extend([0.0] * (63 - len(hand_data)))
        else:
            features.extend([0.0] * 63)

    return features


class WordVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.detector = create_hand_detector()

        self.sequence = []
        self.timestamp = 0

        # Keep the last recognized word on screen
        self.current_word = "Show a hand gesture"
        self.confidence = 0.0

        self.lock = Lock()

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        annotated_image, landmarks_list = detect_hand(
            image_rgb,
            self.detector,
            self.timestamp
        )

        self.timestamp += 33

        # Read the latest saved prediction
        with self.lock:
            displayed_word = self.current_word
            displayed_confidence = self.confidence

        # Collect frames only when a hand is detected
        if landmarks_list:

            features = get_frame_features(landmarks_list)
            self.sequence.extend(features)

        else:
            # Reset incomplete sequence if hand disappears
            self.sequence = []

        frames_collected = len(self.sequence) // FEATURES_PER_FRAME

        # Predict after collecting 45 frames
        if len(self.sequence) >= FRAMES_PER_SAMPLE * FEATURES_PER_FRAME:

            print("\n45 FRAMES COLLECTED")
            print("STARTING WORD PREDICTION")

            try:
                prediction = predict_word(self.sequence)
                confidence = get_word_confidence(self.sequence)

                with self.lock:
                    self.current_word = prediction
                    self.confidence = confidence

                    displayed_word = self.current_word
                    displayed_confidence = self.confidence

                print("PREDICTED WORD:", prediction)
                print("CONFIDENCE:", round(confidence * 100, 2), "%")
                print("WORD BEING DISPLAYED:", displayed_word)

            except Exception as error:
                print("PREDICTION ERROR:", error)

            finally:
                self.sequence = []

            frames_collected = 0

        # Draw information on webcam video
        cv2.putText(
            annotated_image,
            "SIGNVOX AI - WORD RECOGNITION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if landmarks_list:
            hand_status = "Hand detected"
        else:
            hand_status = "No hand detected"

        cv2.putText(
            annotated_image,
            hand_status,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_image,
            f"Frames: {frames_collected}/{FRAMES_PER_SAMPLE}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # Display the last recognized word
        cv2.putText(
            annotated_image,
            f"WORD: {displayed_word}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if displayed_confidence > 0:
            cv2.putText(
                annotated_image,
                f"CONFIDENCE: {displayed_confidence * 100:.1f}%",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(
            annotated_image,
            format="rgb24"
        )


def word_recognition():

    st.title("SignVox AI - Word Recognition")

    st.write(
        "Perform one of your 10 collected custom gestures "
        "in front of the webcam."
    )

    st.info(
        "Keep your hand visible and perform the gesture "
        "consistently for the camera."
    )

    ctx = webrtc_streamer(
        key="word-recognition",
        video_processor_factory=WordVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=False
    )

    if ctx.video_processor:

        processor = ctx.video_processor

        with processor.lock:
            recognized_word = processor.current_word
            confidence = processor.confidence

        st.subheader("Latest Recognition")

        st.write("Recognized word:", recognized_word)

        if confidence > 0:
            st.write(
                "Model confidence:",
                f"{confidence * 100:.2f}%"
            )

        if recognized_word not in [
            "Waiting for gesture...",
            "Show a hand gesture"
        ]:

            if st.button("Add recognized word to Text Translator"):

                existing_text = st.session_state.get(
                    "translated_text",
                    ""
                )

                if existing_text:
                    st.session_state.translated_text = (
                        existing_text + " " + recognized_word
                    )
                else:
                    st.session_state.translated_text = recognized_word

                st.success(
                    f"{recognized_word} added to Text Translator!"
                )

    else:
        st.warning(
            "Start the webcam using the START button above."
        )
