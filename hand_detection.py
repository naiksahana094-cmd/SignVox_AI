
import cv2
import mediapipe as mp

# Model file location
MODEL_PATH = "models/hand_landmarker.task"


def create_hand_detector():

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )


def detect_hand(image_rgb, detector, timestamp_ms):

    # Convert image into MediaPipe format
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    # Detect hand landmarks
    results = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    annotated_image = image_rgb.copy()
    landmarks_list = []

    if results.hand_landmarks:

        for hand_landmarks in results.hand_landmarks:

            height, width, _ = annotated_image.shape

            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                # Draw landmark points
                cv2.circle(
                    annotated_image,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections between landmarks
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

            for connection in connections:
                start = connection.start
                end = connection.end

                cv2.line(
                    annotated_image,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

            # Extract 63 coordinates
            landmarks = []

            for point in hand_landmarks:
                landmarks.extend([
                    point.x,
                    point.y,
                    point.z
                ])

            landmarks_list.append(landmarks)

    return annotated_image, landmarks_list
