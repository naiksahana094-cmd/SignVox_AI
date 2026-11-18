
import cv2
from hand_detection import create_hand_detector, detect_hand

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Webcam opened successfully!")
print("Show your hand in front of the camera.")
print("Press Q to close.")

timestamp_ms = 0

try:
    # Create detector only once
    detector = create_hand_detector()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot read webcam")
            break

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        timestamp_ms += 33

        # Detect hand
        result_image, landmarks = detect_hand(
            image_rgb,
            detector,
            timestamp_ms
        )

        # Display detection result
        if landmarks:
            cv2.putText(
                result_image,
                "HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            print("Hand detected! Landmarks:", len(landmarks[0]))

        else:
            print("No hand detected")

        # Display webcam
        cv2.imshow(
            "SignVox AI - Hand Detection",
            cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    if "detector" in locals():
        detector.close()

    print("Camera closed.")
