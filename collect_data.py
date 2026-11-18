
import cv2
import csv
import os
import time
from hand_detection import create_hand_detector, detect_hand

# CSV file location
CSV_FILE = "dataset/isl_data.csv"

# Create dataset folder if it does not exist
os.makedirs("dataset", exist_ok=True)

# Create CSV headings
columns = ["label"]

for hand in range(1, 3):
    for point in range(21):
        columns.extend([
            f"hand{hand}_x{point}",
            f"hand{hand}_y{point}",
            f"hand{hand}_z{point}"
        ])

# Create CSV file with headings if it does not exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

# Ask which ISL sign we are collecting
label = input("Enter the ISL sign label (example: A): ").strip().upper()

if not label:
    print("Please enter a valid label.")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("\nWebcam opened!")
print("Show the sign:", label)
print("Press S to save one sample.")
print("Press Q to quit.")

timestamp_ms = 0
saved_count = 0

detector = create_hand_detector()

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read webcam.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms += 33

        result_image, landmarks = detect_hand(
            image_rgb,
            detector,
            timestamp_ms
        )

        # Display current label and instructions
        cv2.putText(
            result_image,
            f"Sign: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result_image,
            f"Saved samples: {saved_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result_image,
            "Press S to save | Q to quit",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISL Data Collection",
            cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        )

        key = cv2.waitKey(1) & 0xFF

        # Save sample when S is pressed
        if key == ord("s"):

            if not landmarks:
                print("No hand detected. Sample not saved.")
                continue

            # Prepare space for two hands (126 coordinates)
            sample = []

            for hand_index in range(2):
                if hand_index < len(landmarks):
                    sample.extend(landmarks[hand_index])
                else:
                    sample.extend([0.0] * 63)

            with open(CSV_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([label] + sample)

            saved_count += 1
            print(f"Saved sample {saved_count} for sign {label}")

        # Quit when Q is pressed
        elif key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

print(f"\nCollection finished. Samples saved: {saved_count}")
print("CSV file:", CSV_FILE)
