
import cv2
import csv
import os
import time

from hand_detection import create_hand_detector, detect_hand


# ==================================
# SETTINGS
# ==================================

DATASET_FILE = "dataset/word_dataset/word_data.csv"

WORDS = [
    "HELLO",
    "THANK YOU",
    "PLEASE",
    "YES",
    "NO",
    "GOOD MORNING",
    "GOOD NIGHT",
    "SORRY",
    "HELP",
    "HOW ARE YOU"
]

FRAMES_PER_SAMPLE = 45
SAMPLES_PER_WORD = 20


# ==================================
# CREATE DATASET FOLDER
# ==================================

os.makedirs("dataset/word_dataset", exist_ok=True)


# ==================================
# CREATE CSV HEADINGS
# ==================================

columns = ["label"]

for frame in range(FRAMES_PER_SAMPLE):
    for hand in range(2):
        for point in range(21):
            for coordinate in ["x", "y", "z"]:
                columns.append(
                    f"frame{frame}_hand{hand}_point{point}_{coordinate}"
                )


if not os.path.exists(DATASET_FILE):

    with open(DATASET_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)


# ==================================
# COUNT EXISTING SAMPLES
# ==================================

def count_samples(word):

    count = 0

    with open(DATASET_FILE, "r", newline="") as file:

        reader = csv.reader(file)
        next(reader, None)

        for row in reader:

            if row and row[0] == word:
                count += 1

    return count


# ==================================
# EXTRACT HAND FEATURES
# ==================================

def get_frame_features(landmarks_list):

    features = []

    for hand_index in range(2):

        if hand_index < len(landmarks_list):

            features.extend(
                landmarks_list[hand_index]
            )

        else:

            features.extend([0.0] * 63)

    return features


# ==================================
# COLLECT ONE SAMPLE
# ==================================

def collect_sample(cap, detector, timestamp):

    sequence = []

    print("Recording started! Perform the gesture.")

    for frame_number in range(FRAMES_PER_SAMPLE):

        success, frame = cap.read()

        if not success:

            print("Camera frame could not be captured.")
            return None, timestamp

        image_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        annotated_image, landmarks_list = detect_hand(
            image_rgb,
            detector,
            timestamp
        )

        timestamp += 33

        features = get_frame_features(landmarks_list)

        sequence.extend(features)

        display_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_RGB2BGR
        )

        cv2.putText(
            display_image,
            f"Recording: {frame_number + 1}/{FRAMES_PER_SAMPLE}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SignVox AI - Word Collection",
            display_image
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            return None, timestamp

    print("Recording completed!")

    return sequence, timestamp


# ==================================
# MAIN COLLECTION PROGRAM
# ==================================

def main():

    print("\n===== SignVox AI Word Data Collection =====")

    print("\nWords to collect:")

    for word in WORDS:
        print("-", word)

    print("\nEach word requires", SAMPLES_PER_WORD, "samples.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        return

    detector = None

    try:

        detector = create_hand_detector()

        timestamp = 0

        for word in WORDS:

            existing_samples = count_samples(word)

            if existing_samples >= SAMPLES_PER_WORD:

                print(
                    f"\nSkipping {word}: "
                    f"{existing_samples} samples already collected."
                )

                continue

            remaining_samples = (
                SAMPLES_PER_WORD - existing_samples
            )

            print("\n================================")
            print("Current word:", word)
            print("Existing samples:", existing_samples)
            print("Remaining samples:", remaining_samples)
            print("================================")

            input(
                f"Press ENTER when ready to collect {word}: "
            )

            for sample_number in range(remaining_samples):

                current_sample = existing_samples + sample_number + 1

                print(
                    f"\n{word} - Sample "
                    f"{current_sample}/{SAMPLES_PER_WORD}"
                )

                print("Press R in the camera window to record.")

                while True:

                    success, frame = cap.read()

                    if not success:

                        print("Camera error.")
                        return

                    cv2.putText(
                        frame,
                        f"Word: {word}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Press R to record | Q to quit",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.imshow(
                        "SignVox AI - Word Collection",
                        frame
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("r"):
                        break

                    if key == ord("q"):

                        print("Collection stopped.")
                        return

                sequence, timestamp = collect_sample(
                    cap,
                    detector,
                    timestamp
                )

                if sequence is None:

                    print("Collection stopped.")
                    return

                with open(DATASET_FILE, "a", newline="") as file:

                    writer = csv.writer(file)

                    writer.writerow(
                        [word] + sequence
                    )

                print("Sample saved successfully!")

                time.sleep(0.5)

            print(f"\n{word} collection completed!")

        print("\n================================")
        print("ALL WORD SAMPLES COLLECTED!")
        print("================================")

        print("Dataset saved at:", DATASET_FILE)

    except Exception as e:

        print("ERROR:", e)

    finally:

        if detector is not None:
            detector.close()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
