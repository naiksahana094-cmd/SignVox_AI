
import csv
import numpy as np

DATASET_FILE = "dataset/isl_data.csv"

data = {}

with open(DATASET_FILE, "r", newline="") as file:
    reader = csv.reader(file)
    headings = next(reader)

    for row in reader:
        if len(row) != len(headings):
            continue

        label = row[0]
        features = [float(value) for value in row[1:]]

        data.setdefault(label, []).append(features)

print("\nFEATURE VARIATION CHECK")

for letter in sorted(data):

    samples = np.array(data[letter])

    variation = np.mean(np.std(samples, axis=0))

    print(
        f"{letter}: "
        f"{len(samples)} samples | "
        f"Variation: {variation:.6f}"
    )
