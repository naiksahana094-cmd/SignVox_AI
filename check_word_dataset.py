
import csv
from collections import Counter

DATASET_FILE = "dataset/word_dataset/word_data.csv"

with open(DATASET_FILE, "r", newline="") as file:
    reader = csv.reader(file)

    header = next(reader)
    rows = list(reader)

print("\n===== WORD DATASET REPORT =====")

print("Total samples:", len(rows))
print("Columns per sample:", len(header))

counts = Counter(row[0] for row in rows if row)

print("\nSamples collected for each word:")

for word, count in sorted(counts.items()):
    print(f"{word}: {count}")

print("\nChecking dataset...")

if len(rows) == 200 and all(len(row) == len(header) for row in rows):
    print("\nSUCCESS! Dataset structure is correct.")
else:
    print("\nWARNING! Please check the dataset.")

print("\n===== CHECK COMPLETED =====")
