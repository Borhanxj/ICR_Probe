import json

from sklearn.model_selection import train_test_split


DATA_FILE = "artifacts/triviaqa_icr_500_em_labeled.jsonl"
OUTPUT_FILE = "artifacts/triviaqa_fixed_split.json"

RANDOM_SEED = 42


with open(DATA_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]


ids = [record["id"] for record in records]
labels = [record["label"] for record in records]


train_ids, test_ids = train_test_split(
    ids,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=labels,
)


split = {
    "random_seed": RANDOM_SEED,
    "train_ids": sorted(train_ids),
    "test_ids": sorted(test_ids),
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(split, f, indent=2)


label_by_id = {
    record["id"]: record["label"]
    for record in records
}


train_correct = sum(label_by_id[i] == 0 for i in train_ids)
train_incorrect = sum(label_by_id[i] == 1 for i in train_ids)

test_correct = sum(label_by_id[i] == 0 for i in test_ids)
test_incorrect = sum(label_by_id[i] == 1 for i in test_ids)


print("Saved:", OUTPUT_FILE)
print()
print("Train examples:", len(train_ids))
print("  Correct:", train_correct)
print("  Incorrect:", train_incorrect)

print()
print("Test examples:", len(test_ids))
print("  Correct:", test_correct)
print("  Incorrect:", test_incorrect)

print()
print("Test IDs:")
print(sorted(test_ids))