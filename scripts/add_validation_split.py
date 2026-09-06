import json

from sklearn.model_selection import train_test_split


DATA_FILE = "artifacts/triviaqa_icr_500_em_labeled.jsonl"
SPLIT_FILE = "artifacts/triviaqa_fixed_split.json"

RANDOM_SEED = 42


with open(DATA_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)


label_by_id = {
    record["id"]: record["label"]
    for record in records
}


original_train_ids = split["train_ids"]
test_ids = split["test_ids"]

train_labels = [
    label_by_id[i]
    for i in original_train_ids
]


train_ids, val_ids = train_test_split(
    original_train_ids,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=train_labels,
)


split["train_ids"] = sorted(train_ids)
split["val_ids"] = sorted(val_ids)
split["test_ids"] = sorted(test_ids)


with open(SPLIT_FILE, "w", encoding="utf-8") as f:
    json.dump(split, f, indent=2)


def counts(ids):
    correct = sum(label_by_id[i] == 0 for i in ids)
    incorrect = sum(label_by_id[i] == 1 for i in ids)
    return correct, incorrect


train_correct, train_incorrect = counts(train_ids)
val_correct, val_incorrect = counts(val_ids)
test_correct, test_incorrect = counts(test_ids)


print("Train:", len(train_ids))
print("  Correct:", train_correct)
print("  Incorrect:", train_incorrect)

print()

print("Validation:", len(val_ids))
print("  Correct:", val_correct)
print("  Incorrect:", val_incorrect)

print()

print("Test:", len(test_ids))
print("  Correct:", test_correct)
print("  Incorrect:", test_incorrect)