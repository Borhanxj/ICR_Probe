import json
import os

from datasets import load_dataset
from scripts.icr_pipeline import extract_icr_features


TOTAL_EXAMPLES = 500

OLD_FILE = "artifacts/triviaqa_icr_100.jsonl"
OUTPUT_FILE = "artifacts/triviaqa_icr_500.jsonl"


def load_existing_ids(path):
    ids = set()

    if not os.path.exists(path):
        return ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                ids.add(record["id"])

    return ids


# --------------------------------------------------
# Start the 500-example file using our existing 100.
# --------------------------------------------------

if not os.path.exists(OUTPUT_FILE):
    with open(OLD_FILE, "r", encoding="utf-8") as src:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as dst:
            for line in src:
                dst.write(line)

    print("Copied existing 100 examples to:", OUTPUT_FILE)


existing_ids = load_existing_ids(OUTPUT_FILE)

print("Already collected:", len(existing_ids))


# --------------------------------------------------
# Load the exact first 500 TriviaQA validation items.
# --------------------------------------------------

dataset = load_dataset(
    "trivia_qa",
    "rc",
    split=f"validation[:{TOTAL_EXAMPLES}]",
)


with open(OUTPUT_FILE, "a", encoding="utf-8") as f:

    for i, example in enumerate(dataset):

        # Allows the script to safely resume if interrupted.
        if i in existing_ids:
            continue

        question = example["question"]
        correct_answer = example["answer"]["value"]
        aliases = example["answer"]["aliases"]

        # EXACT same Prompt A used in our pilot.
        prompt = (
            "Answer the following question with only the short answer. "
            "Do not explain.\n"
            f"Question: {question}\n"
            "Answer:"
        )

        model_answer, features = extract_icr_features(
            prompt,
            max_new_tokens=20,
        )

        record = {
            "id": i,
            "question": question,
            "correct_answer": correct_answer,
            "aliases": aliases,
            "model_answer": model_answer.strip(),
            "features": features,
        }

        f.write(json.dumps(record) + "\n")
        f.flush()

        existing_ids.add(i)

        print(
            f"[{len(existing_ids)}/{TOTAL_EXAMPLES}] "
            f"ID={i} | Gold={correct_answer} | "
            f"Model={model_answer.strip()}"
        )


print("\nFinished.")
print("Saved:", OUTPUT_FILE)