import json
import os

from datasets import load_dataset
from scripts.icr_pipeline import extract_icr_features


TOTAL_EXAMPLES = 500

OLD_FILE = "artifacts/triviaqa_icr_100_prompt_d.jsonl"
OUTPUT_FILE = "artifacts/triviaqa_icr_500_prompt_d.jsonl"


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


def extract_final_answer(text):
    marker = "Final answer:"

    if marker.lower() not in text.lower():
        return None

    index = text.lower().rfind(marker.lower())

    answer = text[index + len(marker):].strip()
    answer = answer.split("\n")[0].strip()

    return answer


# Copy the already-computed first 100 only once.
if not os.path.exists(OUTPUT_FILE):

    with open(OLD_FILE, "r", encoding="utf-8") as src:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as dst:
            for line in src:
                dst.write(line)

    print("Copied existing 100 examples to:", OUTPUT_FILE)


existing_ids = load_existing_ids(OUTPUT_FILE)

print("Already collected:", len(existing_ids))


dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "rc.nocontext",
    split=f"validation[:{TOTAL_EXAMPLES}]",
)


with open(OUTPUT_FILE, "a", encoding="utf-8") as f:

    for i, example in enumerate(dataset):

        # Safe resume if the process gets interrupted.
        if i in existing_ids:
            continue

        question = example["question"]
        correct_answer = example["answer"]["value"]
        aliases = example["answer"]["aliases"]

        # EXACT same CoT prompt used in the pilot.
        prompt = (
            "Answer the following trivia question. "
            "Reason step by step using at most 3 short steps. "
            "Then give your short final answer using exactly this format:\n"
            "Final answer: <answer>\n\n"
            f"Question: {question}"
        )

        full_output, features = extract_icr_features(
            prompt,
            max_new_tokens=128,
        )

        final_answer = extract_final_answer(full_output)

        record = {
            "id": i,
            "prompt_style": "D",
            "question": question,
            "correct_answer": correct_answer,
            "aliases": aliases,
            "model_output": full_output.strip(),
            "model_answer": final_answer,
            "features": features,
        }

        f.write(json.dumps(record) + "\n")
        f.flush()

        existing_ids.add(i)

        print(
            f"[{len(existing_ids)}/{TOTAL_EXAMPLES}] "
            f"ID={i} | Gold={correct_answer} | "
            f"Final={final_answer}"
        )


print("\nFinished.")
print("Saved:", OUTPUT_FILE)