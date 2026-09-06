import json

from datasets import load_dataset
from scripts.icr_pipeline import extract_icr_features


NUM_EXAMPLES = 100
OUTPUT_FILE = "artifacts/triviaqa_icr_100_prompt_c.jsonl"


print(f"Loading {NUM_EXAMPLES} TriviaQA examples...")

dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "rc.nocontext",
    split=f"validation[:{NUM_EXAMPLES}]"
)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i, example in enumerate(dataset):

        question = example["question"]
        correct_answer = example["answer"]["value"]
        aliases = example["answer"]["aliases"]

        prompt = (
            "Answer each trivia question using only the short answer.\n\n"
            "Question: What is the capital of France?\n"
            "Answer: Paris\n\n"
            "Question: Who wrote Hamlet?\n"
            "Answer: William Shakespeare\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        model_answer, features = extract_icr_features(
            prompt,
            max_new_tokens=20
        )

        record = {
            "id": i,
            "prompt_style": "C",
            "question": question,
            "correct_answer": correct_answer,
            "aliases": aliases,
            "model_answer": model_answer.strip(),
            "features": features,
        }

        f.write(json.dumps(record) + "\n")
        f.flush()

        print(
            f"[{i + 1}/{NUM_EXAMPLES}] "
            f"Correct: {correct_answer} | "
            f"Model: {model_answer.strip()}"
        )


print(f"\nSaved dataset to: {OUTPUT_FILE}")