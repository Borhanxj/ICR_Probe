import json

from datasets import load_dataset
from scripts.icr_pipeline import extract_icr_features


NUM_EXAMPLES = 5
OUTPUT_FILE = "artifacts/triviaqa_icr_prompt_d_test.jsonl"


def extract_final_answer(text):
    """
    Extract everything after the last 'Final answer:' marker.
    """

    marker = "Final answer:"

    if marker.lower() not in text.lower():
        return None

    # Find the final occurrence case-insensitively.
    index = text.lower().rfind(marker.lower())

    answer = text[index + len(marker):].strip()

    # Keep only the first line after the marker.
    answer = answer.split("\n")[0].strip()

    return answer


dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "rc.nocontext",
    split=f"validation[:{NUM_EXAMPLES}]",
)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i, example in enumerate(dataset):

        question = example["question"]
        correct_answer = example["answer"]["value"]
        aliases = example["answer"]["aliases"]

        prompt = (
            "Answer the following trivia question. "
            "Think through the problem step by step before answering. "
            "At the end, you MUST write your short final answer using exactly "
            "this format:\n"
            "Final answer: <answer>\n\n"
            f"Question: {question}"
        )

        full_output, features = extract_icr_features(
            prompt,
            max_new_tokens=96,
        )

        final_answer = extract_final_answer(full_output)

        record = {
            "id": i,
            "prompt_style": "D",
            "question": question,
            "correct_answer": correct_answer,
            "aliases": aliases,

            # Full generated reasoning.
            "model_output": full_output.strip(),

            # Only the answer used for correctness.
            "model_answer": final_answer,

            # ICR was computed over the full generation.
            "features": features,
        }

        f.write(json.dumps(record) + "\n")
        f.flush()

        print("\n" + "=" * 60)
        print(f"Example {i + 1}")
        print("Correct answer:", correct_answer)
        print("Full output:")
        print(full_output)
        print("Extracted final answer:", final_answer)


print("\nSaved:", OUTPUT_FILE)