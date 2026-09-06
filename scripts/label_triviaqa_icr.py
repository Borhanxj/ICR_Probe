import json
import re


INPUT_FILE = "artifacts/triviaqa_icr_100_prompt_b.jsonl"
OUTPUT_FILE = "artifacts/triviaqa_icr_100_prompt_b_labeled.jsonl"


def normalize(text):
    """
    Convert text into a simpler form for comparison.
    """

    text = text.lower().strip()

    # Remove punctuation.
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_correct(model_answer, correct_answer, aliases):
    """
    Return True when the generated answer clearly matches
    the official answer or one of its aliases.
    """

    model = normalize(model_answer)

    candidates = [correct_answer] + aliases

    for answer in candidates:
        candidate = normalize(answer)

        if not candidate:
            continue

        # Exact normalized match
        if model == candidate:
            return True

        # Allow simple extensions such as:
        #
        # "Nikkei" -> "Nikkei Index"
        # "Kilimanjaro" -> "Mount Kilimanjaro"
        # "Vancouver" -> "Vancouver British Columbia"
        #
        if candidate in model:
            return True

    return False


correct_count = 0
wrong_count = 0


with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:

        record = json.loads(line)

        correct = is_correct(
            record["model_answer"],
            record["correct_answer"],
            record["aliases"],
        )

        # ICR Probe convention for our experiment:
        #
        # 0 = correct
        # 1 = hallucination / incorrect
        label = 0 if correct else 1

        record["label"] = label

        if label == 0:
            correct_count += 1
        else:
            wrong_count += 1

        fout.write(json.dumps(record) + "\n")


print("Finished labeling.")
print("Correct:", correct_count)
print("Incorrect:", wrong_count)
print("Saved:", OUTPUT_FILE)