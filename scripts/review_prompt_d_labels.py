import json
import re
import string


INPUT_FILE = "artifacts/triviaqa_icr_100_prompt_d.jsonl"


def normalize_answer(text):
    if text is None:
        return ""

    text = text.lower()
    text = text.replace("_", " ")

    punctuation = set(string.punctuation + "‘’´`")

    text = "".join(
        " " if char in punctuation else char
        for char in text
    )

    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text.strip()


def exact_match(model_answer, correct_answer, aliases):
    prediction = normalize_answer(model_answer)

    for answer in [correct_answer] + aliases:
        if prediction == normalize_answer(answer):
            return True

    return False


records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))


exact_correct = 0
needs_review = []


for record in records:

    if exact_match(
        record["model_answer"],
        record["correct_answer"],
        record["aliases"],
    ):
        exact_correct += 1

    else:
        needs_review.append(record)


print("Total:", len(records))
print("Exact-match correct:", exact_correct)
print("Needs review:", len(needs_review))

print("\n==============================")
print("ANSWERS NEEDING REVIEW")
print("==============================")

for record in needs_review:

    print(f"\nID: {record['id']}")
    print("Question:", record["question"])
    print("Correct:", record["correct_answer"])
    print("Model:", record["model_answer"])