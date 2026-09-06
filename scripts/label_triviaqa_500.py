import json
import re
import string


FILES = {
    "A": (
        "artifacts/triviaqa_icr_500.jsonl",
        "artifacts/triviaqa_icr_500_em_labeled.jsonl",
    ),
    "D": (
        "artifacts/triviaqa_icr_500_prompt_d.jsonl",
        "artifacts/triviaqa_icr_500_prompt_d_em_labeled.jsonl",
    ),
}


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


for prompt_name, (input_file, output_file) in FILES.items():

    correct = 0
    incorrect = 0
    missing_answers = 0

    with open(input_file, "r", encoding="utf-8") as src, \
         open(output_file, "w", encoding="utf-8") as dst:

        for line in src:

            record = json.loads(line)

            if record["model_answer"] is None:
                missing_answers += 1

            is_correct = exact_match(
                record["model_answer"],
                record["correct_answer"],
                record["aliases"],
            )

            # Convention used throughout our project:
            # 0 = correct
            # 1 = hallucinated / incorrect
            record["label"] = 0 if is_correct else 1
            record["label_method"] = "triviaqa_exact_match"

            if is_correct:
                correct += 1
            else:
                incorrect += 1

            dst.write(json.dumps(record) + "\n")

    print(
        f"Prompt {prompt_name}: "
        f"Correct={correct}, "
        f"Incorrect={incorrect}, "
        f"Missing final answers={missing_answers}"
    )