import json
import re
import string


FILES = {
    "A": (
        "artifacts/triviaqa_icr_100.jsonl",
        "artifacts/triviaqa_icr_100_em_labeled.jsonl",
    ),
    "B": (
        "artifacts/triviaqa_icr_100_prompt_b.jsonl",
        "artifacts/triviaqa_icr_100_prompt_b_em_labeled.jsonl",
    ),
    "C": (
        "artifacts/triviaqa_icr_100_prompt_c.jsonl",
        "artifacts/triviaqa_icr_100_prompt_c_em_labeled.jsonl",
    ),
}


def normalize_answer(text):
    text = text.lower()
    text = text.replace("_", " ")

    punctuation = set(string.punctuation + "‘’´`")

    text = "".join(
        " " if char in punctuation else char
        for char in text
    )

    # Remove English articles.
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove duplicated spaces.
    text = " ".join(text.split())

    return text.strip()


def is_correct(model_answer, correct_answer, aliases):
    prediction = normalize_answer(model_answer)

    possible_answers = [correct_answer] + aliases

    for answer in possible_answers:
        if prediction == normalize_answer(answer):
            return True

    return False


for prompt_style, (input_file, output_file) in FILES.items():

    correct_count = 0
    incorrect_count = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)

            correct = is_correct(
                record["model_answer"],
                record["correct_answer"],
                record["aliases"],
            )

            # 0 = correct
            # 1 = hallucination / incorrect
            record["label"] = 0 if correct else 1
            record["label_method"] = "triviaqa_exact_match"

            if correct:
                correct_count += 1
            else:
                incorrect_count += 1

            fout.write(json.dumps(record) + "\n")

    print(
        f"Prompt {prompt_style}: "
        f"Correct={correct_count}, "
        f"Incorrect={incorrect_count}"
    )