import json


A_FILE = "artifacts/triviaqa_icr_500_em_labeled.jsonl"
D_FILE = "artifacts/triviaqa_icr_500_prompt_d_em_labeled.jsonl"
SPLIT_FILE = "artifacts/triviaqa_fixed_split.json"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


a = load(A_FILE)
d = load(D_FILE)

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)


a_by_id = {x["id"]: x for x in a}
d_by_id = {x["id"]: x for x in d}


shown = 0

for idx in split["test_ids"]:

    A = a_by_id[idx]
    D = d_by_id[idx]

    # Skip cases where both already exact-match.
    if A["label"] == 0 and D["label"] == 0:
        continue

    shown += 1

    print("\n" + "=" * 75)
    print("ID:", idx)
    print("Question:", A["question"])
    print("Gold:", A["correct_answer"])

    print()
    print("Prompt A:")
    print("Answer:", A["model_answer"])
    print("Strict label:", A["label"])

    print()
    print("Prompt D:")
    print("Answer:", D["model_answer"])
    print("Strict label:", D["label"])


print("\nCases shown:", shown)