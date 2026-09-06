import json
from sklearn.model_selection import train_test_split


A_FILE = "artifacts/triviaqa_icr_100_em_labeled.jsonl"
D_FILE = "artifacts/triviaqa_icr_100_prompt_d_em_labeled.jsonl"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


a = load(A_FILE)
d = load(D_FILE)

ids = [x["id"] for x in a]
labels = [x["label"] for x in a]

_, val_ids = train_test_split(
    ids,
    test_size=0.20,
    random_state=42,
    stratify=labels,
)

a_by_id = {x["id"]: x for x in a}
d_by_id = {x["id"]: x for x in d}


for i in val_ids:
    A = a_by_id[i]
    D = d_by_id[i]

    if A["label"] != D["label"]:
        continue

    print("\n" + "=" * 70)
    print("ID:", i)
    print("Question:", A["question"])
    print("Gold:", A["correct_answer"])
    print("A answer:", A["model_answer"])
    print("D answer:", D["model_answer"])
    print("Label:", A["label"])