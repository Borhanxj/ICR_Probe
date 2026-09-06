import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from src.utils import ICRProbe


A_FILE = "artifacts/triviaqa_icr_500_em_labeled.jsonl"
D_FILE = "artifacts/triviaqa_icr_500_prompt_d_em_labeled.jsonl"
SPLIT_FILE = "artifacts/triviaqa_fixed_split.json"
MODEL_FILE = "artifacts/icr_probe_500/model.pth"


# --------------------------------------------------
# Frozen manual corrections decided before test AUROC
# --------------------------------------------------

A_LABEL_OVERRIDES = {
    11: 0,
    123: 0,
    144: 0,
    187: 0,
    359: 0,
    420: 0,
    499: 0,
}

D_LABEL_OVERRIDES = {
    109: 0,
    168: 0,
    187: 0,
    420: 0,
}


# --------------------------------------------------
# Load files
# --------------------------------------------------

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


a_records = load_jsonl(A_FILE)
d_records = load_jsonl(D_FILE)

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)


a_by_id = {x["id"]: x for x in a_records}
d_by_id = {x["id"]: x for x in d_records}


# --------------------------------------------------
# Apply frozen label corrections
# --------------------------------------------------

for idx, label in A_LABEL_OVERRIDES.items():
    a_by_id[idx]["label"] = label

for idx, label in D_LABEL_OVERRIDES.items():
    d_by_id[idx]["label"] = label


test_ids = split["test_ids"]


# --------------------------------------------------
# Load trained probe
# --------------------------------------------------

input_dim = len(a_by_id[test_ids[0]]["features"])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ICRProbe(input_dim=input_dim).to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True,
    )
)

model.eval()


# --------------------------------------------------
# Get probe scores
# --------------------------------------------------

def get_scores(records_by_id, ids):

    X = np.array(
        [records_by_id[i]["features"] for i in ids],
        dtype=np.float32,
    )

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        scores = model(tensor).squeeze(1).cpu().numpy()

    return scores


a_scores = get_scores(a_by_id, test_ids)
d_scores = get_scores(d_by_id, test_ids)

a_labels = np.array(
    [a_by_id[i]["label"] for i in test_ids]
)

d_labels = np.array(
    [d_by_id[i]["label"] for i in test_ids]
)


# --------------------------------------------------
# Full test-set results
# --------------------------------------------------

a_auc = roc_auc_score(a_labels, a_scores)
d_auc = roc_auc_score(d_labels, d_scores)

a_pred = (a_scores >= 0.5).astype(int)
d_pred = (d_scores >= 0.5).astype(int)

a_acc = accuracy_score(a_labels, a_pred)
d_acc = accuracy_score(d_labels, d_pred)


print("\n================================")
print("FULL TEST RESULTS")
print("================================")

print("Examples:", len(test_ids))

print("\nPrompt A:")
print("Correct:", int((a_labels == 0).sum()))
print("Hallucinated:", int((a_labels == 1).sum()))
print(f"AUROC: {a_auc:.4f}")
print(f"Accuracy: {a_acc:.4f}")

print("\nPrompt D:")
print("Correct:", int((d_labels == 0).sum()))
print("Hallucinated:", int((d_labels == 1).sum()))
print(f"AUROC: {d_auc:.4f}")
print(f"Accuracy: {d_acc:.4f}")

print("\nAUROC change:")
print(f"{a_auc:.4f} -> {d_auc:.4f}")
print(f"Difference: {d_auc - a_auc:+.4f}")


# --------------------------------------------------
# Stable-label subset
# --------------------------------------------------

stable_ids = [
    i
    for i in test_ids
    if a_by_id[i]["label"] == d_by_id[i]["label"]
]

stable_a_labels = np.array(
    [a_by_id[i]["label"] for i in stable_ids]
)

stable_d_labels = np.array(
    [d_by_id[i]["label"] for i in stable_ids]
)

stable_a_scores = get_scores(
    a_by_id,
    stable_ids,
)

stable_d_scores = get_scores(
    d_by_id,
    stable_ids,
)

stable_a_auc = roc_auc_score(
    stable_a_labels,
    stable_a_scores,
)

stable_d_auc = roc_auc_score(
    stable_d_labels,
    stable_d_scores,
)


print("\n================================")
print("STABLE-LABEL RESULTS")
print("================================")

print("Examples:", len(stable_ids))

print(
    "Correct:",
    int((stable_a_labels == 0).sum()),
)

print(
    "Hallucinated:",
    int((stable_a_labels == 1).sum()),
)

print(f"Prompt A AUROC: {stable_a_auc:.4f}")
print(f"Prompt D AUROC: {stable_d_auc:.4f}")

print(
    f"Difference: "
    f"{stable_d_auc - stable_a_auc:+.4f}"
)