import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from src.utils import ICRProbe


PROMPT_A_FILE = "artifacts/triviaqa_icr_100_labeled.jsonl"
PROMPT_B_FILE = "artifacts/triviaqa_icr_100_prompt_b_labeled.jsonl"

MODEL_FILE = "artifacts/icr_probe_test/model.pth"


# --------------------------------------------------
# 1. Load both datasets
# --------------------------------------------------

def load_data(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


prompt_a = load_data(PROMPT_A_FILE)
prompt_b = load_data(PROMPT_B_FILE)


# --------------------------------------------------
# 2. Reconstruct the EXACT validation split
#    used when we trained the Prompt-A probe
# --------------------------------------------------

ids = np.array([record["id"] for record in prompt_a])
labels_a = np.array([record["label"] for record in prompt_a])


train_ids, val_ids = train_test_split(
    ids,
    test_size=0.20,
    random_state=42,
    stratify=labels_a,
)


print("Validation IDs:")
print(sorted(val_ids.tolist()))


# --------------------------------------------------
# 3. Select those same IDs from both prompt styles
# --------------------------------------------------

a_by_id = {record["id"]: record for record in prompt_a}
b_by_id = {record["id"]: record for record in prompt_b}


X_a = np.array(
    [a_by_id[i]["features"] for i in val_ids],
    dtype=np.float32,
)

y_a = np.array(
    [a_by_id[i]["label"] for i in val_ids],
    dtype=np.float32,
)


X_b = np.array(
    [b_by_id[i]["features"] for i in val_ids],
    dtype=np.float32,
)

y_b = np.array(
    [b_by_id[i]["label"] for i in val_ids],
    dtype=np.float32,
)


# --------------------------------------------------
# 4. Make sure our ICR data is numerically valid
# --------------------------------------------------

if np.isnan(X_a).any():
    raise ValueError("Prompt A contains NaNs!")

if np.isnan(X_b).any():
    raise ValueError("Prompt B contains NaNs!")


# --------------------------------------------------
# 5. Load the EXISTING Prompt-A probe
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

input_dim = X_a.shape[1]

model = ICRProbe(
    input_dim=input_dim
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True,
    )
)

model.eval()


# --------------------------------------------------
# 6. Get hallucination scores
# --------------------------------------------------

def predict_scores(X):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        scores = model(X_tensor).squeeze(1)

    return scores.cpu().numpy()


scores_a = predict_scores(X_a)
scores_b = predict_scores(X_b)


# --------------------------------------------------
# 7. Calculate AUROC
# --------------------------------------------------

auc_a = roc_auc_score(y_a, scores_a)
auc_b = roc_auc_score(y_b, scores_b)


# Threshold accuracy is secondary,
# but useful to inspect.
pred_a = (scores_a >= 0.5).astype(int)
pred_b = (scores_b >= 0.5).astype(int)

acc_a = accuracy_score(y_a, pred_a)
acc_b = accuracy_score(y_b, pred_b)


# --------------------------------------------------
# 8. Results
# --------------------------------------------------

print("\n================================")
print("CROSS-PROMPT RESULTS")
print("================================")

print(f"Validation examples: {len(val_ids)}")

print("\nPrompt A:")
print(f"Correct: {(y_a == 0).sum()}")
print(f"Hallucinated: {(y_a == 1).sum()}")
print(f"AUROC: {auc_a:.4f}")
print(f"Accuracy: {acc_a:.4f}")

print("\nPrompt B:")
print(f"Correct: {(y_b == 0).sum()}")
print(f"Hallucinated: {(y_b == 1).sum()}")
print(f"AUROC: {auc_b:.4f}")
print(f"Accuracy: {acc_b:.4f}")

print("\nAUROC change:")
print(f"{auc_a:.4f} -> {auc_b:.4f}")
print(f"Difference: {auc_b - auc_a:+.4f}")