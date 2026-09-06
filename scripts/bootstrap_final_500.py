import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.utils import ICRProbe


A_FILE = "artifacts/triviaqa_icr_500_em_labeled.jsonl"
D_FILE = "artifacts/triviaqa_icr_500_prompt_d_em_labeled.jsonl"
SPLIT_FILE = "artifacts/triviaqa_fixed_split.json"
MODEL_FILE = "artifacts/icr_probe_500/model.pth"

SEED = 42
N_BOOTSTRAPS = 10000


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


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


a_records = load_jsonl(A_FILE)
d_records = load_jsonl(D_FILE)

a_by_id = {x["id"]: x for x in a_records}
d_by_id = {x["id"]: x for x in d_records}


with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)


for idx, label in A_LABEL_OVERRIDES.items():
    a_by_id[idx]["label"] = label

for idx, label in D_LABEL_OVERRIDES.items():
    d_by_id[idx]["label"] = label


test_ids = split["test_ids"]

stable_ids = [
    i
    for i in test_ids
    if a_by_id[i]["label"] == d_by_id[i]["label"]
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

input_dim = len(a_by_id[stable_ids[0]]["features"])

model = ICRProbe(input_dim=input_dim).to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True,
    )
)

model.eval()


def scores_for(records, ids):

    X = np.array(
        [records[i]["features"] for i in ids],
        dtype=np.float32,
    )

    X = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        return model(X).squeeze(1).cpu().numpy()


labels = np.array(
    [a_by_id[i]["label"] for i in stable_ids]
)

a_scores = scores_for(a_by_id, stable_ids)
d_scores = scores_for(d_by_id, stable_ids)


observed_a = roc_auc_score(labels, a_scores)
observed_d = roc_auc_score(labels, d_scores)
observed_diff = observed_d - observed_a


rng = np.random.default_rng(SEED)

a_boot = []
d_boot = []
diff_boot = []

n = len(stable_ids)


for _ in range(N_BOOTSTRAPS):

    indices = rng.integers(
        0,
        n,
        size=n,
    )

    sampled_labels = labels[indices]

    # AUROC requires both classes.
    if len(np.unique(sampled_labels)) < 2:
        continue

    a_auc = roc_auc_score(
        sampled_labels,
        a_scores[indices],
    )

    d_auc = roc_auc_score(
        sampled_labels,
        d_scores[indices],
    )

    a_boot.append(a_auc)
    d_boot.append(d_auc)
    diff_boot.append(d_auc - a_auc)


a_ci = np.percentile(a_boot, [2.5, 97.5])
d_ci = np.percentile(d_boot, [2.5, 97.5])
diff_ci = np.percentile(diff_boot, [2.5, 97.5])


print("================================")
print("PAIRED BOOTSTRAP RESULTS")
print("================================")

print("Stable examples:", len(stable_ids))
print("Bootstrap samples:", len(diff_boot))

print()

print(f"Prompt A AUROC: {observed_a:.4f}")
print(
    f"95% CI: [{a_ci[0]:.4f}, {a_ci[1]:.4f}]"
)

print()

print(f"Prompt D AUROC: {observed_d:.4f}")
print(
    f"95% CI: [{d_ci[0]:.4f}, {d_ci[1]:.4f}]"
)

print()

print(f"Difference D - A: {observed_diff:+.4f}")
print(
    f"95% CI: [{diff_ci[0]:+.4f}, "
    f"{diff_ci[1]:+.4f}]"
)

print()

if diff_ci[1] < 0:
    print(
        "The AUROC degradation is statistically supported "
        "by the paired bootstrap."
    )
else:
    print(
        "The confidence interval includes zero."
    )