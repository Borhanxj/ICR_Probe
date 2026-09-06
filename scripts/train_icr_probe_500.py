import json
import random

import numpy as np
import torch

from src.icr_probe import ICRProbeTrainer
from src.utils import ICRProbeConfig


DATA_FILE = "artifacts/triviaqa_icr_500_em_labeled.jsonl"
SPLIT_FILE = "artifacts/triviaqa_fixed_split.json"
MODEL_DIR = "artifacts/icr_probe_500"

SEED = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------
# Load data
# --------------------------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)


record_by_id = {
    record["id"]: record
    for record in records
}


def build_xy(ids):
    X = np.array(
        [record_by_id[i]["features"] for i in ids],
        dtype=np.float32,
    )

    y = np.array(
        [record_by_id[i]["label"] for i in ids],
        dtype=np.float32,
    )

    return X, y


X_train, y_train = build_xy(split["train_ids"])
X_val, y_val = build_xy(split["val_ids"])


print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)


# --------------------------------------------------
# Probe configuration
# --------------------------------------------------

config = ICRProbeConfig(
    input_dim=X_train.shape[1],
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=16,
    epochs=50,
    lr_factor=0.5,
    lr_patience=5,
    halu_threshold=0.5,
    save_dir=MODEL_DIR,
)


trainer = ICRProbeTrainer(config)


# --------------------------------------------------
# Train
# --------------------------------------------------

trainer.train(
    X_train,
    y_train,
    X_val,
    y_val,
)


# --------------------------------------------------
# Reload best validation checkpoint
# --------------------------------------------------

trainer.model.load_state_dict(
    torch.load(
        f"{MODEL_DIR}/model.pth",
        map_location=trainer.device,
        weights_only=True,
    )
)


# --------------------------------------------------
# Final validation metrics
# --------------------------------------------------

results = trainer.evaluate(
    X_val,
    y_val,
)


print("\n==============================")
print("FINAL VALIDATION RESULTS")
print("==============================")

for key, value in results.items():
    print(f"{key}: {value}")