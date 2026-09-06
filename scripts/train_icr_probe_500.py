import json
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.icr_probe import ICRProbeTrainer


DATA_FILE = "artifacts/triviaqa_icr_500_em_labeled.jsonl"
SPLIT_FILE = "artifacts/triviaqa_fixed_split.json"
MODEL_DIR = "artifacts/icr_probe_500"

SEED = 42


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------
# 2. Load dataset and fixed split
# --------------------------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)


record_by_id = {
    record["id"]: record
    for record in records
}


# --------------------------------------------------
# 3. Build features/labels for specific IDs
# --------------------------------------------------

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


print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

print(
    "Training labels:",
    int((y_train == 0).sum()),
    "correct /",
    int((y_train == 1).sum()),
    "incorrect",
)

print(
    "Validation labels:",
    int((y_val == 0).sum()),
    "correct /",
    int((y_val == 1).sum()),
    "incorrect",
)


# --------------------------------------------------
# 4. Safety checks
# --------------------------------------------------

if np.isnan(X_train).any():
    raise ValueError("Training features contain NaN values!")

if np.isnan(X_val).any():
    raise ValueError("Validation features contain NaN values!")


# --------------------------------------------------
# 5. PyTorch datasets/loaders
# --------------------------------------------------

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
)


# --------------------------------------------------
# 6. Configuration
# --------------------------------------------------

config = SimpleNamespace(
    learning_rate=1e-3,
    weight_decay=1e-4,

    lr_factor=0.5,
    lr_patience=5,

    num_epochs=50,

    halu_threshold=0.5,

    save_dir=MODEL_DIR,
)


# --------------------------------------------------
# 7. Create and initialize probe
# --------------------------------------------------

trainer = ICRProbeTrainer(
    model=None,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
)

trainer.setup_model()


# --------------------------------------------------
# 8. Train
# --------------------------------------------------

trainer.train()


# --------------------------------------------------
# 9. Reload best validation checkpoint
# --------------------------------------------------

trainer.model.load_state_dict(
    torch.load(
        f"{MODEL_DIR}/model.pth",
        map_location=trainer.device,
        weights_only=True,
    )
)


# --------------------------------------------------
# 10. Evaluate on validation set only
# --------------------------------------------------

metrics = trainer._validate_epoch()


print("\n==============================")
print("FINAL VALIDATION RESULTS")
print("==============================")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")