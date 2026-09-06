import json
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from src.icr_probe import ICRProbeTrainer


DATA_FILE = "artifacts/triviaqa_icr_100_labeled.jsonl"


# --------------------------------------------------
# 1. Load features and labels
# --------------------------------------------------

features = []
labels = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        features.append(record["features"])
        labels.append(record["label"])


X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.float32)


print("Dataset shape:", X.shape)
print("Correct examples:", int((y == 0).sum()))
print("Hallucinated examples:", int((y == 1).sum()))


# --------------------------------------------------
# 2. Safety check
# --------------------------------------------------

if np.isnan(X).any():
    raise ValueError("Dataset contains NaN values!")


# --------------------------------------------------
# 3. Split into training and validation data
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))


# --------------------------------------------------
# 4. Convert to PyTorch datasets
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
# 5. Training configuration
# --------------------------------------------------

config = SimpleNamespace(
    learning_rate=1e-3,
    weight_decay=1e-4,

    lr_factor=0.5,
    lr_patience=5,

    num_epochs=30,

    halu_threshold=0.5,

    save_dir="artifacts/icr_probe_test",
)


# --------------------------------------------------
# 6. Create the ICR Probe trainer
# --------------------------------------------------

trainer = ICRProbeTrainer(
    model=None,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
)

trainer.setup_model()


# --------------------------------------------------
# 7. Train
# --------------------------------------------------

trainer.train()


# --------------------------------------------------
# 8. Load the best saved model
# --------------------------------------------------

trainer.model.load_state_dict(
    torch.load(
        "artifacts/icr_probe_test/model.pth",
        map_location=trainer.device,
        weights_only=True,
    )
)


# --------------------------------------------------
# 9. Evaluate the best model
# --------------------------------------------------

metrics = trainer._validate_epoch()


print("\n==============================")
print("FINAL VALIDATION RESULTS")
print("==============================")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")