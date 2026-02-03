import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from utils.dataset import PatchDataset
from models.patch_cnn import PatchCNN

# ---------- CONFIG ----------
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Starting Training")
# ---------- DATA ----------
train_original = PatchDataset(
    "data/archive/TRAINING_CG-1050/TRAINING/ORIGINAL",
    label=0
)
print("Training original dataset size:", len(train_original))
train_tampered = PatchDataset(
    "data/archive/TRAINING_CG-1050/TRAINING/TAMPERED",
    label=1
)
 
print("Training tampered dataset size:", len(train_tampered))
train_dataset = ConcatDataset([train_original, train_tampered])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Total training patches:", len(train_dataset))

# ---------- MODEL ----------
model = PatchCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print("start loop")
# ---------- TRAIN ----------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for patches, labels in train_loader:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# ---------- SAVE MODEL ----------
torch.save(model.state_dict(), "models/tamper_classifier.pth")
print("Model saved to models/tamper_classifier.pth")
