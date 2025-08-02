# train.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet
from torch.optim import Adam
from tqdm import tqdm

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = 'augmented_dataset'  # 2 images per Pokémon
MODEL_PATH = 'models/efficientnetb0.pth'
CLASS_NAMES_PATH = 'models/class_names.txt'
BATCH_SIZE = 16
EPOCHS = 3
IMAGE_SIZE = (128, 128)
TRAIN_SPLIT = 0.8

os.makedirs('models', exist_ok=True)

print(f"🖥 Using device: {DEVICE}")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# === Load Dataset ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

with open(CLASS_NAMES_PATH, 'w') as f:
    f.write('\n'.join(class_names))

# === Train/Val Split ===
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

print(f"🔢 Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

# === Model Setup ===
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(class_names))
model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# === Training ===
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"📦 Epoch {epoch+1}/{EPOCHS} [Training]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100

    # === Validation ===
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total * 100

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Best model updated (Val Acc: {val_acc:.2f}%)")

    print(f"📊 Epoch {epoch+1}: Train Loss={total_loss:.4f} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

print(f"✅ Final model saved to {MODEL_PATH}")
