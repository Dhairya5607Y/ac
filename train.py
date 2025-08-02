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
DATA_DIR = 'augmented_dataset'
MODEL_PATH = 'models/efficientnetb0.pth'
CLASS_NAMES_PATH = 'models/class_names.txt'
BATCH_SIZE = 8
EPOCHS = 10
IMAGE_SIZE = (224, 224)
TRAIN_SPLIT = 0.9

print(f"🖥 Using device: {DEVICE}")
os.makedirs('models', exist_ok=True)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# === Dataset ===
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = full_dataset.classes
with open(CLASS_NAMES_PATH, 'w') as f:
    f.write('\n'.join(class_names))

train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

print(f"🔢 Training samples: {train_size} | Validation samples: {val_size}")

# === Model ===
model = EfficientNet.from_name('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, len(class_names))
model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# === Training ===
model.train()
total_loss, correct, total = 0, 0, 0

for epoch in range(EPOCHS):
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

    print(f"\n📊 Epoch {epoch+1}: Train Loss={total_loss:.4f} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
