import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ==============================
# 🔷 SETUP EXPERIMENT
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "augmented_dataset")  # Using augmented dataset
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "mobilenet_v2_face.pth")
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

if not os.path.exists(DATASET_DIR):
    print(f"❌ Error: Dataset directory {DATASET_DIR} not found.")
    exit()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# ==============================
# 🔷 DATA PREPARATION
# ==============================
print("\n🔄 Preparing dataset... (Transforming & Splitting)")

# Transforms (Data Augmentation & Normalization for MobileNetV2)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # Normalizing with ImageNet stats as required by PyTorch pretrained models
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
print(f"✅ Found {len(dataset)} images belonging to {len(dataset.classes)} classes: {dataset.classes}")

# Split dataset (70% Train, 15% Validation, 15% Test)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Set seed for reproducibility
generator = torch.Generator().manual_seed(42)
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size], generator=generator)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"📊 Split sizes -> Train: {train_size} | Validation: {val_size} | Test: {test_size}")

# ==============================
# 🔷 MODEL ARCHITECTURE (TRANSFER LEARNING)
# ==============================
print("\n🚀 Loading pretrained MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Modify final classification layer
model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==============================
# 🔷 TRAINING LOOP
# ==============================
print(f"\n🔥 Starting Training for {EPOCHS} Epochs...\n")

history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    total_loss, correct_train, total_train = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = total_loss / total_train
    train_acc = correct_train / total_train

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_loss / total_val
    val_acc = correct_val / total_val

    # Save to history for plotting
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

# ==============================
# 💾 SAVE MODEL
# ==============================
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': dataset.classes,
    'transform_mean': [0.485, 0.456, 0.406],
    'transform_std': [0.229, 0.224, 0.225]
}, MODEL_SAVE_PATH)
print(f"\n✅ Model state successfully saved at: {MODEL_SAVE_PATH}")

# ==============================
# 📊 GENERATE PLOTS (FOR REPORT)
# ==============================
print("\n📈 Generating training metric plots...")

plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), history['train_loss'], label='Train Loss', marker='o')
plt.plot(range(1, EPOCHS+1), history['val_loss'], label='Val Loss', marker='o')
plt.title('Loss vs Epochs (MobileNetV2)')
plt.xlabel('Epochs')
plt.ylabel('CrossEntropy Loss')
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), history['train_acc'], label='Train Acc', marker='o')
plt.plot(range(1, EPOCHS+1), history['val_acc'], label='Val Acc', marker='o')
plt.title('Accuracy vs Epochs (MobileNetV2)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plot_path = os.path.join(PLOTS_DIR, 'mobilenet_training_metrics.png')
plt.savefig(plot_path)
print(f"✅ Training plots saved at: {plot_path}")

print("\n🎉 DONE! You can now run `utils/evaluate_cnn_mobilenet.py` to get test metrics correctly.")
