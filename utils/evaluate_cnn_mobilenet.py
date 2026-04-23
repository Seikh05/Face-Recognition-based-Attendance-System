import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================
# 🔷 SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "augmented_dataset")  # Same dataset config as training
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "mobilenet_v2_face.pth")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
BATCH_SIZE = 32

if not os.path.exists(MODEL_SAVE_PATH):
    print(f"❌ Error: Model not found at {MODEL_SAVE_PATH}. Please run train_cnn_mobilenet.py first.")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# ==============================
# 🔷 PREPARE TEST DATASET
# ==============================
print("🔄 Loading dataset and recovering Test Split...")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

# Important: Must use exactly the SAME split seed and sizes used during training
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator().manual_seed(42)
_, _, test_data = random_split(dataset, [train_size, val_size, test_size], generator=generator)

test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
print(f"✅ Loaded Test Split consisting of {len(test_data)} unseen images.")

# ==============================
# 🚀 LOAD TRAINED MODEL
# ==============================
print("⏳ Loading Trained MobileNetV2 Model...")

checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
class_names = checkpoint['class_names']

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# ==============================
# 📊 EVALUATION PHASE
# ==============================
print("\n🔍 Running Evaluation Metrics (Precision, Recall, F1-Score)...")

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Generate Classification Report
print("\n" + "="*50)
print("Classification Report (MobileNetV2 Transfer Learning)")
print("="*50)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

# ==============================
# 🎨 CONFUSION MATRIX
# ==============================
print("\n📈 Generating Confusion Matrix...")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - MobileNetV2 Evaluate')
plt.ylabel('Actual Identity')
plt.xlabel('Predicted Identity')

cm_path = os.path.join(PLOTS_DIR, 'mobilenet_confusion_matrix.png')
plt.savefig(cm_path)
print(f"✅ Confusion Matrix saved at: {cm_path}")
print("🎉 Evaluation Complete!")
