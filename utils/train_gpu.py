import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
import joblib


# ==============================
# 🔷 PATH SETUP (ROBUST)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "augmented_dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_model_gpu.pkl")

print("📂 Dataset path:", DATASET_DIR)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ==============================
# 🔷 DEVICE SETUP
# ==============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔥 Using device: {device}")

# ==============================
# 🔷 MTCNN (TUNED)
# ==============================
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=20,
    thresholds=[0.5, 0.6, 0.6],
    device=device
)

# ==============================
# 🔷 FACENET MODEL
# ==============================
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

encodings = []
labels = []

# ==============================
# 🔷 PROCESS DATASET
# ==============================
print("\n🔄 Processing dataset...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)

    if not os.path.isdir(person_path):
        continue

    print(f"\n👤 Processing: {person}")

    images = os.listdir(person_path)

    for img_name in tqdm(images):
        img_path = os.path.join(person_path, img_name)

        # ==============================
        # 📷 LOAD IMAGE
        # ==============================
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to load: {img_name}")
            continue

        h, w = img.shape[:2]

        # ==============================
        # ❌ SKIP VERY SMALL IMAGES
        # ==============================
        if h < 50 or w < 50:
            print(f"❌ Too small: {img_name}")
            continue

        # ==============================
        # 🔥 RESIZE SMALL IMAGES (CRITICAL FIX)
        # ==============================
        if min(h, w) < 160:
            scale = 160 / min(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ==============================
        # 🔍 FACE DETECTION
        # ==============================
        boxes, probs = mtcnn.detect(img_rgb)

        if boxes is None:
            # print(f"❌ No face: {img_name}")
            continue

        if len(boxes) != 1:
            # print(f"⚠️ Skipping {img_name} - {len(boxes)} faces")
            continue

        # ==============================
        # 🎯 FACE EXTRACTION
        # ==============================
        face = mtcnn(img_rgb)

        if face is None:
            # print(f"❌ Extraction failed: {img_name}")
            continue

        # ==============================
        # 🚀 EMBEDDING (GPU USED HERE)
        # ==============================
        face = face.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(face)

        encodings.append(embedding.cpu().numpy()[0])
        labels.append(person)

# ==============================
# 🔷 CHECK BEFORE TRAINING
# ==============================
print(f"\n✅ Total valid samples: {len(encodings)}")

if len(encodings) == 0:
    print("❌ ERROR: No faces detected in dataset. Training aborted.")
    exit(1)

# ==============================
# 🚀 TRAIN SVM
# ==============================
print("\n🚀 Training SVM model...")

model = SVC(kernel='linear', probability=True)
model.fit(encodings, labels)

# ==============================
# 💾 SAVE MODEL
# ==============================
joblib.dump(model, MODEL_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")