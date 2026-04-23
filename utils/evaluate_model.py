import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==============================
# 🔷 PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "augmented_dataset")

# ==============================
# 🔷 DEVICE SETUP
# ==============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔥 Using device: {device}")

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Dataset directory not found at {DATASET_DIR}")
        return

    # ==============================
    # 🔷 INITIALIZE MTCNN & FACENET
    # ==============================
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=20,
        thresholds=[0.5, 0.6, 0.6],
        device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    encodings = []
    labels = []

    # ==============================
    # 🔷 PROCESS DATASET
    # ==============================
    print("\n🔄 Extracting embeddings for evaluation...")

    persons = [p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p))]
    
    if len(persons) < 2:
        print("❌ Error: Need at least 2 people in the dataset to evaluate.")
        return

    for person in persons:
        person_path = os.path.join(DATASET_DIR, person)
        images = os.listdir(person_path)

        for img_name in tqdm(images, desc=f"👤 {person}"):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None: continue
            h, w = img.shape[:2]
            if h < 50 or w < 50: continue

            if min(h, w) < 160:
                scale = 160 / min(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, probs = mtcnn.detect(img_rgb)

            if boxes is None or len(boxes) != 1: continue

            face = mtcnn(img_rgb)
            if face is None: continue

            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(face)

            encodings.append(embedding.cpu().numpy()[0])
            labels.append(person)

    if len(encodings) == 0:
        print("❌ ERROR: No faces detected.")
        return

    X = np.array(encodings)
    y = np.array(labels)

    print(f"\n✅ Total valid samples: {len(X)}")

    # ==============================
    # 📊 EVALUATE MODEL (5-Fold CV)
    # ==============================
    print("\n📊 Running 5-Fold Cross Validation...")
    model = SVC(kernel='linear', probability=True)
    
    # We use StratifiedKFold to ensure equal class distributions in each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    try:
        results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        print("\n==============================")
        print("📈 MODEL CROSS-VALIDATION METRICS")
        print("==============================")
        print(f"Accuracy:  {np.mean(results['test_accuracy']) * 100:.2f}% (± {np.std(results['test_accuracy']) * 100:.2f}%)")
        print(f"Precision: {np.mean(results['test_precision_macro']) * 100:.2f}%")
        print(f"Recall:    {np.mean(results['test_recall_macro']) * 100:.2f}%")
        print(f"F1-Score:  {np.mean(results['test_f1_macro']) * 100:.2f}%")
        print("==============================\n")
        
        # Cross-val predict for confusion matrix
        print("📊 Generating Confusion Matrix Plot...")
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        labels_unique = np.unique(y)
        cm = confusion_matrix(y, y_pred, labels=labels_unique)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_unique, yticklabels=labels_unique)
        plt.title('5-Fold Cross-Validation Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plot_path = os.path.join(BASE_DIR, 'confusion_matrix.png')
        plt.savefig(plot_path)
        print(f"✅ Confusion Matrix saved at: {plot_path}")
        
    except ValueError as e:
        print(f"⚠️ Could not run Cross-Validation (maybe not enough samples per class?): {e}")
        print("Running a standard Train/Test split evaluation instead...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print("\n==============================")
        print("📈 MODEL TRAIN/TEST SPLIT METRICS")
        print("==============================")
        print(classification_report(y_test, y_pred))
        
        print("📊 Generating Confusion Matrix Plot...")
        labels_unique = np.unique(y)
        cm = confusion_matrix(y_test, y_pred, labels=labels_unique)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_unique, yticklabels=labels_unique)
        plt.title('Train/Test Split Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plot_path = os.path.join(BASE_DIR, 'confusion_matrix.png')
        plt.savefig(plot_path)
        print(f"✅ Confusion Matrix saved at: {plot_path}")

if __name__ == "__main__":
    main()
