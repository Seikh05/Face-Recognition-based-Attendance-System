import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F

# ==============================
# 🔷 PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenet_v2_face.pth")

def main():
    # ==============================
    # 🔷 DEVICE SETUP
    # ==============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the CNN model first using utils/train_cnn_mobilenet.py")
        return

    # ==============================
    # 💾 LOAD TRAINED CNN MODEL
    # ==============================
    print("⏳ Loading Trained MobileNetV2 Model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")

    # ==============================
    # 🔷 INITIALIZE MTCNN (For Detection)
    # ==============================
    print("⏳ Loading MTCNN for Face Detection...")
    # We only use MTCNN to locate the box. MobileNet will classify the face.
    mtcnn = MTCNN(keep_all=True, device=device)

    # Transforms required by ImageNet/MobileNetV2
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ==============================
    # 📷 START WEBCAM
    # ==============================
    print("📷 Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        # Convert to RGB for MTCNN and PIL
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ==============================
        # 🔍 FACE DETECTION
        # ==============================
        boxes, probs = mtcnn.detect(img_rgb)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                # Ignore loose boxes
                if prob < 0.8:
                    continue
                
                # Extract Bounding Box
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Crop the face natively
                face_crop = img_rgb[y1:y2, x1:x2]
                
                # Check for bad crops at boarders
                if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    continue

                # ==============================
                # 🚀 PREPARE FOR CNN
                # ==============================
                pil_image = Image.fromarray(face_crop)
                input_tensor = transform(pil_image).unsqueeze(0).to(device)

                # ==============================
                # 🎯 CLASSIFY WITH MOBILENETV2
                # ==============================
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)
                    max_prob, predicted_idx = torch.max(probabilities, 1)

                max_prob = max_prob.item()
                predicted_label = class_names[predicted_idx.item()]

                # Threshold to detect "Unknown"
                if max_prob < 0.7:  # You can adjust this strictness threshold!
                    label = "Unknown"
                    color = (0, 0, 255) # Red bounding box
                else:
                    label = predicted_label
                    color = (0, 255, 0) # Green bounding box

                text = f"{label} ({max_prob*100:.1f}%)"

                # Draw Bounding Box and Label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show the framed video feed
        cv2.imshow("Live Face Recognition (MobileNetV2 Transfer Learning)", frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
