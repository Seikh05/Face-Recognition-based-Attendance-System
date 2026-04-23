import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import os
import numpy as np
import joblib

# ==============================
# 🔷 PATH SETUP & CONSTRAINTS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(BASE_DIR, "models", "fasterrcnn_widerface.pth")
SVM_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_model_gpu.pkl")

# Neural Thresolds
DETECTION_THRESHOLD = 0.65  # Brain 1 certainty rating to isolate a face mathematically
RECOGNITION_THRESHOLD = 0.60 # Brain 2 SVM certainty rating to claim Identity

def main():
    print("🔥 BOOT SEQUENCE: initializing The Dual-Brain System...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using Master Engine: {device}")

    # ==============================
    # 🧠 BRAIN 1: FACE DETECTOR (Faster R-CNN)
    # ==============================
    print("⏳ Loading Brain 1 (WIDER FACE Object Detector)...")
    if not os.path.exists(DETECTOR_PATH):
        print(f"❌ Error: Missing Detector at {DETECTOR_PATH}")
        return
        
    model_det = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model_det.roi_heads.box_predictor.cls_score.in_features
    # Binary Classification (Face vs Background)
    model_det.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model_det.load_state_dict(torch.load(DETECTOR_PATH, map_location=device))
    model_det.to(device)
    model_det.eval()  # Lock weights to Evaluation Mode
    
    transform_det = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # ==============================
    # 🧠 BRAIN 2: IDENTITY RECOGNIZER (FaceNet + SVM)
    # ==============================
    print("⏳ Loading Brain 2 (FaceNet Embeddings + Traditional SVM)...")
    if not os.path.exists(SVM_MODEL_PATH):
        print(f"❌ Error: Missing Classifier at {SVM_MODEL_PATH}")
        return

    # Load the Pre-Trained FaceNet Embedding Extractor
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Load the Scikit-Learn SVM Classifier
    svm_classifier = joblib.load(SVM_MODEL_PATH)

    print("\n✅ TANDEM NETWORK SUCCESS! Firing up Webcam Matrix...")

    # ==============================
    # 📷 LIVE WEBCAM LOOP
    # ==============================
    print("\nPress 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ CRITICAL ERROR: Could not map to webcam hardware! Make sure Zoom/Skype is closed.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FastRCNN expects native RGB parsing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # =========================================================
        # --- BRAIN 1 EXECUTION (Instantly Search for bounding boxes) ---
        # =========================================================
        input_tensor_det = transform_det(img_rgb).to(device)
        
        with torch.no_grad():
            prediction = model_det([input_tensor_det])[0]

        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] >= DETECTION_THRESHOLD:
                x1, y1, x2, y2 = [int(coord) for coord in boxes[i]]
                
                # Math bounds safely trap coordinates from exceeding window pixels
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # --- SYNAPSE TRANSFER: Slice out the Face Matrix geometrically ---
                face_crop = img_rgb[y1:y2, x1:x2]
                
                # Failsafe verify the crop actually exists
                if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    continue

                # =========================================================
                # --- BRAIN 2 EXECUTION (Feed Crop into FaceNet -> SVM) ---
                # =========================================================
                # 1. Standardize Crop for FaceNet (Strictly requires 160x160)
                face_pil = Image.fromarray(face_crop).resize((160, 160))
                face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
                
                # 2. Mathematical Normalization (-1 to +1 range roughly)
                face_tensor = (face_tensor - 127.5) / 128.0
                face_tensor = face_tensor.unsqueeze(0).to(device)
                
                # 3. Extract Deep Learning Embedding (512-dimensional vector)
                with torch.no_grad():
                    embedding = resnet(face_tensor).cpu().numpy()
                
                # 4. Process Embedding cleanly through Traditional Machine Learning SVM
                prediction_label = svm_classifier.predict(embedding)
                prob_svm = svm_classifier.predict_proba(embedding)
                max_prob_val = np.max(prob_svm)

                # Logic Check
                if max_prob_val < RECOGNITION_THRESHOLD:
                    label = "Unknown Intruder"
                    color = (0, 0, 255) # Warning Red
                else:
                    label = prediction_label[0]
                    color = (0, 255, 0) # Verified Green

                # =========================================================
                # --- UI OVERLAY ---
                # =========================================================
                text = f"{label} ({max_prob_val*100:.1f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Custom Advanced Attendance System (FasterRCNN + FaceNet/SVM)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup Hardware
    cap.release()
    cv2.destroyAllWindows()
    print("Session Terminated Safely.")

if __name__ == "__main__":
    main()
