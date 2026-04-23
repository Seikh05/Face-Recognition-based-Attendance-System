import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
import os

# ==============================
# 🔷 PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_model_gpu.pkl")

def main():
    # ==============================
    # 🔷 DEVICE SETUP
    # ==============================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Using device: {device}")

    # ==============================
    # 💾 LOAD TRAINED CLASSIFIER
    # ==============================
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_gpu.py")
        return

    try:
        svm_model = joblib.load(MODEL_PATH)
        print("✅ Classifier model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # ==============================
    # 🔷 INITIALIZE MTCNN & FACENET
    # ==============================
    print("⏳ Loading MTCNN and InceptionResnetV1...")
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=20,
        thresholds=[0.5, 0.6, 0.6],
        keep_all=True, # Allow detecting multiple faces in one frame
        device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("✅ face models loaded successfully.")

    # ==============================
    # 📷 START WEBCAM
    # ==============================
    print("📷 Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    # Allow some time for camera to initialize
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        # Convert to RGB for MTCNN
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ==============================
        # 🔍 FACE DETECTION
        # ==============================
        boxes, probs = mtcnn.detect(img_rgb)
        
        if boxes is not None:
            # Extract face crops
            faces = mtcnn(img_rgb)
            
            if faces is not None:
                # MTCNN might return a 3D tensor if only one face is detected (depending on version/behavior)
                if faces.dim() == 3:
                    faces = faces.unsqueeze(0)
                    
                faces = faces.to(device)
                
                # ==============================
                # 🚀 GET EMBEDDINGS
                # ==============================
                with torch.no_grad():
                    embeddings = resnet(faces).cpu().numpy()
                
                # ==============================
                # 🎯 PREDICT WITH SVM
                # ==============================
                predictions = svm_model.predict(embeddings)
                
                # Check if SVM was trained with probability=True
                prob_available = hasattr(svm_model, "predict_proba")
                if prob_available:
                    probabilities = svm_model.predict_proba(embeddings)
                
                for i, box in enumerate(boxes):
                    if prob_available:
                        prob = np.max(probabilities[i])
                        # Set a threshold to identify "Unknown" faces
                        label = predictions[i] if prob > 0.6 else "Unknown"
                        text = f"{label} ({prob*100:.1f}%)"
                    else:
                        label = predictions[i]
                        text = f"{label}"

                    # Determine bounding box color
                    color = (0, 0, 255) if label == "Unknown" else (0, 255, 0)
                    
                    x1, y1, x2, y2 = [int(b) for b in box]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Display name and probability
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show the frame
        cv2.imshow("Live Face Recognition", frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
