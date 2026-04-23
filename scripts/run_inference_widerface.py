import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
import torchvision.transforms as T
import os

# ==============================
# 🔷 PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fasterrcnn_widerface.pth")

def main():
    # ==============================
    # 🔷 DEVICE LOGIC
    # ==============================
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🔥 Using Device: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not locate Face Detector at {MODEL_PATH}")
        print("You must train it first by running `utils/train_widerface.py`")
        return

    # ==============================
    # 🚀 LOAD FASTER R-CNN
    # ==============================
    print("⏳ Loading Custom WIDER FACE Detector (Faster R-CNN)...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 2 classes: Background, Face
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode!
    
    print("✅ Model loaded successfully.")

    # Image transformation logic
    transform = T.Compose([
        T.ToTensor() # Converts [0,255] numpy HWC to [0,1] tensor CHW
    ])

    # ==============================
    # 📷 START WEBCAM
    # ==============================
    print("📷 Starting live feed... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Webcam not found.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # PyTorch object detection models expect RGB format
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to Tensor dynamically
        input_tensor = transform(img_rgb).to(device)

        # ==============================
        # 🔍 INFERENCE PASS
        # ==============================
        with torch.no_grad():
            prediction = model([input_tensor])[0]

        # Structure of prediction:
        # 'boxes' -> [N, 4] tensor of bounding boxes
        # 'labels' -> [N] tensor of discrete labels (will all be 1 for Face)
        # 'scores' -> [N] tensor of confidence probabilities
        
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        # ==============================
        # 🎨 DRAW BOUNDING BOXES
        # ==============================
        # Set a slightly softer threshold since custom Object Detectors often hover around 20-40% confidence early in training
        CONFIDENCE_THRESHOLD = 0.15

        for i in range(len(boxes)):
            if scores[i] >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = [int(coord) for coord in boxes[i]]
                
                # Bounding Box (Dark Blue)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Confidence Tag
                text = f"Face ({scores[i]*100:.1f}%)"
                cv2.putText(frame, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show Output
        cv2.imshow("Custom WIDER FACE Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
