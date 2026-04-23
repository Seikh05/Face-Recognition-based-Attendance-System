import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
import joblib

# ==============================
# 🔷 PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(BASE_DIR, "assets", "image.png")

# Models
SVM_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_model_gpu.pkl")
MOBILENET_PATH = os.path.join(BASE_DIR, "models", "mobilenet_v2_face.pth")
WIDERFACE_PATH = os.path.join(BASE_DIR, "models", "fasterrcnn_widerface.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using Device: {device}")

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Could not find image at {IMAGE_PATH}")
    exit(1)

# ==============================
# 1. FACENET + SVM
# ==============================
def test_facenet(image):
    print("\n⏳ Testing FaceNet + SVM...")
    frame = image.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    classifier = joblib.load(SVM_MODEL_PATH)
    
    boxes, probs = mtcnn.detect(img_rgb)
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob < 0.8: continue
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0: continue
            
            face_pil = Image.fromarray(face_crop).resize((160, 160))
            face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = resnet(face_tensor).cpu().numpy()
            
            prediction = classifier.predict(embedding)
            prob_svm = classifier.predict_proba(embedding)
            max_prob = np.max(prob_svm)
            
            if max_prob < 0.6:
                label = "Unknown"
                color = (0, 0, 255)
            else:
                label = prediction[0]
                color = (0, 255, 0)
                
            text = f"{label} ({max_prob*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imwrite(os.path.join(BASE_DIR, "test_outputs", "output_facenet.jpg"), frame)
    print("✅ Result saved to test_outputs/output_facenet.jpg")

# ==============================
# 2. MOBILENET V2
# ==============================
def test_mobilenet(image):
    print("\n⏳ Testing MobileNetV2...")
    frame = image.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mtcnn = MTCNN(keep_all=True, device=device)
    checkpoint = torch.load(MOBILENET_PATH, map_location=device)
    class_names = checkpoint['class_names']

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    boxes, probs = mtcnn.detect(img_rgb)
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob < 0.8: continue
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0: continue
            
            pil_image = Image.fromarray(face_crop)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, 1)

            max_prob = max_prob.item()
            predicted_label = class_names[predicted_idx.item()]
            
            if max_prob < 0.7:
                label = "Unknown"
                color = (0, 0, 255)
            else:
                label = predicted_label
                color = (0, 255, 0)
                
            text = f"{label} ({max_prob*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(os.path.join(BASE_DIR, "test_outputs", "output_mobilenet.jpg"), frame)
    print("✅ Result saved to test_outputs/output_mobilenet.jpg")

# ==============================
# 3. FASTER R-CNN (WIDER FACE)
# ==============================
def test_fasterrcnn(image):
    print("\n⏳ Testing Custom Faster R-CNN (WIDER FACE)...")
    frame = image.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(WIDERFACE_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = model([input_tensor])[0]
        
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    CONFIDENCE_THRESHOLD = 0.70
    for i in range(len(boxes)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = [int(coord) for coord in boxes[i]]
            text = f"Face ({scores[i]*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
    cv2.imwrite(os.path.join(BASE_DIR, "test_outputs", "output_fasterrcnn.jpg"), frame)
    print("✅ Result saved to test_outputs/output_fasterrcnn.jpg")


if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ OpenCV Error: Could not read {IMAGE_PATH}")
    else:
        test_facenet(img)
        test_mobilenet(img)
        test_fasterrcnn(img)
        print("\n🎉 All 3 Tests completed. Check your folder for the output images!")
