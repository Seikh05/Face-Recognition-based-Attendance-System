import os
import cv2
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

# ==============================
# 🔷 PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_IMAGES_DIR = os.path.join(BASE_DIR, "archive", "WIDER_val", "images")

# Sometimes the archive extraction double layers the foldering.
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, "archive", "WIDER_val", "WIDER_val", "images")

MODEL_PATH = os.path.join(BASE_DIR, "models", "fasterrcnn_widerface.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "bulk_results")

# Automatically generate the output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Settings
NUM_IMAGES_TO_TEST = 100
CONFIDENCE_THRESHOLD = 0.50

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🔥 Using Device: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not locate Face Detector at {MODEL_PATH}")
        return

    # ==============================
    # 🔍 INVENTORY ALL VALIDATION IMAGES
    # ==============================
    print(f"\n📂 Scanning directories in: {VAL_IMAGES_DIR}")
    all_image_paths = []
    
    # os.walk recursively searches through every sub-folder (0--Parade, 1--Handshaking, etc)
    for root, dirs, files in os.walk(VAL_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))
                
    if len(all_image_paths) == 0:
        print("❌ Error: No images found in the WIDER_val directory!")
        return
        
    print(f"✅ Found {len(all_image_paths)} total images in the Validation Archive.")
    
    # Randomly select exact target amount
    selected_images = random.sample(all_image_paths, min(NUM_IMAGES_TO_TEST, len(all_image_paths)))
    print(f"🎯 Randomly selected {len(selected_images)} images for Bulk Processing.")

    # ==============================
    # 🚀 LOAD FASTER R-CNN
    # ==============================
    print("\n⏳ Loading Neural Network Parameters...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])

    # ==============================
    # ⚙️ BULK PROCESSING ENGINE
    # ==============================
    print(f"\n🏎️ Starting high-speed processing sequence. Saving to: {OUTPUT_DIR}")
    
    for i, img_path in enumerate(selected_images):
        filename = os.path.basename(img_path)
        
        # Load Raw Image
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).to(device)

        # Predict
        with torch.no_grad():
            prediction = model([input_tensor])[0]

        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        faces_found = 0
        for b_idx in range(len(boxes)):
            if scores[b_idx] >= CONFIDENCE_THRESHOLD:
                faces_found += 1
                x1, y1, x2, y2 = [int(coord) for coord in boxes[b_idx]]
                
                # Draw Blue Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw Score
                text = f"{scores[b_idx]*100:.0f}%"
                cv2.putText(frame, text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Output to Disk
        save_path = os.path.join(OUTPUT_DIR, f"result_{i+1}_{filename}")
        cv2.imwrite(save_path, frame)
        
        if (i+1) % 10 == 0:
            print(f"   [{i+1}/{len(selected_images)}] Processed {filename} -> Found {faces_found} Faces.")

    print("\n🎉 Bulk Processing Complete! Open the `bulk_results/` folder to visually check how well the model learned!")

if __name__ == "__main__":
    main()
