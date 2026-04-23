import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from sklearn.metrics import auc

# ==============================
# 🔷 PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Handle double-nested WIDER_val directory
VAL_IMAGES_DIR = os.path.join(BASE_DIR, "archive", "WIDER_val", "WIDER_val", "images")
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, "archive", "WIDER_val", "images")

ANNOTATION_FILE = os.path.join(BASE_DIR, "archive", "wider_face_annotations", "wider_face_split", "wider_face_val_bbx_gt.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fasterrcnn_widerface.pth")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Settings
MAX_IMAGES = 800  # Evaluates 800 images to prevent the computer from running for hours
IOU_THRESHOLD = 0.50

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union between two bounding boxes"""
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    # Compute union area
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)

    # Calculate IoU
    return interArea / unionArea

def main():
    print("🔥 Initializing Mathematical Evaluation Engine...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using Processing Device: {device}")

    # ==============================
    # 1. PARSE GROUND TRUTH ANNOTATIONS
    # ==============================
    print("🔄 Parsing Ground-Truth Annotations...")
    ground_truths = {}
    total_gt_faces = 0

    if not os.path.exists(ANNOTATION_FILE):
        print(f"❌ Error: Could not find Annotation file at {ANNOTATION_FILE}")
        print("Please ensure the 'wider_face_split' folder is extracted inside your archive map.")
        return

    with open(ANNOTATION_FILE, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        filename = lines[i].strip()
        i += 1
        if not filename.endswith('.jpg'):
            continue
            
        try:
            num_boxes = int(lines[i].strip())
        except ValueError:
            num_boxes = 0
            
        i += 1
        boxes = []
        if num_boxes == 0:
            i += 1
        else:
            for _ in range(num_boxes):
                parts = list(map(int, lines[i].split()[:4]))
                x1, y1, w, h = parts
                x2, y2 = x1 + w, y1 + h
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                i += 1
                
        if len(boxes) > 0:
            ground_truths[filename] = {"boxes": boxes, "used": [False] * len(boxes)}
            total_gt_faces += len(boxes)

        if len(ground_truths) >= MAX_IMAGES:
            break

    print(f"✅ Loaded {len(ground_truths)} valid images with {total_gt_faces} target faces to find.")

    # ==============================
    # 2. LOAD NETWORK DICTIONARY
    # ==============================
    print("⏳ Loading Custom WIDER FACE Model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])

    # ==============================
    # 3. RUN INFERENCE & GATHER PREDICTIONS
    # ==============================
    all_predictions = []  # List of dicts: {'confidence': float, 'img_name': str, 'box': list}
    processed_count = 0

    print("🚀 Starting Matrix Verification Loop (This will take a few minutes)...")
    for filename, gt_data in ground_truths.items():
        img_path = os.path.join(VAL_IMAGES_DIR, *filename.split('/'))
        if not os.path.exists(img_path):
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = model([input_tensor])[0]

        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for b_idx in range(len(boxes)):
            all_predictions.append({
                'confidence': scores[b_idx],
                'img_name': filename,
                'box': boxes[b_idx].tolist()
            })

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"   Mathematically Verified Images: [{processed_count}/{len(ground_truths)}]")

    # ==============================
    # 4. CALCULATE TRUE/FALSE POSITIVES
    # ==============================
    print("\n🧠 Computing Intersection-Over-Union (IoU) Precision Matrix...")
    # Sort predictions by confidence so we test most confident boxes first natively
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

    TP = np.zeros(len(all_predictions))
    FP = np.zeros(len(all_predictions))

    for idx, pred in enumerate(all_predictions):
        filename = pred['img_name']
        pred_box = pred['box']
        
        gt_boxes = ground_truths[filename]["boxes"]
        gt_used = ground_truths[filename]["used"]

        best_iou = 0.0
        best_gt_idx = -1

        # Check this prediction against all Ground Truth boxes inside this specific image
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Mathematically Score it (Must be over 50% overlap to pass industry standard)
        if best_iou >= IOU_THRESHOLD:
            if not gt_used[best_gt_idx]:
                TP[idx] = 1
                gt_used[best_gt_idx] = True # Mark it so another box can't double-score
            else:
                FP[idx] = 1 # Multiple predictions hit the exact same face (penalty)
        else:
            FP[idx] = 1 # Box predicted nothing but background mathematically

    # Cumulative Sums for curve mapping
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)

    recalls = acc_TP / total_gt_faces
    precisions = np.divide(acc_TP, (acc_FP + acc_TP), out=np.zeros_like(acc_TP), where=(acc_FP + acc_TP)!=0)

    # Mean Average Precision Area
    mAP = auc(recalls, precisions) * 100

    print("\n" + "="*40)
    print("📈 FINAL REPORT: FASTER R-CNN RESULTS")
    print("="*40)
    print(f"Total Images Evaluated: {processed_count}")
    print(f"Ground Truth Faces Mapped: {total_gt_faces}")
    print(f"Total Network Predictions: {len(all_predictions)}")
    print(f"IoU Accuracy Standard: {IOU_THRESHOLD*100}% Overlap")
    print("-" * 40)
    print(f"🥇 Mean Average Precision (mAP@0.5): {mAP:.1f}%")
    print(f"   Max Recall Achieved: {recalls[-1]*100:.1f}%")
    print("="*40)

    # ==============================
    # 5. GENERATE PRECISION-RECALL CURVE
    # ==============================
    print("📊 Plotting Precision-Recall Curve to disk...")
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f'Model Performance (mAP: {mAP:.1f}%)', color='darkblue', linewidth=2)
    plt.fill_between(recalls, precisions, color='blue', alpha=0.1)
    plt.title('Precision-Recall Curve (WIDER FACE Evaluation)', fontsize=14, fontweight='bold')
    plt.xlabel('Recall \n(Percentage of All Faces Found)', fontsize=12)
    plt.ylabel('Precision \n(Accuracy of Placed Boxes)', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='lower left')
    
    save_path = os.path.join(PLOT_DIR, "widerface_pr_curve.png")
    plt.savefig(save_path)
    print(f"🎉 Success! Final metric graph exported to: {save_path}")

if __name__ == "__main__":
    main()
