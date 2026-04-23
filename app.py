import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import os
import numpy as np
import joblib
import threading
from flask import Flask, render_template, Response, request, jsonify
from datetime import datetime

app = Flask(__name__)

# ==============================
# 🔷 PATH SETUP & CONSTRAINTS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(BASE_DIR, "models", "fasterrcnn_widerface.pth")
SVM_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_model_gpu.pkl")

DETECTION_THRESHOLD = 0.65  
RECOGNITION_THRESHOLD = 0.60 

# ==============================
# STATE MANAGEMENT
# ==============================
current_period = "Period 1"
attendance_records = {} # Format: {'John Doe': '08:30:00'}

# ==============================
# GLOBALS FOR MODELS
# ==============================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_det = None
resnet = None
svm_classifier = None
transform_det = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def init_models():
    global model_det, resnet, svm_classifier
    print(f"Loading Models on {device}...")
    
    # 🧠 Brain 1: Detector
    if os.path.exists(DETECTOR_PATH):
        model_det = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model_det.roi_heads.box_predictor.cls_score.in_features
        model_det.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model_det.load_state_dict(torch.load(DETECTOR_PATH, map_location=device))
        model_det.to(device)
        model_det.eval()
    else:
        print(f"Error: Missing {DETECTOR_PATH}")

    # 🧠 Brain 2: Recognizer
    if os.path.exists(SVM_MODEL_PATH):
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        svm_classifier = joblib.load(SVM_MODEL_PATH)
    else:
        print(f"Error: Missing {SVM_MODEL_PATH}")

# Call immediately
init_models()


# ==============================
# 🚀 ANTI-LATENCY CAMERA THREAD
# ==============================
# OpenCV notoriously buffers frames if the AI is slower than the camera 30fps.
# This causes the video feed to eventually lag seconds behind.
# We fix this by having a background thread constantly empty the buffer and solely save the absolute LATEST frame.

class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Try to force hardware to 0 buffer (only works on some backends)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        if self.cap.isOpened():
            # Start background thread immediately
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
        else:
            print("❌ CRITICAL ERROR: Could not map to webcam hardware.")

    def update(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.latest_frame = frame

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# Initialize global hardware stream
camera_stream = CameraStream()

# ==============================
# VIDEO STREAM GENERATOR
# ==============================
def gen_frames():
    while True:
        success, frame = camera_stream.read()
        if not success:
            continue
            
        if model_det and svm_classifier:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor_det = transform_det(img_rgb).to(device)
            
            with torch.no_grad():
                prediction = model_det([input_tensor_det])[0]
            
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()

            for i in range(len(boxes)):
                if scores[i] >= DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = [int(coord) for coord in boxes[i]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    face_crop = img_rgb[y1:y2, x1:x2]
                    if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                        continue
                        
                    face_pil = Image.fromarray(face_crop).resize((160, 160))
                    face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
                    face_tensor = (face_tensor - 127.5) / 128.0
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        embedding = resnet(face_tensor).cpu().numpy()
                        
                    prediction_label = svm_classifier.predict(embedding)
                    prob_svm = svm_classifier.predict_proba(embedding)
                    max_prob_val = np.max(prob_svm)

                    if max_prob_val < RECOGNITION_THRESHOLD:
                        label = "Unknown Intruder"
                        color = (0, 0, 255) # Red
                        display_text = f"Intruder ({max_prob_val*100:.1f}%)"
                    else:
                        label = prediction_label[0]
                        color = (0, 255, 0) # Green
                        display_text = f"{label} ({max_prob_val*100:.1f}%)"
                        
                        # Log attendance
                        if label not in attendance_records:
                            attendance_records[label] = datetime.now().strftime("%H:%M:%S")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, display_text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify({
        'current_period': current_period,
        'attendance': attendance_records
    })

@app.route('/api/period', methods=['POST'])
def set_period():
    global current_period
    data = request.json
    if 'period' in data:
        current_period = data['period']
        return jsonify({'status': 'success', 'period': current_period})
    return jsonify({'status': 'error', 'message': 'Missing period data'}), 400

@app.route('/api/reset', methods=['POST'])
def reset_attendance():
    global attendance_records
    attendance_records.clear()
    return jsonify({'status': 'success', 'message': 'Attendance reset for ' + current_period})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
