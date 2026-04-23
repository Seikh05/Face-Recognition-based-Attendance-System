import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
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
SVM_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_model_gpu.pkl")

RECOGNITION_THRESHOLD = 0.60 

# ==============================
# STATE MANAGEMENT
# ==============================
current_period = "Period 1"
attendance_records = {} # Format: {'John Doe': '08:30:00'}

# ==============================
# GLOBALS FOR MODELS (RPi OPTIMIZED)
# ==============================
# For RPi, this will automatically fall back to CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

mtcnn = None
resnet = None
svm_classifier = None

def init_models():
    global mtcnn, resnet, svm_classifier
    print(f"Loading Edge-Optimized Models on {device}...")
    
    # 🧠 Brain 1: Detector (MTCNN) 
    # Lightweight, built for CPUs/Edge Devices, replaces Faster-RCNN.
    # keep_all=True allows it to detect multiple faces in one frame.
    mtcnn = MTCNN(keep_all=True, device=device)

    # 🧠 Brain 2: Recognizer (FaceNet + SVM)
    if os.path.exists(SVM_MODEL_PATH):
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        svm_classifier = joblib.load(SVM_MODEL_PATH)
        print("✅ Edge Models Loaded.")
    else:
        print(f"❌ Error: Missing Classification Model at {SVM_MODEL_PATH}")

# Call immediately
init_models()


# ==============================
# 🚀 ANTI-LATENCY CAMERA THREAD
# ==============================
class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # RPi Linux environment frame buffer reduction
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        # Optimize resolution for RPi to save CPU cycles
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        if self.cap.isOpened():
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

camera_stream = CameraStream()

# ==============================
# VIDEO STREAM GENERATOR
# ==============================
def gen_frames():
    while True:
        success, frame = camera_stream.read()
        if not success:
            continue
            
        if mtcnn and svm_classifier:
            # OpenCV captures BGR, MTCNN expects RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # 1. Detect bounding boxes using lightweight MTCNN
            boxes, probs = mtcnn.detect(pil_img)

            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Box format: [x1, y1, x2, y2]
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    
                    # Prevent outbound array slicing
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    face_crop = img_rgb[y1:y2, x1:x2]
                    
                    # Failsafe verify the crop actually exists
                    if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                        continue
                        
                    # 2. Recognition Block
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

                    # Draw bounds
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, display_text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Re-encode and stream to Flask Webpage
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
    # Threaded=True prevents UI hanging while sending video chunks
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
