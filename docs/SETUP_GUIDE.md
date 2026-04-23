# Face Recognition Attendance System - Setup & Execution Guide

This guide provides step-by-step instructions for setting up the environment, collecting the image dataset, and training the embedding model for the Face Recognition system.

---

## 1. Environment Setup

This project uses **two different Conda environments**:
1. `venv`: Used for dataset collection and general CPU tasks.
2. `face_gpu`: Used for GPU-accelerated training and inference.

### Step 1.1: Create the Environments
Open your Anaconda Prompt to create and configure both environments:

**Data Collection Environment (`venv`)**
```bash
conda create -n venv python=3.9 -y
conda activate venv
pip install -r requirements_venv.txt
```

**Training & Inference Environment (`face_gpu`)**
```bash
conda create -n face_gpu python=3.9 -y
conda activate face_gpu
pip install -r requirements_face_gpu.txt
```
> [!NOTE] 
> You can verify your GPU status in the `face_gpu` environment at any time by running `python utils\check_gpu.py`.

---

## 2. Data Collection

The system relies on a dataset of faces collected via webcam. **Make sure you have activated the `venv` environment for this phase.**

### Step 2.1: Run the Collection Notebook
1. Activate the environment and start Jupyter:
   ```bash
   conda activate venv
   jupyter lab
   ```
2. Navigate to `notebooks\dataset_collection.ipynb`.
3. Run the cells to initialize your webcam. The script uses OpenCV and Haar Cascades to detect your face and save cropped face images directly into the `dataset\` directory.

### Step 2.2: Data Augmentation (Recommended)
To improve the robustness of the trained model, you can augment your dataset (applying rotations, flips, and brightness changes).
You can either:
- Run the augmentation notebook: `notebooks\augmentation.ipynb`
- Or run the standalone script:
  ```bash
  python utils\augment_data.py
  ```
The augmented images will be stored in the `augmented_dataset\` directory.

---

## 3. Training the Embedding Model

Once you have gathered and augmented your dataset, you are ready to train the face embedding model. **Make sure you have activated the `face_gpu` environment for training.**

### Step 3.1: Start GPU Training
We use a GPU-accelerated script to speed up training. Ensure your processed dataset is in place, then run:
```bash
conda activate face_gpu
python utils\train_gpu.py
```
> [!IMPORTANT]
> The script will read images from `dataset\` and `augmented_dataset\`. The trained model artifacts will automatically be saved into the `models\` directory.

### Step 3.2: Evaluate the Model
After training, check your model's accuracy and performance metrics. 
- You can run the evaluation script:
  ```bash
  python utils\evaluate_model.py
  ```
- This will generate visual reports, including the `confusion_matrix.png` found in the root directory.

---

## 4. Running Inference

To test the trained model on live webcam data or new static images, ensure you are using the GPU environment:
```bash
conda activate face_gpu
python run_inference.py
```
The script will load the saved model from the `models\` directory and output real-time predictions.
