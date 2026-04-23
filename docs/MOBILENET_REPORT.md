# Deep Learning Face Recognition Report: MobileNetV2 Transfer Learning

## 1. Introduction
This report details the implementation of an end-to-end Deep Learning approach for Face Recognition. Instead of relying on traditional feature extraction coupled with a standard Machine Learning classifier (e.g., FaceNet + SVM), this project explores **Transfer Learning** using a state-of-the-art Convolutional Neural Network (CNN).

By leveraging a pre-trained CNN, the model learns facial features and classifies the identity directly within a single, unified architecture. This was implemented to compare generalization capabilities and robustness against the classic embedding approach.

---

## 2. Model Architecture
### MobileNetV2 with Transfer Learning
For this experiment, **MobileNetV2** was selected as the core architecture. MobileNetV2 is specifically designed to be lightweight, incredibly fast, and computationally efficient while maintaining high accuracy, making it ideal for real-time webcam inference.

**Modifications Made:**
- **Base Model:** We initialized the network with weights pre-trained on ImageNet (`MobileNet_V2_Weights.IMAGENET1K_V1`), giving the network a foundational understanding of shapes, edges, and textures.
- **Custom Classifier:** The final dense classification layer of MobileNetV2 was removed. In its place, a new Fully Connected (Linear) layer was injected, matching its output nodes to our specific number of identities (classes in the dataset).

---

## 3. Training Setup and Hyperparameters
The dataset consisted of face samples (augmented with rotations, brightness shifts, noise, etc.) divided rigorously to ensure no data leakage.

* **Dataset Split:** 70% Training / 15% Validation / 15% Test
* **Data Augmentation (Train Phase):** Random Horizontal Flips, Resizing to `224x224`, and standard ImageNet Normalization.
* **Epochs:** 10
* **Batch Size:** 32
* **Optimizer:** Adam Optimizer (Learning Rate = `0.001`)
* **Loss Function:** Categorical Cross-Entropy Loss
* **Hardware:** CUDA (GPU)

---

## 4. Performance Metrics

The model was evaluated exclusively on the **15% unseen Test Split** to guarantee an objective measure of real-world performance. The test set contained 1,621 unseen images.

### Overall Macro Metrics
* **Test Accuracy:** 99.88%
* **Macro Average Precision:** 99.88%
* **Macro Average Recall:** 99.88%
* **Macro Average F1-Score:** 99.88%

### Per-Class Breakdown

| Identity  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Akhilesh  | 1.0000    | 1.0000 | 1.0000   | 267     |
| Ashutosh  | 0.9962    | 1.0000 | 0.9981   | 264     |
| Babul     | 0.9964    | 0.9964 | 0.9964   | 276     |
| Mustakim  | 1.0000    | 1.0000 | 1.0000   | 248     |
| Srikanta  | 1.0000    | 0.9965 | 0.9982   | 284     |
| arpit     | 1.0000    | 1.0000 | 1.0000   | 282     |

---

## 5. Graphs and Visualizations

Visualizing the training progression confirms that our Transfer Learning effectively adapted to the face dataset without severely overfitting or underfitting.

### 5.1 Training Progression (Loss & Accuracy vs Epoch)
The graphs below plot how the Cross-Entropy Loss decreased and Accuracy scaled during the 10 epochs.
*(Convergence between the Train and Validation lines indicates robust learning).*

![Training Metrics](plots/mobilenet_training_metrics.png)

### 5.2 Confusion Matrix
The Confusion Matrix maps exactly where the model succeeded and where it became confused on the unseen test dataset. A strong diagonal represents perfect classification.

![Confusion Matrix](plots/mobilenet_confusion_matrix.png)

---

## 6. Conclusion and Viva Takeaway
**Viva Answer / Summary Statement:**
*"We implemented both a traditional embedding-based approach (FaceNet + SVM) and an end-to-end deep learning framework. For the deep learning framework, we built a CNN model utilizing Transfer Learning on the MobileNetV2 architecture. This allowed us to achieve excellent generalization with our limited data without having to train millions of parameters entirely from scratch. The MobileNetV2 architecture demonstrated extraordinary performance, correctly classifying identities with a near-perfect test accuracy of 99.88% on 1,621 unseen images."*
