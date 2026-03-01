# 🩺 Diabetic Retinopathy Detection using Deep Learning

A deep learning-based web application that detects the stage of **Diabetic Retinopathy (DR)** from retinal fundus images using a fine-tuned **MobileNetV2** model. The application is deployed using **Streamlit Cloud** for real-time predictions.

---

## 🚀 Live Demo

🔗 https://your-streamlit-app-link.streamlit.app  


---

## 📌 Problem Statement

Diabetic Retinopathy is a serious diabetes-related eye condition that can lead to vision loss if not detected early.

This project aims to:

- Classify retinal images into 5 DR stages
- Provide instant predictions through a web interface
- Demonstrate real-world deployment of a deep learning model

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (ImageNet pretrained)
- Transfer Learning (Feature Extraction)
- Global Average Pooling Layer
- Fully Connected Dense Layers
- Dropout Regularization
- Softmax Output Layer (5 Classes)

### 📊 Classification Categories

1. Mild
2. Moderate
3. No_DR
4. Proliferate_DR
5. Severe

---

## 🏗 Tech Stack

### 🔹 Model Development
- Python
- TensorFlow / Keras
- NumPy
- PIL

### 🔹 Deployment
- Streamlit
- Streamlit Cloud
- Google Drive (Model Hosting via gdown)

---
## 📂 Project Structure

```
Diabetic-Retinopathy-Detection/
│
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version configuration
└── README.md           # Project documentation
```


---

## ⚙️ How the Application Works

1. User uploads a retinal image (.jpg/.png/.jpeg)
2. Image is resized to 224 × 224 pixels
3. Preprocessing is applied
4. Model generates prediction probabilities
5. Highest confidence class is displayed

---

## ▶️ Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mukeshshah08/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
