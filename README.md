 🫁 Lung Cancer Detection System
 Dual-Mode System — Symptom Analysis + CT Scan Classification
 🔗 GitHub Repository
https://github.com/Vishruti1309/lung-cancer-detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.6.1-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning web application that detects lung cancer risk using two independent modes:
1. **Symptom-based prediction** using a Random Forest classifier
2. **CT Scan image classification** using MobileNetV2 deep learning model

Built as a Semester IV project — School of Engineering, JG University.

---

 📌 Table of Contents
- [Project Description]
- [Features]
- [Tech Stack]
- [Dataset]
- [Model Performance]
- [Project Structure]
- [How to Run]
- [Screenshots]
- [Key Concepts]
- [Future Work]
- [Team]

---

 📖 Project Description

Lung cancer is responsible for 18% of all cancer deaths worldwide. Symptoms appear very late, and diagnostic tools like CT scans and biopsies are expensive and inaccessible for many patients.

This project provides a **free, accessible, AI-powered dual screening tool**:

- **Mode 1 — Symptom Check:** Patient fills a form with 15 clinical symptoms. A trained Random Forest model predicts HIGH RISK or LOW RISK with a probability percentage.
- **Mode 2 — CT Scan Analysis:** A doctor or patient uploads a CT scan image. A MobileNetV2 CNN model classifies it as Malignant, Benign, or Normal with confidence scores.

> ⚠️ **Disclaimer:** This tool is for educational and screening purposes only. It is not a substitute for professional medical diagnosis.

---

 ✨ Features

- 🔬 Dual-mode detection — symptoms + CT scan image
- 📊 Probability percentage shown for symptom prediction
- 🖼️ CT scan image upload with live preview
- 📈 Confidence scores for all 3 classes (Malignant / Benign / Normal)
- 🎨 Color-coded results — Red (High Risk), Orange (Medium), Green (Low Risk)
- 🔁 Auto tab switch to CT scan results after image prediction
- ✅ Class imbalance handled with SMOTE
- ✅ Keras version mismatch handled by saving weights only

---

 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10 |
| Web Framework | Flask 3.0.0 |
| ML Model | Scikit-learn (Random Forest) |
| Deep Learning | TensorFlow 2.16.1 / Keras |
| CNN Architecture | MobileNetV2 (Transfer Learning) |
| Data Balancing | SMOTE (imbalanced-learn) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML, CSS, Jinja2 |
| Training Platform | Google Colab (T4 GPU) |
| Model Saving | Joblib (.pkl), HDF5 (.h5) |

---

 📂 Dataset

 Dataset 1 — Symptom Survey
- **Source:** [Kaggle — Lung Cancer Survey](https://www.kaggle.com/)
- **Records:** 309 patients, 16 columns
- **Features:** 15 symptoms (Age, Smoking, Chest Pain, Yellow Fingers, etc.)
- **Target:** LUNG_CANCER (YES / NO)
- **Class Distribution:** Cancer=270 (87.4%), No Cancer=39 (12.6%)

 Dataset 2 — CT Scan Images
- **Source:** [IQ-OTH/NCCD Lung Cancer Dataset — Kaggle](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)
- **Total Images:** 1,097 CT scans
- **Classes:** Malignant=561, Normal=416, Benign=120

---

 📊 Model Performance

 Symptom Model — Random Forest

| Model | Accuracy | Cancer Recall | No Cancer Recall |
|---|---|---|---|
| Logistic Regression | 90.32% | 94% | 62% |
| **Random Forest (Final)** | **91.93%** | **94%** | **75%** |

 CNN Model — MobileNetV2

| Metric | Value |
|---|---|
| Final Train Accuracy | 85% |
| Final Validation Accuracy | 68% |

 CNN Prediction Results

| CT Scan | Predicted | Confidence |
|---|---|---|
| Malignant | Malignant | 94.8% ✅ |
| Normal | Normal | 60.6% ✅ |
| Benign | Benign | 56.3% ✅ |

---

 🗂️ Project Structure

 lung_cancer_project/
├── dataset/
│   └── survey_lung_cancer.csv
├── notebooks/
│   ├── Copy_of_Lung_Cancer_detection__1_.ipynb   ← Symptom ML model
│   └── CNN_Cancer_Detection.ipynb                ← CNN image model
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── cnn_weights.weights.h5
├── app/
│   ├── app.py
│   ├── uploads/
│   └── templates/
│       └── index.html
├── outputs/
├── screenshots/
├── README.md
└── requirements.txt




---

▶️ How to Run

1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/lung-cancer-detection.git
cd lung-cancer-detection
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Add Model Files
Place these files inside the `app/` folder:
- `model.pkl`
- `scaler.pkl`
- `cnn_weights.weights.h5`

4. Run the Flask App
```bash
cd app
python app.py
```

5. Open in Browser
http://127.0.0.1:5000

> ⚠️ Never open `index.html` directly. Always run through Flask at `http://127.0.0.1:5000`

---

📸 Screenshots

Symptom Check Tab
> Add screenshot: `screenshots/symptom_form.png`
> *(Show the filled form with all 15 symptoms selected)*

Symptom Prediction Result — HIGH RISK
> Add screenshot: `screenshots/symptom_result_high.png`
> *(Show red HIGH RISK result with probability %)*

CT Scan Upload Tab
> Add screenshot: `screenshots/ct_upload.png`
> *(Show the upload area with image preview)*

CT Scan Result — Malignant
> Add screenshot: `screenshots/ct_result_malignant.png`
> *(Show MALIGNANT HIGH RISK result with confidence bars for all 3 classes)*

CT Scan Result — Normal
> Add screenshot: `screenshots/ct_result_normal.png`

---

**📌 What screenshots to actually take:**
1. Home page — symptom tab (empty form)
2. Symptom tab — filled form before clicking predict
3. Symptom result — HIGH RISK output (red)
4. Symptom result — LOW RISK output (green)
5. CT scan tab — after uploading a malignant CT image (shows preview)
6. CT scan result — MALIGNANT with all 3 probability bars visible
7. CT scan result — NORMAL output

---

🧠 Key Concepts

SMOTE (Synthetic Minority Oversampling Technique)
The dataset had severe class imbalance — 270 Cancer vs only 39 No Cancer cases. Without fixing this, the model would always predict Cancer. SMOTE generates synthetic samples of the minority class, balancing training to 216 vs 216.

Transfer Learning
With only 1,097 CT scan images, training a CNN from scratch would overfit badly. MobileNetV2, pretrained on 14 million ImageNet images, was used with frozen base layers. Only the custom classification head was trained for our 3 classes.

Recall over Accuracy
In cancer detection, a false negative (missing a real cancer case) is far more dangerous than a false positive. So we optimize for **Recall** — our model achieves **94% Cancer Recall**.

Keras Version Mismatch Fix
Training was done in Google Colab (Keras 3.x). To avoid version conflicts on local machines, model weights are saved as `.h5` and the architecture is rebuilt inside `app.py` before loading weights.

---

🔮 Future Work

- Fine-tune MobileNetV2 by unfreezing top layers for better CNN accuracy
- Collect more Benign CT scan images to improve that class performance
- Add Grad-CAM heatmap visualization to show which lung regions the model focused on
- Deploy on cloud platform (Render / Railway / Hugging Face Spaces)
- Add user login and patient history tracking
- Add DICOM format support for real hospital CT scans


---

📄 License

This project is for academic purposes only.
