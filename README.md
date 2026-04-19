<div align="center">
  <h1>🛡️ IDS MANET Security System</h1>
  <h3>Adaptive Voting Mechanism with Artificial Butterfly Algorithm for Intrusion Detection</h3>
  
  <p>
    <b>A professional-grade, machine learning-powered security analysis tool designed for Mobile Ad Hoc Networks (MANETs).</b>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Framework-Flask-black.svg?logo=flask" alt="Flask">
    <img src="https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg" alt="Machine Learning">
    <img src="https://img.shields.io/badge/UI-Glassmorphism-purple.svg" alt="UI Design">
  </p>
</div>

---

## 🌟 Project Overview

Mobile Ad Hoc Networks (MANETs) consist of mobile devices communicating dynamically without centralized infrastructure. This flexibility makes them highly vulnerable to various cyber attacks. Traditional security tools often fail to adapt to these dynamic topologies. 

This project introduces a proactive, highly accurate **Intrusion Detection System (IDS)** that utilizes cutting-edge ensemble machine learning and bio-inspired optimization algorithms to detect anomalous network traffic in real-time.

---

## ✨ Key Innovations

1. **Artificial Butterfly Optimizer (ABO):** A bio-inspired feature selection algorithm that mimics the mating and foraging behaviors of butterflies. It drastically reduces the dimensionality of the NSL-KDD dataset (selecting the optimal subset of features), thereby accelerating model inference without sacrificing accuracy.
2. **Adaptive Voting Ensemble:** Instead of relying on a single classifier, this system utilizes 7 distinct machine learning models (Decision Tree, Random Forest, XGBoost, SVM, Logistic Regression, MLP, Boosted Regression Tree). An adaptive voting mechanism dynamically assigns weights to each model based on their training accuracy, yielding an ensemble prediction that outperforms any individual model.

---

## 💻 Web Application Modules

The project features a premium, responsive **Glassmorphism** web interface with the following core modules:

*   📊 **Analytics Dashboard:** A comprehensive view of model performance comparing accuracy, training times, ROC-AUC curves, feature importance, and granular classification reports (Precision, Recall, F1).
*   📂 **Batch CSV Processing:** Upload bulk network traffic data (.csv) for automated, large-scale threat detection. The system parses, scales, and runs the ensemble model on all rows simultaneously.
*   🕒 **Audit History Log:** An integrated SQLite database automatically persists all predictions (both single and batch), providing a searchable audit trail of past network scans and identified threats.
*   🔍 **Dataset Explorer:** Interactive Exploratory Data Analysis (EDA) visualizations showing class distributions, feature histograms, and correlation heatmaps of the ABO-selected features.
*   📑 **One-Click Reporting:** Instantly generate and download professional PDF/HTML reports summarizing system metrics for academic submission or stakeholder review.

---

## 🛠️ Technology Stack

| Category | Technologies Used |
| :--- | :--- |
| **Backend** | Python 3, Flask, SQLite3 |
| **Machine Learning** | Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE) |
| **Data Processing** | Pandas, NumPy, Matplotlib, Seaborn |
| **Frontend UI** | HTML5, CSS3 (Custom Glassmorphism), Bootstrap 5, Chart.js, Bootstrap Icons |

---

## 🚀 Step-by-Step Setup Guide

Follow these instructions to configure and run the project on your local machine.

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### 2. Extract and Prepare
Extract the project archive. Ensure that `Train_data.csv` and `Test_data.csv` are located in the root directory alongside `app.py`.

### 3. Install Dependencies
Open your terminal (or Command Prompt / PowerShell) and navigate to the project folder. Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 4. Execute the Training Pipeline
Before running the web application, you **must** train the machine learning models. This script will apply SMOTE, run the ABO feature selection, train all 7 classifiers, calculate the adaptive weights, and generate the necessary visualization artifacts.
*(Note: Depending on your CPU, training the SVM and Random Forest models may take a few minutes).*

```bash
python train_model.py
```
*You should see a "Training complete!" message when it finishes.*

### 5. Start the Web Server
Launch the Flask backend server:
```bash
python app.py
```

### 6. Access the Application
Open your web browser and navigate to the local development server:
👉 **`http://127.0.0.1:5000`**

---

## 📂 Directory Structure

```text
AVM/
│
├── train_model.py              # Core ML Pipeline (ABO, Training, Evaluation)
├── app.py                      # Flask Application Server & Routing
├── butterfly_optimizer.py      # ABO Algorithm Implementation
├── inspect_data.py             # Data sanity check utility
├── requirements.txt            # Dependency list
├── signup.db                   # SQLite Database (Users & Prediction History)
│
├── Train_data.csv              # NSL-KDD Training Dataset
├── Test_data.csv               # NSL-KDD Testing Dataset
│
├── models/                     # Auto-generated serialized models (.pkl) & results (.json)
│
├── templates/                  # Frontend HTML Views
│   ├── base.html               # Master layout & Navigation
│   ├── home.html               # Landing page & Quick stats
│   ├── dashboard.html          # Analytics & Metrics visualization
│   ├── batch.html              # CSV upload interface
│   ├── dataset.html            # EDA Explorer
│   ├── history.html            # Audit logging table
│   └── login.html / signup.html
│
└── static/                     # Static Assets
    ├── css/style.css           # Custom Glassmorphism UI styles
    └── images/                 # Auto-generated metric plots (ROC, Heatmaps, etc.)
```

---

## 📊 Dataset & Preprocessing

The system evaluates threats using the **NSL-KDD Dataset**. Because real-world network traffic is heavily imbalanced (safe traffic vastly outnumbers attack traffic), the pipeline applies **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the classes. Continuous features are standardized using **Z-score normalization** before being passed to the classifiers, ensuring optimal performance across distance-based algorithms like SVMs and Neural Networks.
