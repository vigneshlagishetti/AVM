# Adaptive Voting Mechanism with Artificial Butterfly Algorithm for IDS in MANET

An advanced, professional-grade Intrusion Detection System (IDS) designed specifically for Mobile Ad Hoc Networks (MANETs). This project addresses the complex security vulnerabilities inherent in dynamic MANET topologies by utilizing cutting-edge machine learning and optimization techniques.

## 🚀 Key Features

*   **Artificial Butterfly Optimizer (ABO):** A bio-inspired feature selection algorithm that optimizes the NSL-KDD dataset by extracting only the most critical network traffic features, reducing dimensionality and improving model efficiency.
*   **Adaptive Voting Ensemble:** A dynamic ensemble of 7 machine learning classifiers (Decision Tree, Random Forest, XGBoost, SVM, Logistic Regression, MLP, Boosted Regression Tree). The system dynamically weights the predictions of each model based on their individual accuracy.
*   **Premium Analytics Dashboard:** A stunning, responsive glassmorphism UI featuring detailed model performance metrics, ROC-AUC curves, feature importance charts, and training time comparisons.
*   **Batch CSV Processing:** Upload bulk network traffic data via CSV for automated, large-scale threat detection and analysis.
*   **Dataset EDA Explorer:** Interactive visualizations exploring class distributions and feature correlation heatmaps.
*   **Audit History Logging:** A persistent SQLite database logs all predictions, providing a comprehensive audit trail of past network scans and threat detections.
*   **One-Click Reporting:** Instantly generate and download professional PDF/HTML reports summarizing the model's performance and optimal features for academic submission.

## 🛠️ Technology Stack

*   **Backend:** Python 3, Flask, SQLite
*   **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
*   **Frontend:** HTML5, Vanilla CSS (Glassmorphism design), Bootstrap 5, Chart.js
*   **Data Processing:** Pandas, NumPy, Matplotlib, Seaborn

## 📂 Project Structure

```text
AVM/
├── models/                     # Saved serialized models, scalers, and encoders (.pkl)
├── static/
│   ├── css/style.css           # Premium dark theme and glassmorphism styling
│   └── images/                 # Generated EDA charts and model performance plots
├── templates/                  # HTML templates for the Flask application
├── Train_data.csv              # NSL-KDD Training Dataset
├── Test_data.csv               # NSL-KDD Testing Dataset
├── app.py                      # Flask web application and routing
├── train_model.py              # ML Pipeline: Preprocessing, ABO Selection, Training, Evaluation
├── butterfly_optimizer.py      # Implementation of the Artificial Butterfly Algorithm
├── inspect_data.py             # Utility script for initial data inspection
└── requirements.txt            # Python dependencies
```

## ⚙️ Setup and Installation

1.  **Extract the project:** Ensure all files (including `Train_data.csv` and `Test_data.csv`) are in the project directory.
2.  **Install Dependencies:**
    Open your terminal or command prompt in the project directory and run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Training Pipeline:**
    Before starting the web app, you must train the models and generate the required visualizations. This may take a few minutes depending on your hardware.
    ```bash
    python train_model.py
    ```
4.  **Start the Web Application:**
    Once training is complete, start the Flask server:
    ```bash
    python app.py
    ```
5.  **Access the Dashboard:**
    Open your web browser and navigate to: `http://127.0.0.1:5000`

## 📊 Dataset Information

This project utilizes the **NSL-KDD Dataset**, a refined version of the KDD'99 dataset, widely recognized as a benchmark for evaluating intrusion detection systems. The data is preprocessed using Z-score normalization and SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalances before being passed to the ABO feature selection algorithm.
