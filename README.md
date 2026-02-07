# Bank Customer Churn Prediction

This project implements a machine learning pipeline to predict whether a bank customer will churn based on their profile and activity. It compares multiple models (Logistic Regression, Random Forest, SVM, Decision Tree) and uses MLflow to track experiments, parameters, and metrics.

# Project Overview
The goal is to identify customers at risk of exiting. The pipeline handles:

Data Imbalance: Downsamples the majority class to match the minority class.

Preprocessing: Standard scaling for numerical features and One-Hot encoding for categorical ones.

Experiment Tracking: Logs accuracy, F1-score, confusion matrices, and model artifacts to a local MLflow server.

# Structure
├── data/
│   └── Churn_Modelling.csv   # Source dataset
├── src/
│   └── train.py              # Main training script (Monolithic)
├── mlruns/                   # MLflow local storage
├── README.md
└── requirements.txt
# Setup and Installation
Clone the repository:
git clone <repository-url>
cd <repository-folder>

Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

Install dependencies:
pip install pandas matplotlib scikit-learn mlflow
Usage

Start the MLflow UI:
Open a terminal and run:
mlflow ui
Access the dashboard at http://127.0.0.1:5000.

Run the Training Script:
In a separate terminal (with the environment activated), run:
python src/train.py
This will execute 4 runs (Logistic Regression, Decision Tree, Random Forest, SVM) and log the results to the "churn_models_comparison" experiment.

# Models & Configuration
The script currently tests the following configurations:

Logistic Regression: Standard lbfgs solver.

Decision Tree: Max depth of 5.

Random Forest: 50 trees, max depth of 10.

SVM: RBF kernel.

To modify these, edit the model_configs list in src/train.py.

## Created by Marwan Tamer on 7-2-2026