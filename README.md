# QSAR Biodegradability Prediction – Data Modelling and Machine Intelligence

## Overview
This project investigates the prediction of chemical biodegradability using Quantitative Structure–Activity Relationship (QSAR) descriptors and machine learning techniques. The work was completed as part of the *Data Modelling and Machine Intelligence* coursework and focuses on model comparison, validation, interpretability, and critical analysis rather than pure performance optimisation.

The dataset consists of numerical molecular descriptors representing chemical compounds, with a binary label indicating whether each compound is biodegradable or non-biodegradable.

---

## Project Structure
The project is organised in a modular and reproducible manner:

QSAR_Biodegradability/
│
├── main_run.py # Main script to run the entire pipeline
├── data_utils.py # Data loading, splitting, and preprocessing
├── models.py # Model training functions
├── evaluation.py # Model evaluation and SHAP computation
├── visualization.py # Plotting utilities
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── *.png # Generated figures (used in the report)


---

## Dataset
The QSAR dataset contains:
- **1055 chemical compounds**
- **41 numerical QSAR descriptors**
- **Binary target variable**
  - `1` – biodegradable  
  - `0` – non-biodegradable  

The dataset is imbalanced, with a higher proportion of non-biodegradable compounds. Stratified sampling and balanced evaluation metrics were therefore employed.

---

## Methodology
The following machine learning models were evaluated:

- Logistic Regression (baseline linear model)
- Support Vector Machine (RBF kernel)
- Random Forest (ensemble, bagging-based)
- Gradient Boosting (ensemble, boosting-based)

Data preprocessing included stratified train–test splitting and feature standardization using z-score normalization. Model performance was assessed using both accuracy and balanced accuracy metrics. Model interpretability was enhanced through Random Forest feature importance analysis and SHAP (SHapley Additive exPlanations).

---

## How to Run the Code
1. Ensure Python 3.9+ is installed.
2. (Optional but recommended) Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the main script
    ```powershell
    python main_run.py