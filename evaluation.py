import numpy as np
import shap
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on test data.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred)
    }


def get_feature_importance(model, top_n=10):
    """
    Extracts top N most important features from a tree-based model.
    """
    # get importance scores from the model
    importances = model.feature_importances_
    
    # sort indices by importance in descending order and select top N
    indices = np.argsort(importances)[::-1][:top_n]
    
    return indices, importances[indices]


def compute_shap_values(model, X_background, X_explain, max_samples=200):
    """
    Computes SHAP values for tree-based models in a robust way.
    """
    # initialize SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)

    # limit the number of samples to explain for computational efficiency
    X_subset = X_explain[:max_samples]

    # compute SHAP values for the subset
    shap_values = explainer.shap_values(X_subset)

    # handle different SHAP output formats based on problem type
    if isinstance(shap_values, list):
        # binary classification case → SHAP returns a list with values for each class
        # take class 1 (biodegradable) for interpretation
        shap_values = shap_values[1]

    return shap_values, X_subset