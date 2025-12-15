import shap
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred)
    }

def get_feature_importance(model, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return indices, importances[indices]

def compute_shap_values(model, X_background, X_explain, max_samples=200):
    """
    Computes SHAP values for tree-based models in a robust way.
    """
    explainer = shap.TreeExplainer(model)

    X_subset = X_explain[:max_samples]

    shap_values = explainer.shap_values(X_subset)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Binary classification → take class 1 (biodegradable)
        shap_values = shap_values[1]

    return shap_values, X_subset