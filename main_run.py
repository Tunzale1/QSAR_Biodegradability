# data handling utilities
from data_utils import load_qsar_data, split_data, scale_features

# visualization functions
from visualization import plot_feature_distribution, plot_feature_importance
from visualization import plot_shap_summary

# model training functions
from models import train_logistic_regression, train_svm, train_random_forest, train_gradient_boosting

# model evaluation and interpretation functions
from evaluation import evaluate_model, get_feature_importance, compute_shap_values


# load QSAR biodegradability dataset
X, y = load_qsar_data("QSAR_data.mat")

# split data into training and test sets
X_train, X_test, y_train, y_test = split_data(X, y)

# display dataset information
print("X shape:", X.shape)
print("y shape:", y.shape)
print(
    "Class distribution:",
    {0: int((y == 0).sum()), 1: int((y == 1).sum())}
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# standardize features (mean=0, std=1) for linear models
X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

# verify scaling was applied correctly
print("Feature mean (train):", X_train_s.mean())
print("Feature std (train):", X_train_s.std())


# plot original feature distribution (before scaling)
plot_feature_distribution(
    X_train,
    feature_idx=0,
    title="Feature 1 Distribution (Original)",
    filename="feature1_original.png"
)

# plot scaled feature distribution (after standardization)
plot_feature_distribution(
    X_train_s,
    feature_idx=0,
    title="Feature 1 Distribution (Standardized)",
    filename="feature1_scaled.png"
)


# train logistic regression on scaled features
log_reg = train_logistic_regression(X_train_s, y_train)

# evaluate model performance
log_results = evaluate_model(log_reg, X_test_s, y_test)
print("Logistic Regression Results:", log_results)


# train SVM on scaled features
svm_model = train_svm(X_train_s, y_train)

# evaluate model performance
svm_results = evaluate_model(svm_model, X_test_s, y_test)
print("SVM Results:", svm_results)


# train Random Forest on original (unscaled) features
rf_model = train_random_forest(X_train, y_train)

# evaluate model performance
rf_results = evaluate_model(rf_model, X_test, y_test)
print("Random Forest Results:", rf_results)

# extract and visualize top 10 most important features
top_idx, top_imp = get_feature_importance(rf_model, top_n=10)
plot_feature_importance(top_idx, top_imp, "rf_feature_importance.png")
print("Top 10 important feature indices:", top_idx)


# train Gradient Boosting on original (unscaled) features
gb_model = train_gradient_boosting(X_train, y_train)

# evaluate model performance
gb_results = evaluate_model(gb_model, X_test, y_test)
print("Gradient Boosting Results:", gb_results)


# compute SHAP values to explain Random Forest predictions
# SHAP values show how each feature contributes to individual predictions
shap_values, X_shap = compute_shap_values(
    rf_model,
    X_train,
    X_test
)

# generate summary plot showing feature impact on predictions
plot_shap_summary(shap_values, X_shap, "shap_summary_rf.png")
print("SHAP summary plot saved.")