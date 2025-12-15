# main_run.py
from data_utils import load_qsar_data, split_data , scale_features
from visualization import plot_feature_distribution , plot_feature_importance
from models import train_logistic_regression , train_svm , train_random_forest
from evaluation import evaluate_model , get_feature_importance


X, y = load_qsar_data("QSAR_data.mat")
X_train, X_test, y_train, y_test = split_data(X, y)


print("X shape:", X.shape)
print("y shape:", y.shape)
print(
    "Class distribution:",
    {0: int((y == 0).sum()), 1: int((y == 1).sum())}
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

print("Feature mean (train):", X_train_s.mean())
print("Feature std (train):", X_train_s.std())

plot_feature_distribution(
    X_train,
    feature_idx=0,
    title="Feature 1 Distribution (Original)",
    filename="feature1_original.png"
)

plot_feature_distribution(
    X_train_s,
    feature_idx=0,
    title="Feature 1 Distribution (Standardized)",
    filename="feature1_scaled.png"
)

log_reg = train_logistic_regression(X_train_s, y_train)
log_results = evaluate_model(log_reg, X_test_s, y_test)

print("Logistic Regression Results:", log_results)

svm_model = train_svm(X_train_s, y_train)
svm_results = evaluate_model(svm_model, X_test_s, y_test)

print("SVM Results:", svm_results)

rf_model = train_random_forest(X_train, y_train)
rf_results = evaluate_model(rf_model, X_test, y_test)

print("Random Forest Results:", rf_results)

top_idx, top_imp = get_feature_importance(rf_model, top_n=10)
plot_feature_importance(top_idx, top_imp, "rf_feature_importance.png")

print("Top 10 important feature indices:", top_idx)