# main_run.py
from data_utils import load_qsar_data, split_data, scale_features
from visualization import plot_feature_distribution


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