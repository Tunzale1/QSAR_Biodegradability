# main_run.py
from data_utils import load_qsar_data, split_data

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
