# data_utils.py
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_qsar_data(mat_path):
    """
    Loads QSAR biodegradability dataset.
    Returns:
        X : feature matrix (n_samples, n_features)
        y : labels (n_samples,)
    """
    data = loadmat(mat_path)
    qsar = data['QSAR_data']
    
    X = qsar[:, :-1]
    y = qsar[:, -1].astype(int)

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into train and test sets using stratification.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

def scale_features(X_train, X_test):
    """
    Standardizes features using training data statistics.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler