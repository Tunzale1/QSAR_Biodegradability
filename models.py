from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression classifier with balanced class weights.
    """
    # initialize logistic regression with increased iterations for convergence
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    
    # fit model to training data
    model.fit(X_train, y_train)
    
    return model


def train_svm(X_train, y_train):
    """
    Trains a Support Vector Machine classifier with RBF kernel.
    """
    # initialize SVM with RBF kernel for non-linear decision boundaries
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )
    
    # fit model to training data
    model.fit(X_train, y_train)
    
    return model


def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest classifier with balanced class weights.
    """
    # initialize Random Forest with multiple decision trees
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    # fit model to training data
    model.fit(X_train, y_train)
    
    return model


def train_gradient_boosting(X_train, y_train):
    """
    Trains a Gradient Boosting classifier with conservative hyperparameters.
    """
    # initialize Gradient Boosting with sequential tree building
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    
    # fit model to training data
    model.fit(X_train, y_train)
    
    return model