import matplotlib.pyplot as plt

def plot_feature_distribution(X, feature_idx, title, filename):
    plt.figure(figsize=(6,4))
    plt.hist(X[:, feature_idx], bins=40, edgecolor='black')
    plt.title(title)
    plt.xlabel("Feature value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_feature_importance(indices, importances, filename):
    plt.figure(figsize=(7,4))
    plt.bar(range(len(importances)), importances)
    plt.xticks(range(len(importances)), indices)
    plt.xlabel("Feature index")
    plt.ylabel("Importance")
    plt.title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()    
