import matplotlib.pyplot as plt
import joblib

def plot_feature_importance(model_path="models/hybrid/hybrid.pkl"):

    model = joblib.load(model_path)

    importance = model.feature_importances_

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importance)), importance)
    plt.title("Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.show()