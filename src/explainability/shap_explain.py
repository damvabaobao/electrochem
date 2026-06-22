import shap
import joblib

def explain(X):

    model = joblib.load("models/hybrid/hybrid.pkl")

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X)