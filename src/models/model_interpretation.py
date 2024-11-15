# src/models/model_interpretation.py

import pandas as pd
import numpy as np
from train_models import load_processed_data
from evaluate_model import load_model
import shap

def compute_shap_values(model, X_sample):
    """
    Berechnet die SHAP-Werte für ein gegebenes Modell und Daten.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    return shap_values

def plot_shap_summary(shap_values, X_sample):
    """
    Plottet die SHAP Summary Plot.
    """
    shap.summary_plot(shap_values, X_sample)

if __name__ == "__main__":
    # Daten laden
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Modell laden
    model = load_model('best_xgboost_model.joblib')
    
    # SHAP-Werte berechnen (mit kleinerem Sample für Performance)
    X_sample = X_test.sample(100, random_state=42)
    shap_values = compute_shap_values(model, X_sample)
    
    # SHAP Summary Plot anzeigen
    plot_shap_summary(shap_values, X_sample)
