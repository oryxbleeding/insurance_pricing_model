# src/models/evaluate_model.py

import pandas as pd
import numpy as np
import os
from train_models import load_processed_data, evaluate_model
import joblib

def load_model(filename):
    """
    Lädt das gespeicherte Modell.
    """
    model = joblib.load(f'models/{filename}')
    return model

def plot_performance(y_true, y_pred, title):
    """
    Plottet die tatsächlichen vs. vorhergesagten Werte.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Tatsächliche Tarife')
    plt.ylabel('Vorhergesagte Tarife')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Daten laden
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Modell laden
    model = load_model('best_xgboost_model.joblib')
    
    # Bewertung auf Testdaten
    mse, mae, r2 = evaluate_model(model, X_test, y_test)
    print(f"Testdaten - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    
    # Plotten der Performance
    y_test_pred = model.predict(X_test)
    plot_performance(y_test, y_test_pred, 'Modellleistung auf Testdaten')
