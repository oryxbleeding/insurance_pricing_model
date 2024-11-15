# src/models/train_models.py

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_processed_data():
    """
    Lädt die verarbeiteten und aufgeteilten Daten.
    """
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_val = pd.read_csv('data/processed/y_val.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, X_train, y_train):
    """
    Trainiert das gegebene Modell mit den Trainingsdaten.
    """
    model.fit(X_train, y_train.values.ravel())
    return model

def evaluate_model(model, X, y_true):
    """
    Bewertet das Modell mit den gegebenen Daten.
    """
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    """
    Führt Hyperparameter-Tuning mit GridSearchCV durch.
    """
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, filename):
    """
    Speichert das trainierte Modell in einer Datei.
    """
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{filename}')
    print(f"Modell gespeichert als 'models/{filename}'")

if __name__ == "__main__":
    # Daten laden
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Modelle importieren
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    
    # Modelle initialisieren
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42)
    }
    
    # Ergebnisse speichern
    results = {}
    
    # Modelle trainieren und bewerten
    for name, model in models.items():
        print(f"Training {name}...")
        model = train_model(model, X_train, y_train)
        mse, mae, r2 = evaluate_model(model, X_val, y_val)
        results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        print(f"{name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    
    # Hyperparameter-Tuning für das beste Modell (angenommen XGBoost)
    print("\nHyperparameter-Tuning für XGBoost...")
    xgb_model = xgb.XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1]
    }
    best_xgb, best_params = hyperparameter_tuning(xgb_model, param_grid, X_train, y_train)
    print(f"Beste Parameter: {best_params}")
    
    # Bewertung des getunten Modells
    mse, mae, r2 = evaluate_model(best_xgb, X_val, y_val)
    results['XGBoost Tuned'] = {'MSE': mse, 'MAE': mae, 'R2': r2}
    print(f"XGBoost Tuned - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    
    # Modell speichern
    save_model(best_xgb, 'best_xgboost_model.joblib')
