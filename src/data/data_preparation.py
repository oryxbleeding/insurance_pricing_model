# src/data/data_preparation.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Lädt den Datensatz aus der angegebenen Datei.
    """
    data = pd.read_csv(filepath)
    return data

def clean_data(data):
    """
    Führt die Datenbereinigung durch.
    """
    # Fehlende Werte behandeln (falls vorhanden)
    data = data.dropna()
    
    # Ausreißerbehandlung kann hier hinzugefügt werden
    
    # Datentypen anpassen
    categorical_vars = ['Gender', 'Occupation', 'Marital_Status', 'Region', 'Vehicle_Type', 'Coverage']
    for var in categorical_vars:
        data[var] = data[var].astype('category')
    
    return data

def feature_engineering(data):
    """
    Erstellt neue Features und kodiert kategoriale Variablen.
    """
    # Einkommen pro Familienmitglied
    data['Income_per_Capita'] = data['Income'] / (data['Children'] + 1)
    
    # Verhältnis von Schadenkosten zu gezahlten Prämien
    data['Claims_to_Premium_Ratio'] = data['Claims'] / data['Current_Premium']
    data['Claims_to_Premium_Ratio'] = data['Claims_to_Premium_Ratio'].fillna(0)
    
    # Risikoklassen basierend auf Beruf
    risk_mapping = {
        'Unemployed': 3,
        'Artist': 2,
        'Teacher': 2,
        'Engineer': 1,
        'Doctor': 1,
        'Lawyer': 1
    }
    data['Occupation_Risk'] = data['Occupation'].map(risk_mapping)
    
    # Kategoriale Variablen kodieren
    categorical_vars = ['Gender', 'Marital_Status', 'Region', 'Vehicle_Type', 'Coverage']
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)
    
    # Label-Encoding für 'Occupation_Risk' (da bereits numerisch)
    data['Occupation_Risk'] = data['Occupation_Risk'].astype(int)
    
    return data

def feature_scaling(data, scaler=None):
    """
    Skaliert numerische Features. Wenn kein Skaler angegeben ist, wird StandardScaler verwendet.
    """
    from sklearn.preprocessing import StandardScaler
    
    numeric_vars = ['Age', 'Income', 'Income_per_Capita', 'Vehicle_Age', 'Mileage', 'Claims',
                    'Current_Premium', 'Regional_Claim_Rate', 'Economic_Index', 'Occupation_Risk']
    
    if scaler is None:
        scaler = StandardScaler()
        data[numeric_vars] = scaler.fit_transform(data[numeric_vars])
        return data, scaler
    else:
        data[numeric_vars] = scaler.transform(data[numeric_vars])
        return data

def split_data(data, test_size=0.15, val_size=0.15, random_state=42):
    """
    Teilt die Daten in Trainings-, Validierungs- und Testmengen auf.
    """
    X = data.drop('Tariff', axis=1)
    y = data['Tariff']
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size_adjusted, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Speichert die aufgeteilten Daten in Dateien.
    """
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_val.to_csv('data/processed/X_val.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

if __name__ == "__main__":
    # Daten laden
    data = load_data('data/raw/insurance_pricing_data.csv')
    
    # Daten bereinigen
    data = clean_data(data)
    
    # Feature Engineering
    data = feature_engineering(data)
    
    # Feature Scaling
    data, scaler = feature_scaling(data)
    
    # Daten aufteilen
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    
    # Daten speichern
    save_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("Datenvorbereitung abgeschlossen und Daten gespeichert.")
