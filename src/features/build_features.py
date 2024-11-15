import pandas as pd
import numpy as np

# Daten laden
data = pd.read_csv('data/raw/insurance_pricing_data.csv')

categorical_vars = ['Gender', 'Occupation', 'Marital_Status', 'Region', 'Vehicle_Type', 'Coverage']
for var in categorical_vars:
    data[var] = data[var].astype('category')

# Einkommen pro Familienmitglied
data['Income_per_Capita'] = data['Income'] / (data['Children'] + 1)

# Verhältnis von Schadenkosten zu gezahlten Prämien
data['Claims_to_Premium_Ratio'] = data['Claims'] / data['Current_Premium']

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
data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# Feature Scaling (Standardisierung)
from sklearn.preprocessing import StandardScaler

numeric_vars = ['Age', 'Income', 'Income_per_Capita', 'Vehicle_Age', 'Mileage', 'Claims', 'Current_Premium', 'Regional_Claim_Rate', 'Economic_Index']
scaler = StandardScaler()
data[numeric_vars] = scaler.fit_transform(data[numeric_vars])

# Verarbeitete Daten speichern
data.to_csv('data/processed/insurance_pricing_data_processed.csv', index=False)