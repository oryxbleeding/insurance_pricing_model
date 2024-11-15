# src/data/generate_synthetic_data.py

import numpy as np
import pandas as pd
import os

def generate_synthetic_data(n_samples=10000, random_state=42):
    """
    Generiert einen synthetischen Datensatz für die Versicherungstarifmodellierung.
    """
    np.random.seed(random_state)
    
    # Kundendaten
    age = np.random.randint(18, 80, n_samples)
    gender = np.random.choice(['male', 'female'], n_samples)
    occupation = np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Artist', 'Unemployed'], n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    income = np.clip(income, 10000, None)
    marital_status = np.random.choice(['single', 'married', 'divorced'], n_samples)
    children = np.random.poisson(1, n_samples)
    region = np.random.choice(['urban', 'suburban', 'rural'], n_samples)
    
    # Fahrzeugdaten
    vehicle_type = np.random.choice(['Sedan', 'SUV', 'Truck', 'Sports Car', 'Van'], n_samples)
    vehicle_age = np.random.randint(0, 20, n_samples)
    mileage = np.random.normal(50000, 30000, n_samples)
    mileage = np.clip(mileage, 0, None)
    
    # Versicherungsdaten
    current_premium = np.random.normal(800, 200, n_samples)
    current_premium = np.clip(current_premium, 200, None)
    coverage = np.random.choice(['Basic', 'Standard', 'Premium'], n_samples)
    claims = np.random.poisson(0.2, n_samples)
    
    # Externe Daten
    regional_claim_rate = np.random.uniform(0.05, 0.2, n_samples)
    economic_index = np.random.normal(100, 10, n_samples)
    
    # Tarifberechnung (synthetisch)
    coverage_factor_mapping = {'Basic': 1.0, 'Standard': 1.2, 'Premium': 1.5}
    coverage_factor = np.array([coverage_factor_mapping[c] for c in coverage])
    
    base_premium = 500
    age_factor = (80 - age) * 2
    income_factor = income * 0.0005
    vehicle_age_factor = vehicle_age * 15
    claims_factor = claims * 200
    tariff = (base_premium + age_factor + income_factor + vehicle_age_factor + claims_factor) * coverage_factor
    tariff = np.clip(tariff, 300, None)
    
    # DataFrame erstellen
    data = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Occupation': occupation,
        'Income': income,
        'Marital_Status': marital_status,
        'Children': children,
        'Region': region,
        'Vehicle_Type': vehicle_type,
        'Vehicle_Age': vehicle_age,
        'Mileage': mileage,
        'Current_Premium': current_premium,
        'Coverage': coverage,
        'Claims': claims,
        'Regional_Claim_Rate': regional_claim_rate,
        'Economic_Index': economic_index,
        'Tariff': tariff
    })
    
    return data

if __name__ == "__main__":
    data = generate_synthetic_data()
    
    # Verzeichnis erstellen, falls nicht vorhanden
    os.makedirs('data/raw', exist_ok=True)
    
    # Daten speichern
    data.to_csv('data/raw/insurance_pricing_data.csv', index=False)
    print("Datensatz 'insurance_pricing_data.csv' wurde erfolgreich erstellt.")
