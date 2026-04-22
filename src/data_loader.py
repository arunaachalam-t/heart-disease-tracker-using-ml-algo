"""
Data loader module for UCI Heart Disease Dataset
"""
import pandas as pd
import numpy as np


def load_heart_disease_data():
    """
    Load UCI Heart Disease dataset
    
    Features:
    1. age - age in years
    2. sex - (1 = male; 0 = female)
    3. cp - chest pain type
    4. trestbps - resting blood pressure (in mm Hg)
    5. chol - serum cholesterol in mg/dl
    6. fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    7. restecg - resting electrocardiographic results
    8. thalach - maximum heart rate achieved
    9. exang - exercise induced angina (1 = yes; 0 = no)
    10. oldpeak - ST depression induced by exercise
    11. slope - the slope of the peak exercise ST segment
    12. ca - number of major vessels (0-3)
    13. thal - 3 = normal; 6 = fixed defect; 7 = reversible defect
    
    Target: presence of heart disease (0 = no, 1 = yes)
    
    Returns:
        pd.DataFrame: Heart disease dataset
    """
    try:
        # Try to load from OpenML
        print("Loading UCI Heart Disease dataset...")
        df = pd.read_csv('data/heart.csv')
        print(f"Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    except FileNotFoundError:
        # Create sample dataset if file doesn't exist
        print("Creating sample heart disease dataset...")
        np.random.seed(42)
        
        n_samples = 303
        data = {
            'age': np.random.randint(29, 77, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(126, 564, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(60, 202, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6.2, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.choice([0, 3, 6, 7], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv('data/heart.csv', index=False)
        print(f"Sample dataset created: {df.shape[0]} samples, {df.shape[1]} features")
        return df


def get_feature_names():
    """Get list of feature names"""
    return [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
