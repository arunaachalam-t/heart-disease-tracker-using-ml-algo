"""
Data preprocessing module for heart disease data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess heart disease dataset
    
    Args:
        df (pd.DataFrame): Raw dataset
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Handle missing values if any
    X = X.fillna(X.mean())
    
    # Remove duplicates
    initial_size = len(X)
    X = X.drop_duplicates()
    y = y.iloc[X.index]
    print(f"Removed {initial_size - len(X)} duplicate rows")
    
    # Handle outliers using IQR method
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
        y = y.iloc[X.index]
    
    print(f"Dataset after preprocessing: {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_data_statistics(df):
    """Get basic statistics about the dataset"""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    print(f"\nTarget distribution (%):\n{df['target'].value_counts(normalize=True) * 100}")
    print(f"\nFeature statistics:\n{df.describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
