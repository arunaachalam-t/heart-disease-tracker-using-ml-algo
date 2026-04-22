"""
Model training module with multiple regression algorithms
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import joblib
import os


class ModelTrainer:
    """Train and manage multiple regression models"""
    
    def __init__(self, models_dir='models'):
        """
        Initialize ModelTrainer
        
        Args:
            models_dir (str): Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.trained_models = {}
        
    def build_models(self):
        """Build dictionary of regression models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ),
            'Support Vector Regression': SVR(kernel='rbf', C=100, gamma='scale')
        }
        return models
    
    def train_all_models(self, X_train, y_train):
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            dict: Trained models
        """
        models = self.build_models()
        self.trained_models = {}
        
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Save model
            model_path = os.path.join(self.models_dir, f'{name.replace(" ", "_")}.pkl')
            joblib.dump(model, model_path)
            print(f"✓ {name} trained and saved")
        
        return self.trained_models
    
    def save_model(self, name, model):
        """
        Save a trained model
        
        Args:
            name (str): Model name
            model: Trained model object
        """
        model_path = os.path.join(self.models_dir, f'{name.replace(" ", "_")}.pkl')
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")
    
    def load_model(self, name):
        """
        Load a trained model
        
        Args:
            name (str): Model name
            
        Returns:
            Trained model object
        """
        model_path = os.path.join(self.models_dir, f'{name.replace(" ", "_")}.pkl')
        model = joblib.load(model_path)
        print(f"Model loaded: {model_path}")
        return model
    
    def predict(self, model, X):
        """
        Make predictions using a model
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            np.array: Predictions
        """
        return model.predict(X)
    
    def get_trained_models(self):
        """Get all trained models"""
        return self.trained_models
