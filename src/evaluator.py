"""
Model evaluation and comparison module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, explained_variance_score
)


class ModelEvaluator:
    """Evaluate and compare regression models"""
    
    def __init__(self):
        """Initialize ModelEvaluator"""
        self.results = {}
        sns.set_style("whitegrid")
        
    def evaluate_models(self, models, X_test, y_test):
        """
        Evaluate all models and store results
        
        Args:
            models (dict): Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            pd.DataFrame: Evaluation results
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        results = []
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            medae = median_absolute_error(y_test, y_pred)
            evs = explained_variance_score(y_test, y_pred)
            
            result = {
                'Model': name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2,
                'Median AE': medae,
                'Explained Variance': evs
            }
            results.append(result)
            self.results[name] = result
            
            print(f"\n{name}")
            print("-" * 40)
            print(f"  MSE (Mean Squared Error): {mse:.4f}")
            print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
            print(f"  MAE (Mean Absolute Error): {mae:.4f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  Median Absolute Error: {medae:.4f}")
            print(f"  Explained Variance Score: {evs:.4f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by R² Score (descending)
        results_df = results_df.sort_values('R² Score', ascending=False)
        
        print("\n" + "="*50)
        print("SORTED RESULTS (by R² Score)")
        print("="*50)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def get_best_model(self, results_df):
        """
        Get the best model based on R² Score
        
        Args:
            results_df (pd.DataFrame): Evaluation results
            
        Returns:
            str: Name of best model
        """
        best_model = results_df.iloc[0]['Model']
        best_r2 = results_df.iloc[0]['R² Score']
        print(f"\n✓ Best Model: {best_model}")
        print(f"✓ Best R² Score: {best_r2:.4f}")
        return best_model
    
    def plot_model_comparison(self, results_df, save_path='models/model_comparison.png'):
        """
        Plot model performance comparison
        
        Args:
            results_df (pd.DataFrame): Evaluation results
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # R² Score comparison
        ax1 = axes[0, 0]
        results_df_sorted = results_df.sort_values('R² Score', ascending=True)
        ax1.barh(results_df_sorted['Model'], results_df_sorted['R² Score'], color='skyblue')
        ax1.set_xlabel('R² Score')
        ax1.set_title('R² Score Comparison')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # RMSE comparison
        ax2 = axes[0, 1]
        results_df_sorted = results_df.sort_values('RMSE')
        ax2.barh(results_df_sorted['Model'], results_df_sorted['RMSE'], color='lightcoral')
        ax2.set_xlabel('RMSE')
        ax2.set_title('RMSE Comparison (Lower is Better)')
        
        # MAE comparison
        ax3 = axes[1, 0]
        results_df_sorted = results_df.sort_values('MAE')
        ax3.barh(results_df_sorted['Model'], results_df_sorted['MAE'], color='lightgreen')
        ax3.set_xlabel('MAE')
        ax3.set_title('MAE Comparison (Lower is Better)')
        
        # Explained Variance comparison
        ax4 = axes[1, 1]
        results_df_sorted = results_df.sort_values('Explained Variance', ascending=True)
        ax4.barh(results_df_sorted['Model'], results_df_sorted['Explained Variance'], color='lightyellow')
        ax4.set_xlabel('Explained Variance Score')
        ax4.set_title('Explained Variance Comparison')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved: {save_path}")
        plt.close()
    
    def plot_predictions(self, model, X_test, y_test, model_name, save_path='models/predictions.png'):
        """
        Plot actual vs predicted values
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        y_pred = model.predict(X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{model_name} - Predictions Analysis', fontsize=14, fontweight='bold')
        
        # Actual vs Predicted
        ax1 = axes[0]
        ax1.scatter(y_test, y_pred, alpha=0.6, color='blue')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        ax2 = axes[1]
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Predictions plot saved: {save_path}")
        plt.close()
    
    def get_results_summary(self):
        """Get summary of evaluation results"""
        return self.results
