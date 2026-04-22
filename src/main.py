"""
Main application for Heart Disease Tracker ML System
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_heart_disease_data, get_feature_names
from preprocessing import preprocess_data, get_data_statistics
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
import pandas as pd


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("HEART DISEASE TRACKER - ML PREDICTION SYSTEM")
    print("="*60)
    
    # Step 1: Load Data
    print("\n[STEP 1] Loading Dataset...")
    df = load_heart_disease_data()
    
    # Step 2: Data Statistics
    print("\n[STEP 2] Analyzing Dataset...")
    get_data_statistics(df)
    
    # Step 3: Preprocessing
    print("\n[STEP 3] Preprocessing Data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Step 4: Train Models
    print("\n[STEP 4] Training Models...")
    trainer = ModelTrainer(models_dir='models')
    models = trainer.train_all_models(X_train, y_train)
    
    # Step 5: Evaluate Models
    print("\n[STEP 5] Evaluating Models...")
    evaluator = ModelEvaluator()
    results_df = evaluator.evaluate_models(models, X_test, y_test)
    
    # Step 6: Get Best Model
    print("\n[STEP 6] Identifying Best Model...")
    best_model_name = evaluator.get_best_model(results_df)
    best_model = models[best_model_name]
    
    # Step 7: Visualizations
    print("\n[STEP 7] Generating Visualizations...")
    evaluator.plot_model_comparison(results_df)
    evaluator.plot_predictions(best_model, X_test, y_test, best_model_name)
    
    # Step 8: Sample Predictions
    print("\n[STEP 8] Sample Predictions")
    print("-" * 60)
    sample_data = X_test.iloc[:5]
    predictions = best_model.predict(sample_data)
    print("\nFirst 5 Test Samples:")
    for i, (idx, row) in enumerate(sample_data.iterrows()):
        print(f"\nSample {i+1}:")
        print(f"  Actual Heart Disease Risk: {y_test.iloc[i]:.4f}")
        print(f"  Predicted Risk: {predictions[i]:.4f}")
        print(f"  Absolute Error: {abs(y_test.iloc[i] - predictions[i]):.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Samples: {len(df)}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Number of Features: {X_train.shape[1]}")
    print(f"Number of Models Trained: {len(models)}")
    print(f"Best Performing Model: {best_model_name}")
    print(f"Best Model R² Score: {results_df.iloc[0]['R² Score']:.4f}")
    print(f"Best Model RMSE: {results_df.iloc[0]['RMSE']:.4f}")
    print(f"Best Model MAE: {results_df.iloc[0]['MAE']:.4f}")
    
    print("\n✓ All models saved in 'models/' directory")
    print("✓ Visualizations saved in 'models/' directory")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
