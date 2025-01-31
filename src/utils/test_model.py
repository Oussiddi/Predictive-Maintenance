import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config
from models.model import PredictiveMaintenanceModel
from utils.preprocessing import DataPreprocessor, validate_data

def test_model():
    os.makedirs('models', exist_ok=True)
    
    print("Loading dataset...")
    df = pd.read_csv('src/data/ai4i2020.csv')
    
    print("\nData shape:", df.shape)
    print("\nFailure distribution:")
    print(df[Config.FAILURE_COLUMN].value_counts())
    
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    
    X_raw = df[Config.NUMERICAL_FEATURES + Config.CATEGORICAL_FEATURES].copy()
    y = df[Config.FAILURE_COLUMN]
    
    label_encoder = LabelEncoder()
    X_raw['Type'] = label_encoder.fit_transform(X_raw['Type'])
    
    numerical_features = Config.NUMERICAL_FEATURES
    preprocessor.fit(X_raw, numerical_features, [])
    X_processed = preprocessor.transform(X_raw, numerical_features, [])
    
    print("\nInitializing model...")
    model = PredictiveMaintenanceModel(Config)
    model.preprocessor = preprocessor
    
    print("Training model...")
    evaluation = model.train(X_processed, y)
    
    print("\nModel Evaluation:")
    print("\nClassification Report:")
    print(evaluation['classification_report'])
    
    print("\nFeature Importance:")
    for feature, importance in sorted(
        evaluation['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{feature}: {importance:.4f}")
    
    print(f"\nTrain Accuracy: {evaluation['train_accuracy']:.4f}")
    print(f"Test Accuracy: {evaluation['test_accuracy']:.4f}")
    
    print("\nFailure Type Distribution:")
    for failure_type in Config.TARGET_COLUMNS:
        print(f"{failure_type}: {df[failure_type].sum()} occurrences")
    
    model.label_encoder = label_encoder
    model.save('models/predictive_maintenance_model.joblib')
    
    return model, evaluation

if __name__ == "__main__":
    test_model()