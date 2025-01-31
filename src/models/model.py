import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, List, Tuple, Any

class PredictiveMaintenanceModel:
    def __init__(self, config: Any):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1  
        )
        self.preprocessor = None
        self.label_encoder = None
        self.feature_importance = None
        
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        feature_names = (
            self.config.NUMERICAL_FEATURES + 
            self.config.CATEGORICAL_FEATURES
        )
        self.feature_importance = dict(
            zip(feature_names, self.model.feature_importances_)
        )
        
        y_pred = self.model.predict(X_test)
        
        evaluation = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self.feature_importance,
            'test_accuracy': self.model.score(X_test, y_test),
            'train_accuracy': self.model.score(X_train, y_train)
        }
        
        return evaluation
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def save(self, path: str) -> None:
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance,
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        components = joblib.load(path)
        self.model = components['model']
        self.preprocessor = components['preprocessor']
        self.label_encoder = components['label_encoder']
        self.feature_importance = components['feature_importance']
        self.config = components['config']
        print(f"Model loaded from {path}")