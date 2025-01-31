import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

def train_model():
    print("Loading data...")
    df = pd.read_csv('src/data/ai4i2020.csv')
    
    X_numerical = df[Config.NUMERICAL_FEATURES].copy()
    X_categorical = df[Config.CATEGORICAL_FEATURES].copy()
    
    y = df['Machine failure']
    
    numerical_scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    X_numerical_scaled = numerical_scaler.fit_transform(X_numerical)
    
    X_categorical_encoded = label_encoder.fit_transform(X_categorical['Type'])
    

    X = np.column_stack([X_numerical_scaled, X_categorical_encoded])
    feature_names = Config.NUMERICAL_FEATURES + Config.CATEGORICAL_FEATURES
    X = pd.DataFrame(X, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    

    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=Config.RANDOM_STATE,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Save model and preprocessors
    print("\nSaving model...")
    model_data = {
        'model': model,
        'numerical_scaler': numerical_scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }
    
    joblib.dump(model_data, 'models/predictive_maintenance_model.joblib')
    print("Model saved successfully!")
    
    return model_data

if __name__ == "__main__":
    train_model()