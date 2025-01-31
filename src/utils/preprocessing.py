import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        
    def fit(self, data: pd.DataFrame, numerical_features: list, categorical_features: list) -> None:
        for feature in numerical_features:
            self.scalers[feature] = StandardScaler()
            self.scalers[feature].fit(data[feature].values.reshape(-1, 1))
        
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            self.label_encoders[feature].fit(data[feature])
    
    def transform(self, data: pd.DataFrame, numerical_features: list, 
                 categorical_features: list) -> pd.DataFrame:
        df = data.copy()
        
        for feature in numerical_features:
            df[feature] = self.scalers[feature].transform(
                df[feature].values.reshape(-1, 1)
            )
        
        for feature in categorical_features:
            df[feature] = self.label_encoders[feature].transform(df[feature])
        
        return df

def validate_data(data: pd.DataFrame, config: Any) -> bool:
    required_columns = (
        config.NUMERICAL_FEATURES + 
        config.CATEGORICAL_FEATURES
    )
    
    if not all(col in data.columns for col in required_columns):
        print("Missing required columns")
        return False
    
    if data[required_columns].isnull().any().any():
        print("Found missing values")
        return False
    
    try:
        for col in config.NUMERICAL_FEATURES:
            pd.to_numeric(data[col])
        
        for col in config.CATEGORICAL_FEATURES:
            if not data[col].dtype in ['object', 'string', 'category']:
                print(f"Invalid data type for {col}")
                return False
    except Exception as e:
        print(f"Data validation error: {str(e)}")
        return False
    
    return True