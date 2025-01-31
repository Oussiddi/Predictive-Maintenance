import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config
from utils.preprocessing import DataPreprocessor, validate_data

def test_preprocessing():
    print("Loading dataset...")
    df = pd.read_csv('src/data/ai4i2020.csv')
    
    print("\nInitial data shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    print("\nValidating data...")
    if validate_data(df, Config):
        print("Data validation successful!")
    else:
        print("Data validation failed!")
        return
    
    print("\nInitializing preprocessor...")
    preprocessor = DataPreprocessor()
    
    print("Fitting preprocessor...")
    preprocessor.fit(df, Config.NUMERICAL_FEATURES, Config.CATEGORICAL_FEATURES)
    
    print("Transforming data...")
    transformed_data = preprocessor.transform(
        df, 
        Config.NUMERICAL_FEATURES,
        Config.CATEGORICAL_FEATURES
    )
    
    print("\nSample of transformed data:")
    print(transformed_data.head())
    
    print("\nTransformed numerical features statistics:")
    print(transformed_data[Config.NUMERICAL_FEATURES].describe())
    
    return preprocessor, transformed_data

if __name__ == "__main__":
    test_preprocessing()