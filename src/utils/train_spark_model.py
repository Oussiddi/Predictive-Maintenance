import os
import sys
from pathlib import Path

src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from models.spark_model import SparkPredictiveMaintenanceModel
from config.config import Config

def train_spark_model():
    print("Initializing Spark model...")
    model = SparkPredictiveMaintenanceModel(Config)
    
    print("Starting model training...")
    trained_model = model.train('src/data/ai4i2020.csv')
    
    print("Saving model...")
    model.save_model('models/spark_predictive_maintenance_model')
    
    model.stop_spark()
    
    print("Model training complete and saved!")

if __name__ == "__main__":
    print("Starting training process...")
    train_spark_model() 