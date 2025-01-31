from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import json
import time
from kafka import KafkaProducer
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

class DataSimulator:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("DataSimulator") \
            .getOrCreate()
            
        
        df = self.spark.read.csv('src/data/ai4i2020.csv', header=True, inferSchema=True)
        pdf = df.toPandas()
        
        self.feature_stats = {}
        for feature in Config.NUMERICAL_FEATURES:
            stats = df.select(feature).summary().collect()
            self.feature_stats[feature] = {
                'mean': float(stats[1][1]),
                'std': float(stats[2][1]),
                'min': float(stats[3][1]),
                'max': float(stats[4][1])
            }
        
        self.type_values = [row[0] for row in df.select('Type').distinct().collect()]
        
    def generate_sample(self, add_anomaly=False):
        sample = {}
        
        for feature, stats in self.feature_stats.items():
            if add_anomaly and np.random.random() < 0.3:
                value = stats['mean'] + stats['std'] * np.random.uniform(3, 5)
            else:
                value = np.random.normal(stats['mean'], stats['std'])
                value = np.clip(value, stats['min'], stats['max'])
            sample[feature] = float(value)
        
        sample['Type'] = str(np.random.choice(self.type_values))
        sample['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return sample
        
    def stop_spark(self):
        self.spark.stop()